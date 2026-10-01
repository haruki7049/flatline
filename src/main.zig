const std = @import("std");

// Decodes a sequence of RVQ frames into a contiguous latent matrix [128, T]
pub fn decodeRVQSequence(
    codebooks: []const [*]const f32,
    tokens: []const [8]u16,
    num_frames: usize,
    out_latents: []f32,
) void {
    for (0..128) |c| {
        for (0..num_frames) |t| {
            out_latents[c * num_frames + t] = 0.0;
        }
    }

    for (0..num_frames) |t| {
        const frame_tokens = tokens[t];
        for (frame_tokens, 0..) |token, stage| {
            const base_ptr = codebooks[stage] + (@as(usize, token) * 128);
            for (0..128) |c| {
                out_latents[c * num_frames + t] += base_ptr[c];
            }
        }
    }
}

// 1D Convolution with precomputed normalized weights
// Input:  [C_in, T_in]
// Weight: [C_out, C_in, K]
// Bias:   [C_out]
// Output: [C_out, T_out] where T_out = T_in (with padding = K / 2)
pub fn conv1dSame(
    in: []const f32,
    c_in: usize,
    t_len: usize,
    weight: []const f32,
    bias: []const f32,
    c_out: usize,
    kernel_size: usize,
    out: []f32,
) void {
    const pad = kernel_size / 2;

    for (0..c_out) |co| {
        const bias_val = bias[co];
        const w_co = weight[co * (c_in * kernel_size) .. (co + 1) * (c_in * kernel_size)];

        for (0..t_len) |t| {
            var sum: f32 = bias_val;

            for (0..c_in) |ci| {
                const in_channel = in[ci * t_len .. (ci + 1) * t_len];
                const w_channel = w_co[ci * kernel_size .. (ci + 1) * kernel_size];

                for (0..kernel_size) |k| {
                    const in_t = @as(isize, @intCast(t)) + @as(isize, @intCast(k)) - @as(isize, @intCast(pad));
                    if (in_t >= 0 and in_t < @as(isize, @intCast(t_len))) {
                        sum += in_channel[@as(usize, @intCast(in_t))] * w_channel[k];
                    }
                }
            }

            out[co * t_len + t] = sum;
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const file_path = "src/weights/encodec_24khz.safetensors";

    // 1. Open weight file
    const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    const file_size = (try file.stat(io)).size;
    if (file_size < 8) return error.InvalidFile;

    // 2. Map file into memory space (zero-copy)
    const mapped = try std.posix.mmap(
        null,
        file_size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mapped);

    // 3. Slice out header JSON
    const header_len = std.mem.readInt(u64, mapped[0..8], .little);
    if (header_len > mapped.len - 8) return error.InvalidFile;
    const header_json = mapped[8 .. 8 + header_len];

    var args = init.minimal.args;
    var args_iterator = args.iterate();
    _ = args_iterator.next();
    if (args_iterator.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump-header")) {
            var buffer: [1024]u8 = undefined;
            var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
            const stdout = &stdout_writer.interface;
            try stdout.writeAll(header_json);
            try stdout.writeByte('\n');
            try stdout.flush();
            return;
        }
    }

    const payload_start = 8 + header_len;

    // 4. Parse JSON metadata
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        header_json,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value.object;

    // Get RVQ codebooks
    const NUM_STAGES = 8;
    var codebooks: [NUM_STAGES][*]const f32 = undefined;
    for (0..NUM_STAGES) |stage| {
        var key_buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "quantizer.layers.{d}.codebook.embed", .{stage});
        const tensor_entry = root.get(key) orelse return error.TensorNotFound;
        const offsets = tensor_entry.object.get("data_offsets") orelse return error.InvalidFormat;
        const start_offset = @as(usize, @intCast(offsets.array.items[0].integer));
        codebooks[stage] = @ptrCast(@alignCast(mapped[payload_start + start_offset ..].ptr));
    }

    // Get decoder.layers.0.conv weights
    const get_offset = struct {
        fn run(r: anytype, k: []const u8) !usize {
            const entry = r.get(k) orelse return error.TensorNotFound;
            const offsets = entry.object.get("data_offsets") orelse return error.InvalidFormat;
            return @as(usize, @intCast(offsets.array.items[0].integer));
        }
    }.run;

    const bias_offset = try get_offset(root, "decoder.layers.0.conv.bias");
    const weight_v_offset = try get_offset(root, "decoder.layers.0.conv.weight_v");
    const weight_g_offset = try get_offset(root, "decoder.layers.0.conv.weight_g");

    const bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + bias_offset ..].ptr));
    const weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + weight_v_offset ..].ptr));
    const weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + weight_g_offset ..].ptr));

    // Normalize weights: W = g * (v / ||v||)
    const C_OUT = 512;
    const C_IN = 128;
    const K = 7;
    const weight_norm = try allocator.alloc(f32, C_OUT * C_IN * K);

    for (0..C_OUT) |co| {
        var norm_sq: f32 = 0.0;
        const slice_len = C_IN * K;
        const base = co * slice_len;

        for (0..slice_len) |idx| {
            const val = weight_v[base + idx];
            norm_sq += val * val;
        }

        const scale = weight_g[co] / @sqrt(norm_sq);
        for (0..slice_len) |idx| {
            weight_norm[base + idx] = weight_v[base + idx] * scale;
        }
    }

    // 5. Decode tokens into latents [128, 4]
    var sequence_tokens = [_][8]u16{
        [_]u16{ 120, 450, 89, 730, 210, 95, 600, 314 },
        [_]u16{ 121, 448, 88, 729, 212, 94, 599, 310 },
        [_]u16{ 125, 440, 92, 735, 205, 99, 605, 320 },
        [_]u16{ 130, 435, 95, 740, 200, 102, 610, 325 },
    };
    const num_frames = sequence_tokens.len;

    var latents: [128 * 4]f32 = undefined;
    decodeRVQSequence(&codebooks, &sequence_tokens, num_frames, &latents);

    // 6. Run Conv1d: [128, 4] -> [512, 4]
    const conv_out = try allocator.alloc(f32, C_OUT * num_frames);
    conv1dSame(&latents, C_IN, num_frames, weight_norm, bias[0..C_OUT], C_OUT, K, conv_out);

    // 7. Inspect output
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Successfully computed Conv1d layer 0 ([128, {d}] -> [512, {d}]).\n", .{ num_frames, num_frames });
    try stdout.print("Output channel 0 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{conv_out[0 * num_frames + t]});
    }
    try stdout.print("\nOutput channel 1 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{conv_out[1 * num_frames + t]});
    }
    try stdout.writeByte('\n');
    try stdout.flush();
}
