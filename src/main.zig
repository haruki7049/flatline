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

fn sigmoid(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return 0.0;
    return 1.0 / (1.0 + @exp(-x));
}

fn tanhActivation(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return -1.0;
    return std.math.tanh(x);
}

// Unidirectional LSTM across time dimension
// in: [H, T], out: [H, T] where H = 512
pub fn runLstmLayer(
    in: []const f32,
    out: []f32,
    h_dim: usize,
    t_len: usize,
    w_ih: [*]const f32,
    w_hh: [*]const f32,
    b_ih: [*]const f32,
    b_hh: [*]const f32,
    h_state: []f32,
    c_state: []f32,
) void {
    @memset(h_state, 0.0);
    @memset(c_state, 0.0);

    for (0..t_len) |t| {
        // Compute all 4 gates for all dimensions
        for (0..h_dim) |d| {
            var pre_i = b_ih[d] + b_hh[d];
            var pre_f = b_ih[h_dim + d] + b_hh[h_dim + d];
            var pre_g = b_ih[2 * h_dim + d] + b_hh[2 * h_dim + d];
            var pre_o = b_ih[3 * h_dim + d] + b_hh[3 * h_dim + d];

            const row_ih_i = w_ih + d * h_dim;
            const row_ih_f = w_ih + (h_dim + d) * h_dim;
            const row_ih_g = w_ih + (2 * h_dim + d) * h_dim;
            const row_ih_o = w_ih + (3 * h_dim + d) * h_dim;

            const row_hh_i = w_hh + d * h_dim;
            const row_hh_f = w_hh + (h_dim + d) * h_dim;
            const row_hh_g = w_hh + (2 * h_dim + d) * h_dim;
            const row_hh_o = w_hh + (3 * h_dim + d) * h_dim;

            for (0..h_dim) |k| {
                const in_val = in[k * t_len + t];
                const prev_h = h_state[k];

                pre_i += row_ih_i[k] * in_val + row_hh_i[k] * prev_h;
                pre_f += row_ih_f[k] * in_val + row_hh_f[k] * prev_h;
                pre_g += row_ih_g[k] * in_val + row_hh_g[k] * prev_h;
                pre_o += row_ih_o[k] * in_val + row_hh_o[k] * prev_h;
            }

            const gate_i = sigmoid(pre_i);
            const gate_f = sigmoid(pre_f);
            const gate_g = tanhActivation(pre_g);
            const gate_o = sigmoid(pre_o);

            const next_c = gate_f * c_state[d] + gate_i * gate_g;
            const next_h = gate_o * tanhActivation(next_c);

            out[d * t_len + t] = next_h;
            c_state[d] = next_c;
        }

        // Advance hidden state to t+1
        for (0..h_dim) |d| {
            h_state[d] = out[d * t_len + t];
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

    const get_offset = struct {
        fn run(r: anytype, k: []const u8) !usize {
            const entry = r.get(k) orelse return error.TensorNotFound;
            const offsets = entry.object.get("data_offsets") orelse return error.InvalidFormat;
            return @as(usize, @intCast(offsets.array.items[0].integer));
        }
    }.run;

    // Layer 0 weights (Conv1d)
    const bias_offset = try get_offset(root, "decoder.layers.0.conv.bias");
    const weight_v_offset = try get_offset(root, "decoder.layers.0.conv.weight_v");
    const weight_g_offset = try get_offset(root, "decoder.layers.0.conv.weight_g");

    const conv0_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + bias_offset ..].ptr));
    const conv0_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + weight_v_offset ..].ptr));
    const conv0_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + weight_g_offset ..].ptr));

    // Layer 1 weights (2-Layer LSTM)
    const l0_w_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_ih_l0") ..].ptr)));
    const l0_w_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_hh_l0") ..].ptr)));
    const l0_b_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_ih_l0") ..].ptr)));
    const l0_b_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_hh_l0") ..].ptr)));

    const l1_w_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_ih_l1") ..].ptr)));
    const l1_w_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_hh_l1") ..].ptr)));
    const l1_b_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_ih_l1") ..].ptr)));
    const l1_b_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_hh_l1") ..].ptr)));

    // Normalize weights for Conv1d layer 0
    const C_OUT = 512;
    const C_IN = 128;
    const K = 7;
    const weight_norm = try allocator.alloc(f32, C_OUT * C_IN * K);

    for (0..C_OUT) |co| {
        var norm_sq: f32 = 0.0;
        const slice_len = C_IN * K;
        const base = co * slice_len;

        for (0..slice_len) |idx| {
            const val = conv0_weight_v[base + idx];
            norm_sq += val * val;
        }

        const scale = conv0_weight_g[co] / @sqrt(norm_sq);
        for (0..slice_len) |idx| {
            weight_norm[base + idx] = conv0_weight_v[base + idx] * scale;
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
    conv1dSame(&latents, C_IN, num_frames, weight_norm, conv0_bias[0..C_OUT], C_OUT, K, conv_out);

    // 7. Run 2-Layer LSTM: [512, 4] -> [512, 4]
    const lstm_l0_out = try allocator.alloc(f32, C_OUT * num_frames);
    const lstm_l1_out = try allocator.alloc(f32, C_OUT * num_frames);
    const h_buf = try allocator.alloc(f32, C_OUT);
    const c_buf = try allocator.alloc(f32, C_OUT);

    runLstmLayer(conv_out, lstm_l0_out, C_OUT, num_frames, l0_w_ih, l0_w_hh, l0_b_ih, l0_b_hh, h_buf, c_buf);
    @memset(h_buf, 0.0);
    @memset(c_buf, 0.0);
    runLstmLayer(lstm_l0_out, lstm_l1_out, C_OUT, num_frames, l1_w_ih, l1_w_hh, l1_b_ih, l1_b_hh, h_buf, c_buf);

    // 8. Inspect output
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Successfully computed 2-Layer LSTM ([512, {d}] -> [512, {d}]).\n", .{ num_frames, num_frames });
    try stdout.print("LSTM Layer 1 Channel 0 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{lstm_l1_out[0 * num_frames + t]});
    }
    try stdout.print("\nLSTM Layer 1 Channel 1 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{lstm_l1_out[1 * num_frames + t]});
    }
    try stdout.writeByte('\n');
    try stdout.flush();
}
