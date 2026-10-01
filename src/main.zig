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

    // Check command line arguments for header dump mode
    var args = init.minimal.args;
    var args_iterator = args.iterate();
    _ = args_iterator.next(); // Skip binary name
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

    // 4. Parse JSON metadata to dynamically extract codebook offsets
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

    const NUM_STAGES = 8;
    var codebooks: [NUM_STAGES][*]const f32 = undefined;

    for (0..NUM_STAGES) |stage| {
        var key_buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "quantizer.layers.{d}.codebook.embed", .{stage});

        const tensor_entry = root.get(key) orelse return error.TensorNotFound;
        const offsets = tensor_entry.object.get("data_offsets") orelse return error.InvalidFormat;
        const start_offset = @as(usize, @intCast(offsets.array.items[0].integer));

        const tensor_start = payload_start + start_offset;
        codebooks[stage] = @ptrCast(@alignCast(mapped[tensor_start..].ptr));
    }

    // 5. Decode 4 frames
    var sequence_tokens = [_][8]u16{
        [_]u16{ 120, 450, 89, 730, 210, 95, 600, 314 },
        [_]u16{ 121, 448, 88, 729, 212, 94, 599, 310 },
        [_]u16{ 125, 440, 92, 735, 205, 99, 605, 320 },
        [_]u16{ 130, 435, 95, 740, 200, 102, 610, 325 },
    };
    const num_frames = sequence_tokens.len;

    var latents: [128 * 4]f32 = undefined;
    decodeRVQSequence(&codebooks, &sequence_tokens, num_frames, &latents);

    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Decoded {d} frames sequence into [128, {d}] matrix.\n", .{ num_frames, num_frames });
    try stdout.print("Latent channel 0 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{latents[0 * num_frames + t]});
    }
    try stdout.print("\nLatent channel 1 across 4 frames: ", .{});
    for (0..num_frames) |t| {
        try stdout.print("{d:.4} ", .{latents[1 * num_frames + t]});
    }
    try stdout.writeByte('\n');
    try stdout.flush();
}
