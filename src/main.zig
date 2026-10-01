const std = @import("std");

// Decodes discrete RVQ tokens into a single continuous latent frame
pub fn decodeRVQFrame(
    codebooks: []const [*]const f32,
    tokens: []const u16,
    out_latent: *[128]f32,
) void {
    @memset(out_latent, 0.0);

    for (tokens, 0..) |token, stage| {
        const base_ptr = codebooks[stage] + (@as(usize, token) * 128);
        for (0..128) |d| {
            out_latent[d] += base_ptr[d];
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

    // Collect 8 codebook pointers (stages 0 to 7)
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

    // 5. Test RVQ decoding with dummy tokens
    const dummy_tokens = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var latent: [128]f32 = undefined;

    decodeRVQFrame(&codebooks, &dummy_tokens, &latent);

    // 6. Inspect resulting latent vector
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Successfully parsed {d} RVQ stages.\n", .{NUM_STAGES});
    try stdout.print("Decoded latent frame (first 8 dimensions):\n", .{});
    for (0..8) |d| {
        try stdout.print("  out[{d}] = {d:.6}\n", .{ d, latent[d] });
    }
    try stdout.flush();
}
