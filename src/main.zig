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

// Neutralizes acoustic dynamics by clamping higher stages
pub fn neutralizeAcoustics(tokens: *[8]u16) void {
    // Keep Stage 0 and 1 (linguistic content)
    // Clamp Stage 2 to 7 to constant neutral index
    for (2..8) |stage| {
        tokens[stage] = 0;
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

    // 5. Compare expressive tokens vs clamped tokens
    var dynamic_tokens = [_]u16{ 120, 450, 89, 730, 210, 95, 600, 314 };
    var clamped_tokens = dynamic_tokens;
    neutralizeAcoustics(&clamped_tokens);

    var latent_dynamic: [128]f32 = undefined;
    var latent_clamped: [128]f32 = undefined;

    decodeRVQFrame(&codebooks, &dynamic_tokens, &latent_dynamic);
    decodeRVQFrame(&codebooks, &clamped_tokens, &latent_clamped);

    // 6. Inspect output difference
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Dynamic Tokens: {any}\n", .{dynamic_tokens});
    try stdout.print("Clamped Tokens: {any}\n\n", .{clamped_tokens});

    try stdout.print("Dim |  Dynamic   |  Clamped   |  Delta\n", .{});
    try stdout.print("----+------------+------------+------------\n", .{});
    for (0..8) |d| {
        const delta = latent_dynamic[d] - latent_clamped[d];
        try stdout.print(" {d}  | {d:10.5} | {d:10.5} | {d:10.5}\n", .{
            d,
            latent_dynamic[d],
            latent_clamped[d],
            delta,
        });
    }
    try stdout.flush();
}
