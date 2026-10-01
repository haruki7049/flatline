const std = @import("std");

pub fn main() !void {
    const file_path = "weights/encodec_24khz.safetensors";

    // 1. Open weight file
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = (try file.stat()).size;

    // 2. Map file into memory space (zero-copy)
    const mapped = try std.posix.mmap(
        null,
        file_size,
        std.posix.PROT_READ,
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mapped);

    if (mapped.len < 8) return error.InvalidFile;

    // 3. Read 8-byte little-endian header length
    const header_len = std.mem.readInt(u64, mapped[0..8], .little);
    const header_json = mapped[8 .. 8 + header_len];
    const payload_start = 8 + header_len;

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Header Length: {d} bytes\n", .{header_len});
    try stdout.print("Payload Start Offset: {d}\n\n", .{payload_start});

    // 4. Print header JSON to inspect tensor keys
    try stdout.writeAll(header_json);
    try stdout.writeByte('\n');
}
