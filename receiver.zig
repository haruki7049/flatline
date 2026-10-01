const std = @import("std");

pub fn main() !void {
    var buffer: [4096]u8 = undefined;
    var total_tokens: usize = 0;

    // Read streamed token bytes directly from stdin file descriptor
    while (true) {
        const bytes_read = try std.posix.read(std.posix.STDIN_FILENO, &buffer);
        if (bytes_read == 0) break;

        // Interpret raw bytes as 16-bit signed integer tokens
        const token_count = bytes_read / @sizeOf(i16);
        const tokens: [*]align(@alignOf(i16)) const i16 = @ptrCast(@alignCast(&buffer));

        total_tokens += token_count;
        std.debug.print("Received {d} tokens (First: {d})\n", .{ token_count, tokens[0] });
    }

    std.debug.print("Pipeline completed. Total tokens received: {d}\n", .{total_tokens});
}
