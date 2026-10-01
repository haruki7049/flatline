const std = @import("std");

const MAGIC_NUMBER: u16 = 0xAA55;

const Header = extern struct {
    magic: u16,
    version: u8,
    reserved: u8,
    payload_len: u32,
};

fn readExact(fd: std.posix.fd_t, buffer: []u8) !bool {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const bytes_read = try std.posix.read(fd, buffer[total_read..]);
        if (bytes_read == 0) {
            if (total_read == 0) return false;
            return error.UnexpectedEof;
        }
        total_read += bytes_read;
    }
    return true;
}

pub fn main() !void {
    var total_tokens: usize = 0;
    var frame_count: usize = 0;

    while (true) {
        var header_buf: [@sizeOf(Header)]u8 = undefined;
        const has_header = try readExact(std.posix.STDIN_FILENO, &header_buf);
        if (!has_header) break;

        const header: *const Header = @ptrCast(@alignCast(&header_buf));

        // Validate magic number to discard noise or misaligned bytes
        if (header.magic != MAGIC_NUMBER) {
            std.debug.print("Invalid magic number: 0x{X:0>4}\n", .{header.magic});
            return error.InvalidPacket;
        }

        // Allocate payload buffer and read exact token bytes
        var payload_buf: [4096]u8 = undefined;
        if (header.payload_len > payload_buf.len) return error.BufferOverflow;

        const payload = payload_buf[0..header.payload_len];
        if (!try readExact(std.posix.STDIN_FILENO, payload)) {
            return error.IncompletePayload;
        }

        const token_count = header.payload_len / @sizeOf(i16);
        const tokens: [*]align(@alignOf(i16)) const i16 = @ptrCast(@alignCast(payload.ptr));

        total_tokens += token_count;
        frame_count += 1;

        std.debug.print("Frame {d}: received {d} tokens (First: {d})\n", .{
            frame_count,
            token_count,
            tokens[0],
        });
    }

    std.debug.print("Pipeline completed cleanly. Frames: {d}, Total tokens: {d}\n", .{
        frame_count,
        total_tokens,
    });
}
