const std = @import("std");

const MAGIC_NUMBER: u16 = 0xAA55;
const SILENCE_TOKEN: i16 = 110;
const SILENCE_THRESHOLD_FRAMES: usize = 12; // 40ms * 12 = 480ms of silence
const MIN_SPEECH_FRAMES: usize = 3; // Filter out noise below 120ms

const State = enum {
    idle,
    listening,
};

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

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();

    var token_buffer: std.ArrayListUnmanaged(i16) = .empty;
    defer token_buffer.deinit(allocator);

    var state: State = .idle;
    var silence_frames: usize = 0;
    var speech_frame_count: usize = 0;

    std.debug.print("[flatline-core] VAD buffering active. Waiting for speech...\n", .{});

    while (true) {
        var header_buf: [@sizeOf(Header)]u8 = undefined;
        const has_header = try readExact(std.posix.STDIN_FILENO, &header_buf);
        if (!has_header) break;

        const header: *const Header = @ptrCast(@alignCast(&header_buf));
        if (header.magic != MAGIC_NUMBER) {
            return error.InvalidPacket;
        }

        var payload_buf: [4096]u8 = undefined;
        if (header.payload_len > payload_buf.len) return error.BufferOverflow;

        const payload = payload_buf[0..header.payload_len];
        if (!try readExact(std.posix.STDIN_FILENO, payload)) {
            return error.IncompletePayload;
        }

        const count = header.payload_len / @sizeOf(i16);
        const tokens: [*]align(@alignOf(i16)) const i16 = @ptrCast(@alignCast(payload.ptr));
        const first_token = tokens[0];

        const is_silence = (first_token == SILENCE_TOKEN);

        switch (state) {
            .idle => {
                if (!is_silence) {
                    state = .listening;
                    silence_frames = 0;
                    speech_frame_count = 1;
                    token_buffer.clearRetainingCapacity();
                    try token_buffer.appendSlice(allocator, tokens[0..count]);
                    std.debug.print("\n>>> [SPEECH STARTED]\n", .{});
                }
            },
            .listening => {
                speech_frame_count += 1;
                try token_buffer.appendSlice(allocator, tokens[0..count]);

                if (is_silence) {
                    silence_frames += 1;
                    if (silence_frames >= SILENCE_THRESHOLD_FRAMES) {
                        state = .idle;
                        const valid_frames = speech_frame_count - silence_frames;

                        if (valid_frames < MIN_SPEECH_FRAMES) {
                            std.debug.print("<<< [IGNORED NOISE] Duration: ~{d}ms\n\n", .{valid_frames * 40});
                        } else {
                            const trimmed_len = valid_frames * count;
                            const utterance_tokens = token_buffer.items[0..trimmed_len];
                            const duration_ms = valid_frames * 40;

                            std.debug.print("<<< [SPEECH COMMITTED] Duration: ~{d}ms, Total Tokens: {d}\n", .{
                                duration_ms,
                                utterance_tokens.len,
                            });
                        }

                        token_buffer.clearRetainingCapacity();
                        silence_frames = 0;
                        speech_frame_count = 0;
                    }
                } else {
                    silence_frames = 0;
                }
            },
        }
    }
}
