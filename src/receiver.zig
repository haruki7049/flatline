const std = @import("std");

const MAGIC_NUMBER: u16 = 0xAA55;
const SAMPLES_PER_FRAME: usize = 960; // 40ms @ 24kHz
const SILENCE_THRESHOLD_FRAMES: usize = 12; // 40ms * 12 = 480ms of silence
const MIN_SPEECH_FRAMES: usize = 3; // Filter out noise below 120ms

const State = enum {
    idle,
    listening,
};

const Header = extern struct {
    magic: u16,
    version: u8,
    reserved: u8, // is_speech flag (0 = silence, 1 = speech), set by the sender
    payload_len: u32,
};

// Direct POSIX exact read to bypass runtime abstraction issues on pipes
fn readExact(fd: c_int, buffer: []u8) !bool {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const rc = std.c.read(fd, buffer.ptr + total_read, buffer.len - total_read);
        if (rc == 0) {
            if (total_read == 0) return false;
            return error.UnexpectedEof;
        }
        if (rc < 0) {
            return error.ReadFailed;
        }
        total_read += @intCast(rc);
    }
    return true;
}

// Direct POSIX write
fn writeAll(fd: c_int, buffer: []const u8) !void {
    var total_written: usize = 0;
    while (total_written < buffer.len) {
        const rc = std.c.write(fd, buffer.ptr + total_written, buffer.len - total_written);
        if (rc <= 0) return error.WriteFailed;
        total_written += @intCast(rc);
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();

    // Buffers raw PCM samples (i16, 24kHz, mono) for the in-progress utterance.
    var sample_buffer: std.ArrayListUnmanaged(i16) = .empty;
    defer sample_buffer.deinit(allocator);

    var state: State = .idle;
    var silence_frames: usize = 0;
    var speech_frame_count: usize = 0;

    // Use raw stderr write to guarantee visibility before buffer flush
    _ = std.c.write(2, "[flatline-core] VAD buffering active. Waiting for speech...\n", 60);

    while (true) {
        var header_buf: [@sizeOf(Header)]u8 = undefined;
        const has_header = try readExact(0, &header_buf);
        if (!has_header) break;

        const header: *const Header = @ptrCast(@alignCast(&header_buf));
        if (header.magic != MAGIC_NUMBER) {
            _ = std.c.write(2, "Invalid magic number received\n", 30);
            return error.InvalidPacket;
        }

        var payload_buf: [4096]u8 = undefined;
        if (header.payload_len > payload_buf.len) return error.BufferOverflow;

        const payload = payload_buf[0..header.payload_len];
        if (!try readExact(0, payload)) {
            return error.IncompletePayload;
        }

        const count = header.payload_len / @sizeOf(i16);
        const samples: [*]align(@alignOf(i16)) const i16 = @ptrCast(@alignCast(payload.ptr));

        const is_silence = (header.reserved == 0);

        switch (state) {
            .idle => {
                if (!is_silence) {
                    state = .listening;
                    silence_frames = 0;
                    speech_frame_count = 1;
                    sample_buffer.clearRetainingCapacity();
                    try sample_buffer.appendSlice(allocator, samples[0..count]);
                    _ = std.c.write(2, "\n>>> [SPEECH STARTED]\n", 22);
                }
            },
            .listening => {
                speech_frame_count += 1;
                try sample_buffer.appendSlice(allocator, samples[0..count]);

                if (is_silence) {
                    silence_frames += 1;
                    if (silence_frames >= SILENCE_THRESHOLD_FRAMES) {
                        state = .idle;
                        const valid_frames = speech_frame_count - silence_frames;

                        if (valid_frames < MIN_SPEECH_FRAMES) {
                            _ = std.c.write(2, "<<< [IGNORED NOISE]\n", 20);
                        } else {
                            const trimmed_len = valid_frames * SAMPLES_PER_FRAME;
                            const utterance_samples = sample_buffer.items[0..trimmed_len];

                            _ = std.c.write(2, "<<< [SPEECH COMMITTED] Transmitting to decoder\n", 47);

                            // Forward the committed, silence-trimmed raw PCM utterance via stdout (fd 1)
                            const payload_bytes: []const u8 = std.mem.sliceAsBytes(utterance_samples);
                            const out_header = Header{
                                .magic = MAGIC_NUMBER,
                                .version = 1,
                                .reserved = 1,
                                .payload_len = @intCast(payload_bytes.len),
                            };

                            try writeAll(1, std.mem.asBytes(&out_header));
                            try writeAll(1, payload_bytes);
                        }

                        sample_buffer.clearRetainingCapacity();
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
