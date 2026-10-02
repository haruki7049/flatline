//! Native microphone capture + simple VAD flagging. Replaces
//! python/pipeline/stream_mic_encoder.py: captures 24kHz mono 16-bit PCM from the
//! default input device via zaudio/miniaudio, buffers it into 40ms (960-sample)
//! chunks, tags each chunk with an RMS-energy-based is_speech flag, and writes the
//! 0xAA55-framed packets to stdout for receiver.zig to consume.
//!
//! No Python in this process. miniaudio's capture callback runs on its own
//! real-time audio thread; this file does the RMS calculation and the stdout
//! write directly on that thread (same direct-POSIX-IO style as receiver.zig),
//! since there is exactly one producer and no contention to synchronize.

const std = @import("std");
const zaudio = @import("zaudio");

const MAGIC_NUMBER: u16 = 0xAA55;
const SAMPLE_RATE = 24000;
const SAMPLES_PER_FRAME = 960; // 40ms @ 24kHz
const RMS_THRESHOLD: f32 = 0.015;

const Header = extern struct {
    magic: u16,
    version: u8,
    reserved: u8,
    payload_len: u32,
};

const CaptureState = struct {
    buffer: [SAMPLES_PER_FRAME]i16 = undefined,
    filled: usize = 0,
};

// Direct POSIX write (same technique as receiver.zig/player.zig).
fn writeAll(fd: c_int, buffer: []const u8) void {
    var total_written: usize = 0;
    while (total_written < buffer.len) {
        const rc = std.c.write(fd, buffer.ptr + total_written, buffer.len - total_written);
        if (rc <= 0) {
            // Downstream pipe closed (e.g. receiver exited); nothing useful to do
            // from the audio callback thread other than stop emitting frames.
            return;
        }
        total_written += @intCast(rc);
    }
}

fn emitChunk(samples: *const [SAMPLES_PER_FRAME]i16) void {
    var sum_sq: f32 = 0.0;
    for (samples) |sample| {
        const s: f32 = @as(f32, @floatFromInt(sample)) / 32768.0;
        sum_sq += s * s;
    }
    const rms = @sqrt(sum_sq / @as(f32, SAMPLES_PER_FRAME));
    const is_speech: u8 = if (rms > RMS_THRESHOLD) 1 else 0;

    const payload_bytes = std.mem.sliceAsBytes(samples);
    const header = Header{
        .magic = MAGIC_NUMBER,
        .version = 1,
        .reserved = is_speech,
        .payload_len = @intCast(payload_bytes.len),
    };

    writeAll(1, std.mem.asBytes(&header));
    writeAll(1, payload_bytes);
}

fn dataCallback(
    device: *zaudio.Device,
    output: ?*anyopaque,
    input: ?*const anyopaque,
    frame_count: u32,
) callconv(.c) void {
    _ = output;
    const state: *CaptureState = @ptrCast(@alignCast(device.getUserData().?));
    const in_samples: [*]const i16 = @ptrCast(@alignCast(input orelse return));

    var i: usize = 0;
    while (i < frame_count) : (i += 1) {
        state.buffer[state.filled] = in_samples[i];
        state.filled += 1;
        if (state.filled == SAMPLES_PER_FRAME) {
            emitChunk(&state.buffer);
            state.filled = 0;
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    zaudio.init(allocator);
    defer zaudio.deinit();

    var state: CaptureState = .{};

    var config = zaudio.Device.Config.init(.capture);
    config.sample_rate = SAMPLE_RATE;
    config.capture.format = .signed16;
    config.capture.channels = 1;
    config.data_callback = dataCallback;
    config.user_data = &state;

    const device = try zaudio.Device.create(null, config);
    defer device.destroy();

    std.debug.print("[flatline-capture] Microphone stream active. Transmitting raw PCM frames at {d}Hz...\n", .{SAMPLE_RATE});
    try device.start();

    // The actual work happens on miniaudio's capture thread (dataCallback above);
    // this thread just has to stay alive for the process lifetime.
    while (true) {
        try io.sleep(.fromMilliseconds(1000), .awake);
    }
}
