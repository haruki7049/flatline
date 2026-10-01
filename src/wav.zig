const std = @import("std");

/// Writes float PCM [-1.0, 1.0] as a 16-bit PCM mono WAV stream.
pub fn writeMono16(w: *std.Io.Writer, samples: []const f32, sample_rate: u32) !void {
    const num_samples = std.math.cast(u32, samples.len) orelse return error.TooManySamples;
    const byte_rate = sample_rate * 2; // 1 channel * 2 bytes/sample
    const block_align: u16 = 2; // 1 channel * 2 bytes
    const bits_per_sample: u16 = 16;
    const data_bytes = std.math.mul(u32, num_samples, 2) catch return error.TooManySamples;
    const riff_chunk_size = std.math.add(u32, 36, data_bytes) catch return error.TooManySamples;

    // 1. RIFF Header
    try w.writeAll("RIFF");
    try w.writeInt(u32, riff_chunk_size, .little);
    try w.writeAll("WAVE");

    // 2. fmt Sub-chunk
    try w.writeAll("fmt ");
    try w.writeInt(u32, 16, .little); // Subchunk1Size (16 for PCM)
    try w.writeInt(u16, 1, .little); // AudioFormat (1 for PCM)
    try w.writeInt(u16, 1, .little); // NumChannels (1 = mono)
    try w.writeInt(u32, sample_rate, .little);
    try w.writeInt(u32, byte_rate, .little);
    try w.writeInt(u16, block_align, .little);
    try w.writeInt(u16, bits_per_sample, .little);

    // 3. data Sub-chunk
    try w.writeAll("data");
    try w.writeInt(u32, data_bytes, .little);

    // Convert f32 [-1.0, 1.0] to i16 with clamping
    for (samples) |sample| {
        const clamped = std.math.clamp(sample, -1.0, 1.0);
        const sample_i16 = @as(i16, @intFromFloat(clamped * 32767.0));
        try w.writeInt(i16, sample_i16, .little);
    }
}

/// Writes float PCM [-1.0, 1.0] to a 16-bit PCM mono WAV file.
pub fn writeMono16File(io: std.Io, file_path: []const u8, samples: []const f32, sample_rate: u32) !void {
    const file = try std.Io.Dir.cwd().createFile(io, file_path, .{});
    defer file.close(io);

    var buffer: [1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writeMono16(&writer.interface, samples, sample_rate);
    try writer.interface.flush();
}

test "writeMono16 emits a 44-byte header followed by clamped samples" {
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeMono16(&w, &.{ 0.0, 1.0, -2.0 }, 24000);

    const out = w.buffered();
    try std.testing.expectEqual(@as(usize, 44 + 3 * 2), out.len);
    try std.testing.expectEqualStrings("RIFF", out[0..4]);
    try std.testing.expectEqual(@as(u32, 36 + 6), std.mem.readInt(u32, out[4..8], .little));
    try std.testing.expectEqualStrings("WAVEfmt ", out[8..16]);
    try std.testing.expectEqual(@as(u32, 24000), std.mem.readInt(u32, out[24..28], .little));
    try std.testing.expectEqual(@as(u32, 48000), std.mem.readInt(u32, out[28..32], .little));
    try std.testing.expectEqualStrings("data", out[36..40]);
    try std.testing.expectEqual(@as(u32, 6), std.mem.readInt(u32, out[40..44], .little));
    try std.testing.expectEqual(@as(i16, 0), std.mem.readInt(i16, out[44..46], .little));
    try std.testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, out[46..48], .little));
    try std.testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, out[48..50], .little));
}
