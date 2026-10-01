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

pub const Audio = struct {
    samples: []f32,
    sample_rate: u32,

    pub fn deinit(self: Audio, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

const format_pcm = 1;
const format_ieee_float = 3;
const format_extensible = 0xFFFE;

/// Reads a mono or stereo WAV (16-bit PCM or 32-bit float) and downmixes it to mono by
/// averaging channels, like encodec.utils.convert_audio. 16-bit PCM is scaled by 1/32768,
/// like torchaudio.load. Caller owns the returned samples.
pub fn readMono(allocator: std.mem.Allocator, bytes: []const u8) !Audio {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.InvalidWav;

    var format: ?struct { tag: u16, channels: u16, sample_rate: u32, bits: u16 } = null;
    var data: ?[]const u8 = null;

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const id = bytes[pos..][0..4];
        const size = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        const body = bytes[pos + 8 ..][0..@min(size, bytes.len - pos - 8)];

        if (std.mem.eql(u8, id, "fmt ")) {
            if (body.len < 16) return error.InvalidWav;
            var tag = std.mem.readInt(u16, body[0..2], .little);
            // WAVE_FORMAT_EXTENSIBLE: the actual format is the first 2 bytes of the SubFormat GUID
            if (tag == format_extensible) {
                if (body.len < 26) return error.InvalidWav;
                tag = std.mem.readInt(u16, body[24..26], .little);
            }
            format = .{
                .tag = tag,
                .channels = std.mem.readInt(u16, body[2..4], .little),
                .sample_rate = std.mem.readInt(u32, body[4..8], .little),
                .bits = std.mem.readInt(u16, body[14..16], .little),
            };
        } else if (std.mem.eql(u8, id, "data")) {
            data = body;
        }

        // Chunks are padded to an even size
        pos += 8 + @as(usize, size) + (size & 1);
    }

    const fmt = format orelse return error.InvalidWav;
    const pcm = data orelse return error.InvalidWav;
    if (fmt.channels != 1 and fmt.channels != 2) return error.UnsupportedChannelCount;

    const bytes_per_sample: usize = switch (fmt.tag) {
        format_pcm => if (fmt.bits == 16) 2 else return error.UnsupportedWavFormat,
        format_ieee_float => if (fmt.bits == 32) 4 else return error.UnsupportedWavFormat,
        else => return error.UnsupportedWavFormat,
    };
    const frame_size = bytes_per_sample * fmt.channels;
    const num_samples = pcm.len / frame_size;

    const samples = try allocator.alloc(f32, num_samples);
    errdefer allocator.free(samples);
    for (samples, 0..) |*sample, i| {
        var sum: f32 = 0.0;
        for (0..fmt.channels) |ch| {
            const p = pcm[i * frame_size + ch * bytes_per_sample ..];
            sum += if (bytes_per_sample == 2)
                @as(f32, @floatFromInt(std.mem.readInt(i16, p[0..2], .little))) / 32768.0
            else
                @bitCast(std.mem.readInt(u32, p[0..4], .little));
        }
        sample.* = sum / @as(f32, @floatFromInt(fmt.channels));
    }

    return .{ .samples = samples, .sample_rate = fmt.sample_rate };
}

pub fn readMonoFile(io: std.Io, allocator: std.mem.Allocator, file_path: []const u8) !Audio {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file_path, allocator, .limited(1 << 30));
    defer allocator.free(bytes);
    return readMono(allocator, bytes);
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

test "readMono reads back 16-bit PCM written by writeMono16" {
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeMono16(&w, &.{ 0.0, 0.5, -1.0 }, 24000);

    const audio = try readMono(std.testing.allocator, w.buffered());
    defer audio.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 24000), audio.sample_rate);
    // 0.5 * 32767 truncates to 16383; reading scales by 1/32768
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 16383.0 / 32768.0, -32767.0 / 32768.0 }, audio.samples);
}

test "readMono reads float32 stereo, averaging channels and skipping unknown chunks" {
    const header = "RIFF\x00\x00\x00\x00WAVE" ++
        "LIST\x03\x00\x00\x00abc\x00" ++ // odd-sized chunk with a pad byte
        "fmt \x10\x00\x00\x00\x03\x00\x02\x00\xc0\x5d\x00\x00\x00\x77\x01\x00\x08\x00\x20\x00" ++
        "data\x10\x00\x00\x00";
    var buf: [header.len + 16]u8 = undefined;
    @memcpy(buf[0..header.len], header);
    const pcm = [_]f32{ 1.0, 0.0, -0.5, 0.25 };
    @memcpy(buf[header.len..], std.mem.sliceAsBytes(&pcm));

    const audio = try readMono(std.testing.allocator, &buf);
    defer audio.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 24000), audio.sample_rate);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.125 }, audio.samples);
}

test "readMono rejects unsupported input" {
    try std.testing.expectError(error.InvalidWav, readMono(std.testing.allocator, "RIFX0000WAVE"));
    const pcm8 = "RIFF\x00\x00\x00\x00WAVE" ++
        "fmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0\x5d\x00\x00\xc0\x5d\x00\x00\x01\x00\x08\x00" ++
        "data\x00\x00\x00\x00";
    try std.testing.expectError(error.UnsupportedWavFormat, readMono(std.testing.allocator, pcm8));
}
