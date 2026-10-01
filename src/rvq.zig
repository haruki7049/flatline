const std = @import("std");

pub const num_stages = 8;

/// One RVQ token per stage for a single frame.
pub const Frame = [num_stages]u16;

/// Parses raw little-endian u16 tokens laid out as [num_frames, num_stages].
/// Caller owns the returned slice.
pub fn framesFromBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]Frame {
    if (bytes.len % @sizeOf(Frame) != 0) return error.InvalidTokenFileSize;
    if (bytes.len == 0) return error.EmptyTokenFile;

    const frames = try allocator.alloc(Frame, bytes.len / @sizeOf(Frame));
    for (frames, 0..) |*frame, t| {
        const frame_bytes = bytes[t * @sizeOf(Frame) ..][0..@sizeOf(Frame)];
        for (frame, 0..) |*token, stage| {
            token.* = std.mem.readInt(u16, frame_bytes[stage * 2 ..][0..2], .little);
        }
    }
    return frames;
}

/// Residual vector quantizer: a frame decodes to the sum of one codebook entry per stage.
pub const Quantizer = struct {
    /// Each codebook is [codebook_size, dim].
    codebooks: [num_stages][]const f32,
    codebook_size: usize,
    dim: usize,

    pub fn init(codebooks: [num_stages][]const f32, dim: usize) !Quantizer {
        if (dim == 0 or codebooks[0].len % dim != 0) return error.ShapeMismatch;
        for (codebooks) |codebook| {
            if (codebook.len != codebooks[0].len) return error.ShapeMismatch;
        }
        return .{
            .codebooks = codebooks,
            .codebook_size = codebooks[0].len / dim,
            .dim = dim,
        };
    }

    /// Decodes frames into a latent matrix `out` of shape [dim, frames.len].
    pub fn decode(self: Quantizer, frames: []const Frame, out: []f32) !void {
        const num_frames = frames.len;
        std.debug.assert(out.len == self.dim * num_frames);

        for (frames) |frame| {
            for (frame) |token| {
                if (token >= self.codebook_size) return error.InvalidToken;
            }
        }

        @memset(out, 0.0);

        for (frames, 0..) |frame, t| {
            for (frame, 0..) |token, stage| {
                const entry = self.codebooks[stage][@as(usize, token) * self.dim ..][0..self.dim];
                for (entry, 0..) |val, c| {
                    out[c * num_frames + t] += val;
                }
            }
        }
    }
};

test "framesFromBytes reads little-endian frames" {
    const bytes = [_]u8{
        0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08, 0x00,
        0xff, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const frames = try framesFromBytes(std.testing.allocator, &bytes);
    defer std.testing.allocator.free(frames);

    try std.testing.expectEqual(@as(usize, 2), frames.len);
    try std.testing.expectEqual(Frame{ 1, 2, 3, 4, 5, 6, 7, 8 }, frames[0]);
    try std.testing.expectEqual(Frame{ 1023, 256, 0, 0, 0, 0, 0, 0 }, frames[1]);
}

test "framesFromBytes rejects partial frames and empty input" {
    const partial = [_]u8{0} ** (@sizeOf(Frame) + 2);
    try std.testing.expectError(error.InvalidTokenFileSize, framesFromBytes(std.testing.allocator, &partial));
    try std.testing.expectError(error.EmptyTokenFile, framesFromBytes(std.testing.allocator, &.{}));
}

test "Quantizer sums one entry per stage" {
    // codebook_size = 2, dim = 2; every stage shares the same codebook.
    const codebook = [_]f32{ 1.0, 2.0, 10.0, 20.0 };
    const quantizer = try Quantizer.init(@splat(&codebook), 2);

    const frames = [_]Frame{
        .{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 1, 0, 0, 0, 0, 0, 0, 1 },
    };
    var out: [4]f32 = undefined;
    try quantizer.decode(&frames, &out);

    // Layout is [dim, frames]
    try std.testing.expectEqualSlices(f32, &.{ 8.0, 26.0, 16.0, 52.0 }, &out);
}

test "Quantizer rejects out-of-range tokens" {
    const codebook = [_]f32{ 1.0, 2.0 };
    const quantizer = try Quantizer.init(@splat(&codebook), 2);
    const frames = [_]Frame{.{ 1, 0, 0, 0, 0, 0, 0, 0 }};
    var out: [2]f32 = undefined;
    try std.testing.expectError(error.InvalidToken, quantizer.decode(&frames, &out));
}
