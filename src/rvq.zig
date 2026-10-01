const std = @import("std");

pub const num_stages = 8;

/// One RVQ token per stage for a single frame.
pub const Frame = [num_stages]u16;

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
