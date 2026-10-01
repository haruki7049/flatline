const std = @import("std");
const SafeTensors = @import("model.zig").SafeTensors;

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

/// Writes tokens in the layout read by `framesFromBytes`.
pub fn writeFrames(w: *std.Io.Writer, frames: []const Frame) !void {
    for (frames) |frame| {
        for (frame) |token| {
            try w.writeInt(u16, token, .little);
        }
    }
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
        const codebook_size = codebooks[0].len / dim;
        if (codebook_size == 0 or codebook_size > std.math.maxInt(u16) + 1) return error.ShapeMismatch;
        return .{
            .codebooks = codebooks,
            .codebook_size = codebook_size,
            .dim = dim,
        };
    }

    /// Uses the first `num_stages` codebooks (6 kbps at 75 frames/s).
    pub fn load(weights: *const SafeTensors) !Quantizer {
        var codebooks: [num_stages][]const f32 = undefined;
        var dim: usize = 0;
        for (&codebooks, 0..) |*codebook, stage| {
            const t = try weights.tensorFmt("quantizer.layers.{d}.codebook.embed", .{stage});
            if (t.rank != 2) return error.ShapeMismatch;
            if (stage > 0 and t.shape()[1] != dim) return error.ShapeMismatch;
            codebook.* = t.data;
            dim = t.shape()[1];
        }
        return init(codebooks, dim);
    }

    /// Encodes a latent matrix [dim, num_frames] into `out` (one Frame per time step) by greedy
    /// residual nearest-neighbour search, like encodec's ResidualVectorQuantization.encode:
    /// k = argmin ||r - e_k||^2, r -= e_k, with the distance evaluated as
    /// (||r||^2 - 2 r.e_k) + ||e_k||^2 as in EuclideanCodebook.quantize.
    pub fn encode(self: Quantizer, allocator: std.mem.Allocator, latents: []const f32, out: []Frame) !void {
        const num_frames = out.len;
        const dim = self.dim;
        std.debug.assert(latents.len == dim * num_frames);

        // ||e_k||^2 for every entry of every codebook
        const entry_norms = try allocator.alloc(f32, num_stages * self.codebook_size);
        defer allocator.free(entry_norms);
        for (self.codebooks, 0..) |codebook, stage| {
            for (0..self.codebook_size) |k| {
                var sq: f32 = 0.0;
                for (codebook[k * dim ..][0..dim]) |e| sq += e * e;
                entry_norms[stage * self.codebook_size + k] = sq;
            }
        }

        const residual = try allocator.alloc(f32, dim);
        defer allocator.free(residual);

        for (out, 0..) |*frame, t| {
            for (residual, 0..) |*r, c| r.* = latents[c * num_frames + t];

            for (frame, self.codebooks, 0..) |*token, codebook, stage| {
                var r_sq: f32 = 0.0;
                for (residual) |r| r_sq += r * r;

                var best_k: usize = 0;
                var best_dist = std.math.inf(f32);
                for (0..self.codebook_size) |k| {
                    var dot: f32 = 0.0;
                    for (residual, codebook[k * dim ..][0..dim]) |r, e| dot += r * e;
                    const dist = (r_sq - 2.0 * dot) + entry_norms[stage * self.codebook_size + k];
                    // Strict comparison keeps the first index on ties, like torch.max
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }

                token.* = @intCast(best_k);
                for (residual, codebook[best_k * dim ..][0..dim]) |*r, e| r.* -= e;
            }
        }
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

test "Quantizer.encode picks nearest entries stage by stage" {
    // codebook_size = 3, dim = 2; every stage shares the same codebook.
    const codebook = [_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
    const quantizer = try Quantizer.init(@splat(&codebook), 2);

    // Layout is [dim, frames]: frame 0 = (1.25, 0.125), frame 1 = (0.5, 0.5)
    const latents = [_]f32{ 1.25, 0.5, 0.125, 0.5 };
    var frames: [2]Frame = undefined;
    try quantizer.encode(std.testing.allocator, &latents, &frames);

    // Frame 0: (1, 0) first, then the residual (0.25, 0.125) is closest to (0, 0)
    try std.testing.expectEqual(Frame{ 1, 0, 0, 0, 0, 0, 0, 0 }, frames[0]);
    // Frame 1: all three entries are equally far, so the first index wins
    try std.testing.expectEqual(Frame{ 0, 0, 0, 0, 0, 0, 0, 0 }, frames[1]);
}

test "writeFrames round-trips through framesFromBytes" {
    const frames = [_]Frame{ .{ 1, 2, 3, 4, 5, 6, 7, 8 }, .{ 1023, 0, 512, 7, 9, 11, 13, 1000 } };
    var buf: [2 * @sizeOf(Frame)]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeFrames(&w, &frames);

    const parsed = try framesFromBytes(std.testing.allocator, w.buffered());
    defer std.testing.allocator.free(parsed);
    try std.testing.expectEqualSlices(Frame, &frames, parsed);
}
