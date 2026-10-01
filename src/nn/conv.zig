const std = @import("std");
const Allocator = std.mem.Allocator;

/// Raw weight-normalized convolution parameters, as stored in the checkpoint.
pub const WeightNormParams = struct {
    weight_v: []const f32,
    /// One gain per slice along dimension 0 of `weight_v`.
    weight_g: []const f32,
    bias: []const f32,
    /// Shape of `weight_v`: [c_out, c_in, kernel] for Conv1d,
    /// [c_in, c_out, kernel] for ConvTranspose1d.
    shape: [3]usize,
};

// Normalizes weights per slice along dimension 0: W = g * (v / ||v||)
fn weightNorm(allocator: Allocator, p: WeightNormParams) ![]f32 {
    const slice_len = p.shape[1] * p.shape[2];
    if (p.weight_v.len != p.shape[0] * slice_len) return error.ShapeMismatch;
    if (p.weight_g.len != p.shape[0]) return error.ShapeMismatch;

    const norm = try allocator.alloc(f32, p.weight_v.len);

    for (0..p.shape[0]) |i| {
        var norm_sq: f32 = 0.0;
        const base = i * slice_len;

        for (0..slice_len) |idx| {
            const val = p.weight_v[base + idx];
            norm_sq += val * val;
        }

        const scale = p.weight_g[i] / @sqrt(norm_sq);
        for (0..slice_len) |idx| {
            norm[base + idx] = p.weight_v[base + idx] * scale;
        }
    }

    return norm;
}

/// Causal 1D convolution, matching EnCodec's SConv1d (causal=True, pad_mode='reflect').
/// The input is reflect-padded by kernel_size - stride on the left, plus on the right just
/// enough for the last window to be full, so the output length is ceil(t_in / stride).
/// Tensors are channel-major: [channels, time].
pub const Conv1d = struct {
    /// [c_out, c_in, kernel_size]
    weight: []f32,
    bias: []const f32,
    c_in: usize,
    c_out: usize,
    kernel_size: usize,
    stride: usize,

    pub fn initWeightNorm(allocator: Allocator, p: WeightNormParams, stride: usize) !Conv1d {
        const c_out, const c_in, const kernel_size = p.shape;
        if (p.bias.len != c_out) return error.ShapeMismatch;
        if (stride == 0 or kernel_size < stride) return error.ShapeMismatch;
        return .{
            .weight = try weightNorm(allocator, p),
            .bias = p.bias,
            .c_in = c_in,
            .c_out = c_out,
            .kernel_size = kernel_size,
            .stride = stride,
        };
    }

    pub fn deinit(self: Conv1d, allocator: Allocator) void {
        allocator.free(self.weight);
    }

    pub fn outputLen(self: Conv1d, t_in: usize) usize {
        return std.math.divCeil(usize, t_in, self.stride) catch unreachable;
    }

    /// `in` is [c_in, t_in], `out` is [c_out, outputLen(t_in)].
    pub fn forward(self: Conv1d, in: []const f32, t_in: usize, out: []f32) void {
        const c_in = self.c_in;
        const kernel_size = self.kernel_size;
        const stride = self.stride;
        const t_out = self.outputLen(t_in);

        std.debug.assert(in.len == c_in * t_in);
        std.debug.assert(out.len == self.c_out * t_out);

        // get_extra_padding_for_conv1d: pad the right so the last window is full
        const pad_left = kernel_size - stride;
        const pad_right = t_out * stride - t_in;

        for (0..self.c_out) |co| {
            const bias_val = self.bias[co];
            const w_co = self.weight[co * (c_in * kernel_size) .. (co + 1) * (c_in * kernel_size)];

            for (0..t_out) |t| {
                var sum: f32 = bias_val;

                for (0..c_in) |ci| {
                    const in_channel = in[ci * t_in .. (ci + 1) * t_in];
                    const w_channel = w_co[ci * kernel_size .. (ci + 1) * kernel_size];

                    for (0..kernel_size) |k| {
                        // Position in the padded input; the first `pad_left` samples are padding.
                        const j = t * stride + k;
                        const in_val = if (j >= pad_left and j - pad_left < t_in)
                            in_channel[j - pad_left]
                        else
                            reflectPadded(in_channel, pad_left, pad_right, j);
                        sum += in_val * w_channel[k];
                    }
                }

                out[co * t_out + t] = sum;
            }
        }
    }
};

/// Element `j` of EnCodec's pad1d(x, (pad_left, pad_right), mode='reflect').
/// Like pad1d, an input too short to reflect is first extended with zeros on the right
/// (and those zeros are dropped again from the end of the result).
fn reflectPadded(x: []const f32, pad_left: usize, pad_right: usize, j: usize) f32 {
    const max_pad = @max(pad_left, pad_right);
    const ext_len = if (x.len <= max_pad) max_pad + 1 else x.len;
    const i = if (j < pad_left)
        pad_left - j
    else if (j - pad_left < ext_len)
        j - pad_left
    else
        ext_len - 2 - (j - pad_left - ext_len);
    return if (i < x.len) x[i] else 0.0;
}

/// Causal 1D transposed convolution (upsampling by `stride`), matching EnCodec's
/// SConvTranspose1d (causal=True, trim_right_ratio=1.0): the kernel_size - stride
/// surplus samples are trimmed from the right end.
/// Tensors are channel-major: [channels, time].
pub const ConvTranspose1d = struct {
    /// [c_in, c_out, kernel_size]
    weight: []f32,
    bias: []const f32,
    c_in: usize,
    c_out: usize,
    kernel_size: usize,
    stride: usize,

    pub fn initWeightNorm(allocator: Allocator, p: WeightNormParams, stride: usize) !ConvTranspose1d {
        const c_in, const c_out, const kernel_size = p.shape;
        if (p.bias.len != c_out) return error.ShapeMismatch;
        if (kernel_size < stride) return error.ShapeMismatch;
        return .{
            .weight = try weightNorm(allocator, p),
            .bias = p.bias,
            .c_in = c_in,
            .c_out = c_out,
            .kernel_size = kernel_size,
            .stride = stride,
        };
    }

    pub fn deinit(self: ConvTranspose1d, allocator: Allocator) void {
        allocator.free(self.weight);
    }

    pub fn outputLen(self: ConvTranspose1d, t_in: usize) usize {
        return t_in * self.stride;
    }

    /// `in` is [c_in, t_in], `out` is [c_out, outputLen(t_in)].
    pub fn forward(self: ConvTranspose1d, in: []const f32, t_in: usize, out: []f32) void {
        const c_out = self.c_out;
        const kernel_size = self.kernel_size;
        const stride = self.stride;
        const t_out = self.outputLen(t_in);

        std.debug.assert(in.len == self.c_in * t_in);
        std.debug.assert(out.len == c_out * t_out);

        for (0..c_out) |co| {
            const bias_val = self.bias[co];
            for (0..t_out) |to| {
                out[co * t_out + to] = bias_val;
            }
        }

        for (0..self.c_in) |ci| {
            for (0..c_out) |co| {
                const w_base = ci * (c_out * kernel_size) + co * kernel_size;
                const w_slice = self.weight[w_base .. w_base + kernel_size];

                for (0..t_in) |ti| {
                    const in_val = in[ci * t_in + ti];
                    if (in_val == 0.0) continue;

                    for (0..kernel_size) |k| {
                        const out_time = ti * stride + k;
                        if (out_time >= t_out) break;
                        out[co * t_out + out_time] += in_val * w_slice[k];
                    }
                }
            }
        }
    }
};

test "weight norm rescales each slice to its gain" {
    const conv = try Conv1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &.{ 3.0, 4.0, 0.0, 2.0 },
        .weight_g = &.{ 10.0, 1.0 },
        .bias = &.{ 0.0, 0.0 },
        .shape = .{ 2, 1, 2 },
    }, 1);
    defer conv.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(f32, &.{ 6.0, 8.0, 0.0, 1.0 }, conv.weight);
}

test "weight norm rejects mismatched shapes" {
    try std.testing.expectError(error.ShapeMismatch, Conv1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &.{ 1.0, 2.0, 3.0 },
        .weight_g = &.{1.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 2 },
    }, 1));
}

test "Conv1d reflect-pads on the left only" {
    // Single channel, kernel [1, 1, 1] (norm sqrt(3), gain sqrt(3)) is a 3-tap moving sum.
    const conv = try Conv1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &.{ 1.0, 1.0, 1.0 },
        .weight_g = &.{@sqrt(@as(f32, 3.0))},
        .bias = &.{0.5},
        .shape = .{ 1, 1, 3 },
    }, 1);
    defer conv.deinit(std.testing.allocator);

    // Padded input: [3, 2 | 1, 2, 3, 4]
    const in = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    conv.forward(&in, 4, &out);
    const expected = [_]f32{ 6.5, 5.5, 6.5, 9.5 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

    // Too short to reflect: [1, 2] is extended to [1, 2, 0], so the padded input is [0, 2 | 1, 2]
    const short = [_]f32{ 1.0, 2.0 };
    var short_out: [2]f32 = undefined;
    conv.forward(&short, 2, &short_out);
    const short_expected = [_]f32{ 3.5, 5.5 };
    for (short_expected, short_out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
}

// Expected values below were checked against encodec.modules.conv.pad1d + F.conv1d.
test "Conv1d with stride reflect-pads both ends like SConv1d" {
    const allocator = std.testing.allocator;

    // kernel [1, 1, 1, 1], stride 2. Padded input: [3, 2 | 1, 2, 3, 4, 5 | 4]
    const sum4 = try Conv1d.initWeightNorm(allocator, .{
        .weight_v = &.{ 1.0, 1.0, 1.0, 1.0 },
        .weight_g = &.{2.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 4 },
    }, 2);
    defer sum4.deinit(allocator);

    const in = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectEqual(@as(usize, 3), sum4.outputLen(in.len));
    var out: [3]f32 = undefined;
    sum4.forward(&in, in.len, &out);
    const expected = [_]f32{ 8.0, 10.0, 16.0 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

    // Too short to reflect: [1] is extended to [1, 0, 0]. Padded input: [0, 0 | 1 | 0]
    const single = [_]f32{1.0};
    var single_out: [1]f32 = undefined;
    sum4.forward(&single, 1, &single_out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), single_out[0], 1e-5);
}

test "Conv1d with stride pads short inputs with zeros before reflecting" {
    const allocator = std.testing.allocator;

    // kernel [1..8] (gain = norm, so weights stay 1..8), stride 4.
    // Padded input: [0, 0, 3, 2 | 1, 2, 3 | 0]
    const v = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const conv = try Conv1d.initWeightNorm(allocator, .{
        .weight_v = &v,
        .weight_g = &.{@sqrt(@as(f32, 204.0))},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 8 },
    }, 4);
    defer conv.deinit(allocator);

    const in = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [1]f32 = undefined;
    conv.forward(&in, in.len, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 55.0), out[0], 1e-4);
}

test "ConvTranspose1d trims the surplus from the right" {
    // kernel [1, 1, 1, 1] with stride 2: the full output [1, 1, 11, 11, 10, 10] loses its last 2 samples.
    const conv = try ConvTranspose1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &.{ 1.0, 1.0, 1.0, 1.0 },
        .weight_g = &.{2.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 4 },
    }, 2);
    defer conv.deinit(std.testing.allocator);

    const in = [_]f32{ 1.0, 10.0 };
    var out: [4]f32 = undefined;
    conv.forward(&in, 2, &out);

    const expected = [_]f32{ 1.0, 1.0, 11.0, 11.0 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
}

test "ConvTranspose1d upsamples by stride" {
    // kernel [1, 1] with stride 2 and no padding repeats each input sample twice.
    const conv = try ConvTranspose1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &.{ 1.0, 1.0 },
        .weight_g = &.{@sqrt(@as(f32, 2.0))},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 2 },
    }, 2);
    defer conv.deinit(std.testing.allocator);

    const in = [_]f32{ 1.0, -2.0 };
    var out: [4]f32 = undefined;
    conv.forward(&in, 2, &out);

    const expected = [_]f32{ 1.0, 1.0, -2.0, -2.0 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-6);
}
