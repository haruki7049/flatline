const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const vec_len = 8;
const Vec = @Vector(vec_len, f32);

/// Conv1d output samples computed per register block.
const block_len = 2 * vec_len;
/// Output samples per cache tile: the input a tile reads stays in L2 while every output
/// channel is computed. A multiple of every block size.
const tile_len = 8 * block_len;

const has_fma = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .aarch64 => true,
    else => false,
};

/// a * b + c, fused only when the target has FMA (a software fma would be far slower).
inline fn mulAdd(a: Vec, b: Vec, c: Vec) Vec {
    return if (has_fma) @mulAdd(Vec, a, b, c) else a * b + c;
}

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

    /// Output channels computed together; they share every input load.
    const co_block = 4;

    /// `in` is [c_in, t_in], `out` is [c_out, outputLen(t_in)]. `allocator` holds the padded
    /// input for the duration of the call.
    pub fn forward(self: Conv1d, allocator: Allocator, in: []const f32, t_in: usize, out: []f32) Allocator.Error!void {
        const t_out = self.outputLen(t_in);
        std.debug.assert(in.len == self.c_in * t_in);
        std.debug.assert(out.len == self.c_out * t_out);

        // Blocks read up to tap (kernel_size - 1) / stride past the last block start.
        const row_len = std.mem.alignForward(usize, t_out, block_len) + (self.kernel_size - 1) / self.stride;
        const padded = try allocator.alloc(f32, self.c_in * self.stride * row_len);
        defer allocator.free(padded);
        self.padInput(in, t_in, t_out, row_len, padded);

        var tile_start: usize = 0;
        while (tile_start < t_out) : (tile_start += tile_len) {
            const tile_end = @min(tile_start + tile_len, t_out);
            var co: usize = 0;
            while (co < self.c_out) : (co += co_block) {
                switch (@min(co_block, self.c_out - co)) {
                    inline 1...co_block => |n_co| {
                        var t = tile_start;
                        while (t < tile_end) : (t += block_len) {
                            self.block(n_co, padded, row_len, co, t, out, t_out);
                        }
                    },
                    else => unreachable,
                }
            }
        }
    }

    /// Writes the reflect-padded input (see SConv1d) in polyphase layout: row r of channel
    /// ci holds padded[u * stride + r] at index u, so tap k of output t is at index
    /// t + k / stride of row k % stride, contiguous in t. Samples past the padded input are 0.
    fn padInput(self: Conv1d, in: []const f32, t_in: usize, t_out: usize, row_len: usize, padded: []f32) void {
        const stride = self.stride;
        // get_extra_padding_for_conv1d: pad the right so the last window is full
        const pad_left = self.kernel_size - stride;
        const pad_right = t_out * stride - t_in;
        const padded_len = pad_left + t_in + pad_right;

        for (0..self.c_in) |ci| {
            const x = in[ci * t_in ..][0..t_in];
            for (0..stride) |r| {
                const row = padded[(ci * stride + r) * row_len ..][0..row_len];
                for (row, 0..) |*v, u| {
                    const j = u * stride + r;
                    v.* = if (j >= padded_len)
                        0.0
                    else if (j >= pad_left and j - pad_left < t_in)
                        x[j - pad_left]
                    else
                        reflectPadded(x, pad_left, pad_right, j);
                }
            }
        }
    }

    /// Computes outputs [t, t + block_len) of channels [co, co + n_co), keeping the
    /// accumulators in registers across all input channels and taps.
    inline fn block(
        self: Conv1d,
        comptime n_co: usize,
        padded: []const f32,
        row_len: usize,
        co: usize,
        t: usize,
        out: []f32,
        t_out: usize,
    ) void {
        const c_in = self.c_in;
        const kernel_size = self.kernel_size;
        const stride = self.stride;

        var acc: [n_co][2]Vec = undefined;
        inline for (&acc, 0..) |*a, c| a.* = @splat(@splat(self.bias[co + c]));

        for (0..c_in) |ci| {
            for (0..stride) |r| {
                const row = padded[(ci * stride + r) * row_len + t ..];
                var k = r;
                var m: usize = 0;
                while (k < kernel_size) : ({
                    k += stride;
                    m += 1;
                }) {
                    const x0: Vec = row[m..][0..vec_len].*;
                    const x1: Vec = row[m + vec_len ..][0..vec_len].*;
                    inline for (&acc, 0..) |*a, c| {
                        const w: Vec = @splat(self.weight[((co + c) * c_in + ci) * kernel_size + k]);
                        a[0] = mulAdd(w, x0, a[0]);
                        a[1] = mulAdd(w, x1, a[1]);
                    }
                }
            }
        }

        const n = @min(block_len, t_out - t);
        inline for (acc, 0..) |a, c| {
            var vals: [block_len]f32 = undefined;
            vals[0..vec_len].* = a[0];
            vals[vec_len..].* = a[1];
            @memcpy(out[(co + c) * t_out + t ..][0..n], vals[0..n]);
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
///
/// Computed per output phase: output t * stride + p is the sum over input channels and
/// m of w[p + m * stride] * x[t - m], so each phase is a causal convolution over the input.
pub const ConvTranspose1d = struct {
    /// [c_out, c_in, taps() * stride]: the checkpoint's [c_in, c_out, kernel_size] transposed
    /// and zero-padded to whole taps.
    weight: []f32,
    bias: []const f32,
    c_in: usize,
    c_out: usize,
    kernel_size: usize,
    stride: usize,

    /// Largest supported stride: each output phase takes accumulator registers.
    pub const max_stride = 8;

    pub fn initWeightNorm(allocator: Allocator, p: WeightNormParams, stride: usize) !ConvTranspose1d {
        const c_in, const c_out, const kernel_size = p.shape;
        if (p.bias.len != c_out) return error.ShapeMismatch;
        if (stride == 0 or kernel_size < stride) return error.ShapeMismatch;
        if (stride > max_stride) return error.UnsupportedStride;

        const normalized = try weightNorm(allocator, p);
        defer allocator.free(normalized);

        const w_len = (std.math.divCeil(usize, kernel_size, stride) catch unreachable) * stride;
        const weight = try allocator.alloc(f32, c_out * c_in * w_len);
        @memset(weight, 0.0);
        for (0..c_in) |ci| {
            for (0..c_out) |co| {
                @memcpy(
                    weight[(co * c_in + ci) * w_len ..][0..kernel_size],
                    normalized[(ci * c_out + co) * kernel_size ..][0..kernel_size],
                );
            }
        }

        return .{
            .weight = weight,
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

    /// Input samples each output phase reads: x[t], x[t - 1], ..., x[t - taps() + 1].
    fn taps(self: ConvTranspose1d) usize {
        return std.math.divCeil(usize, self.kernel_size, self.stride) catch unreachable;
    }

    /// `in` is [c_in, t_in], `out` is [c_out, outputLen(t_in)]. `allocator` holds the padded
    /// input for the duration of the call.
    pub fn forward(self: ConvTranspose1d, allocator: Allocator, in: []const f32, t_in: usize, out: []f32) Allocator.Error!void {
        std.debug.assert(in.len == self.c_in * t_in);
        std.debug.assert(out.len == self.c_out * self.outputLen(t_in));

        // taps() - 1 zeros before t = 0, and zeros after the input up to a whole tile.
        // Outputs past t_in * stride, the causal trim, are never computed.
        const lead = self.taps() - 1;
        const row_len = lead + std.mem.alignForward(usize, t_in, tile_len);
        const padded = try allocator.alloc(f32, self.c_in * row_len);
        defer allocator.free(padded);
        for (0..self.c_in) |ci| {
            const row = padded[ci * row_len ..][0..row_len];
            @memset(row[0..lead], 0.0);
            @memcpy(row[lead..][0..t_in], in[ci * t_in ..][0..t_in]);
            @memset(row[lead + t_in ..], 0.0);
        }

        switch (self.stride) {
            inline 1...max_stride => |stride| self.forwardStride(stride, padded, row_len, t_in, out),
            else => unreachable,
        }
    }

    fn forwardStride(
        self: ConvTranspose1d,
        comptime stride: usize,
        padded: []const f32,
        row_len: usize,
        t_in: usize,
        out: []f32,
    ) void {
        // Input vectors per block, so that about 8 accumulators are live.
        const n_vec: usize = @min(4, @max(1, 8 / stride));

        var tile_start: usize = 0;
        while (tile_start < t_in) : (tile_start += tile_len) {
            const tile_end = @min(tile_start + tile_len, t_in);
            for (0..self.c_out) |co| {
                var q = tile_start;
                while (q < tile_end) : (q += n_vec * vec_len) {
                    self.block(stride, n_vec, padded, row_len, co, q, t_in, out);
                }
            }
        }
    }

    /// Computes every phase of input positions [q, q + n_vec * vec_len) for channel co,
    /// i.e. outputs [q * stride, (q + n_vec * vec_len) * stride).
    inline fn block(
        self: ConvTranspose1d,
        comptime stride: usize,
        comptime n_vec: usize,
        padded: []const f32,
        row_len: usize,
        co: usize,
        q: usize,
        t_in: usize,
        out: []f32,
    ) void {
        const n_taps = self.taps();
        const w_len = n_taps * stride;
        const w = self.weight[(co * self.c_in) * w_len ..][0 .. self.c_in * w_len];

        var acc: [n_vec][stride]Vec = @splat(@splat(@splat(self.bias[co])));

        for (0..self.c_in) |ci| {
            const w_ci = w[ci * w_len ..][0..w_len];
            for (0..n_taps) |m| {
                // x[q - m ...] sits at row index lead + q - m
                const x_m = padded[ci * row_len + (n_taps - 1 - m) + q ..];
                var x: [n_vec]Vec = undefined;
                inline for (&x, 0..) |*v, j| v.* = x_m[j * vec_len ..][0..vec_len].*;
                inline for (0..stride) |p| {
                    const w_p: Vec = @splat(w_ci[m * stride + p]);
                    inline for (&acc, x) |*a, v| a[p] = mulAdd(w_p, v, a[p]);
                }
            }
        }

        // Interleave the phases into time order
        var vals: [n_vec * vec_len * stride]f32 = undefined;
        inline for (acc, 0..) |a, j| {
            inline for (a, 0..) |phase, p| {
                const lanes: [vec_len]f32 = phase;
                for (lanes, 0..) |v, i| vals[(j * vec_len + i) * stride + p] = v;
            }
        }
        const n = @min(n_vec * vec_len, t_in - q) * stride;
        @memcpy(out[co * t_in * stride + q * stride ..][0..n], vals[0..n]);
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
    try conv.forward(std.testing.allocator, &in, 4, &out);
    const expected = [_]f32{ 6.5, 5.5, 6.5, 9.5 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

    // Too short to reflect: [1, 2] is extended to [1, 2, 0], so the padded input is [0, 2 | 1, 2]
    const short = [_]f32{ 1.0, 2.0 };
    var short_out: [2]f32 = undefined;
    try conv.forward(std.testing.allocator, &short, 2, &short_out);
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
    try sum4.forward(std.testing.allocator, &in, in.len, &out);
    const expected = [_]f32{ 8.0, 10.0, 16.0 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

    // Too short to reflect: [1] is extended to [1, 0, 0]. Padded input: [0, 0 | 1 | 0]
    const single = [_]f32{1.0};
    var single_out: [1]f32 = undefined;
    try sum4.forward(std.testing.allocator, &single, 1, &single_out);
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
    try conv.forward(std.testing.allocator, &in, in.len, &out);
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
    try conv.forward(std.testing.allocator, &in, 2, &out);

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
    try conv.forward(std.testing.allocator, &in, 2, &out);

    const expected = [_]f32{ 1.0, 1.0, -2.0, -2.0 };
    for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-6);
}

/// Direct evaluation of SConv1d (pad1d reflect + conv1d) on checkpoint-layout weights.
fn referenceConv1d(w: []const f32, bias: []const f32, shape: [3]usize, stride: usize, in: []const f32, t_in: usize, out: []f32) void {
    const c_out, const c_in, const kernel_size = shape;
    const t_out = std.math.divCeil(usize, t_in, stride) catch unreachable;
    const pad_left = kernel_size - stride;
    const pad_right = t_out * stride - t_in;
    for (0..c_out) |co| {
        for (0..t_out) |t| {
            var sum: f32 = bias[co];
            for (0..c_in) |ci| {
                const x = in[ci * t_in ..][0..t_in];
                for (0..kernel_size) |k| {
                    const j = t * stride + k;
                    const v = if (j >= pad_left and j - pad_left < t_in) x[j - pad_left] else reflectPadded(x, pad_left, pad_right, j);
                    sum += v * w[(co * c_in + ci) * kernel_size + k];
                }
            }
            out[co * t_out + t] = sum;
        }
    }
}

/// Direct evaluation of conv_transpose1d with the right end trimmed, on checkpoint-layout weights.
fn referenceConvTranspose1d(w: []const f32, bias: []const f32, shape: [3]usize, stride: usize, in: []const f32, t_in: usize, out: []f32) void {
    const c_in, const c_out, const kernel_size = shape;
    const t_out = t_in * stride;
    for (0..c_out) |co| @memset(out[co * t_out ..][0..t_out], bias[co]);
    for (0..c_in) |ci| {
        for (0..c_out) |co| {
            for (0..t_in) |ti| {
                for (0..kernel_size) |k| {
                    const to = ti * stride + k;
                    if (to >= t_out) break;
                    out[co * t_out + to] += in[ci * t_in + ti] * w[(ci * c_out + co) * kernel_size + k];
                }
            }
        }
    }
}

fn fillRandom(random: std.Random, buf: []f32) void {
    for (buf) |*v| v.* = random.float(f32) * 2.0 - 1.0;
}

test "Conv1d matches direct evaluation across shapes, strides and lengths" {
    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();

    // c_out, c_in, kernel_size, stride, t_in: covers channel and time tails, several tiles,
    // and inputs too short to reflect.
    const cases = [_][5]usize{
        .{ 5, 3, 7, 1, 37 },  .{ 6, 2, 3, 1, 1 },  .{ 1, 5, 7, 1, 300 },
        .{ 9, 4, 1, 1, 129 }, .{ 3, 4, 4, 2, 33 }, .{ 7, 2, 16, 8, 50 },
        .{ 9, 1, 10, 5, 3 },  .{ 2, 3, 8, 4, 1 },  .{ 4, 2, 5, 3, 290 },
    };
    for (cases) |case| {
        const c_out, const c_in, const kernel_size, const stride, const t_in = case;
        const shape = [3]usize{ c_out, c_in, kernel_size };
        const v = try allocator.alloc(f32, c_out * c_in * kernel_size);
        defer allocator.free(v);
        const g = try allocator.alloc(f32, c_out);
        defer allocator.free(g);
        const bias = try allocator.alloc(f32, c_out);
        defer allocator.free(bias);
        const in = try allocator.alloc(f32, c_in * t_in);
        defer allocator.free(in);
        for ([_][]f32{ v, g, bias, in }) |buf| fillRandom(random, buf);

        const conv = try Conv1d.initWeightNorm(allocator, .{ .weight_v = v, .weight_g = g, .bias = bias, .shape = shape }, stride);
        defer conv.deinit(allocator);

        const t_out = conv.outputLen(t_in);
        const out = try allocator.alloc(f32, c_out * t_out);
        defer allocator.free(out);
        const expected = try allocator.alloc(f32, c_out * t_out);
        defer allocator.free(expected);
        try conv.forward(allocator, in, t_in, out);
        referenceConv1d(conv.weight, bias, shape, stride, in, t_in, expected);
        for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "ConvTranspose1d matches direct evaluation across shapes, strides and lengths" {
    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();

    // c_in, c_out, kernel_size, stride, t_in: covers every stride up to max_stride, kernels
    // that are not a multiple of the stride, and several tiles.
    const cases = [_][5]usize{
        .{ 3, 5, 16, 8, 20 }, .{ 4, 3, 10, 5, 9 }, .{ 2, 6, 8, 4, 140 },
        .{ 3, 2, 4, 2, 150 }, .{ 2, 3, 3, 1, 5 },  .{ 2, 3, 5, 2, 17 },
        .{ 1, 2, 7, 7, 1 },   .{ 2, 2, 9, 3, 33 }, .{ 3, 1, 6, 6, 130 },
    };
    for (cases) |case| {
        const c_in, const c_out, const kernel_size, const stride, const t_in = case;
        const shape = [3]usize{ c_in, c_out, kernel_size };
        const v = try allocator.alloc(f32, c_in * c_out * kernel_size);
        defer allocator.free(v);
        const g = try allocator.alloc(f32, c_in);
        defer allocator.free(g);
        const bias = try allocator.alloc(f32, c_out);
        defer allocator.free(bias);
        const in = try allocator.alloc(f32, c_in * t_in);
        defer allocator.free(in);
        for ([_][]f32{ v, g, bias, in }) |buf| fillRandom(random, buf);

        const params: WeightNormParams = .{ .weight_v = v, .weight_g = g, .bias = bias, .shape = shape };
        const conv = try ConvTranspose1d.initWeightNorm(allocator, params, stride);
        defer conv.deinit(allocator);
        const w = try weightNorm(allocator, params);
        defer allocator.free(w);

        const t_out = conv.outputLen(t_in);
        const out = try allocator.alloc(f32, c_out * t_out);
        defer allocator.free(out);
        const expected = try allocator.alloc(f32, c_out * t_out);
        defer allocator.free(expected);
        try conv.forward(allocator, in, t_in, out);
        referenceConvTranspose1d(w, bias, shape, stride, in, t_in, expected);
        for (expected, out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "ConvTranspose1d rejects strides above max_stride" {
    try std.testing.expectError(error.UnsupportedStride, ConvTranspose1d.initWeightNorm(std.testing.allocator, .{
        .weight_v = &([_]f32{1.0} ** 9),
        .weight_g = &.{1.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 9 },
    }, 9));
}
