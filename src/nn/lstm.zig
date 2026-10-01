const std = @import("std");
const Allocator = std.mem.Allocator;
const activation = @import("activation.zig");

pub const LstmParams = struct {
    /// [4 * hidden, hidden], gates ordered i, f, g, o
    weight_ih: []const f32,
    /// [4 * hidden, hidden]
    weight_hh: []const f32,
    /// [4 * hidden]
    bias_ih: []const f32,
    /// [4 * hidden]
    bias_hh: []const f32,
};

/// Single unidirectional LSTM layer whose input size equals its hidden size.
/// Tensors are channel-major: [hidden, time].
pub const LstmLayer = struct {
    w_ih: []const f32,
    w_hh: []const f32,
    b_ih: []const f32,
    b_hh: []const f32,
    hidden_size: usize,

    pub fn init(p: LstmParams) !LstmLayer {
        if (p.bias_ih.len % 4 != 0) return error.ShapeMismatch;
        const h_dim = p.bias_ih.len / 4;
        if (p.bias_hh.len != 4 * h_dim) return error.ShapeMismatch;
        if (p.weight_ih.len != 4 * h_dim * h_dim) return error.ShapeMismatch;
        if (p.weight_hh.len != 4 * h_dim * h_dim) return error.ShapeMismatch;
        return .{
            .w_ih = p.weight_ih,
            .w_hh = p.weight_hh,
            .b_ih = p.bias_ih,
            .b_hh = p.bias_hh,
            .hidden_size = h_dim,
        };
    }

    /// Runs from a zero state. `in` and `out` are [hidden, t_len] and must not alias;
    /// `h_state` and `c_state` are [hidden] scratch buffers.
    pub fn forward(
        self: LstmLayer,
        in: []const f32,
        t_len: usize,
        out: []f32,
        h_state: []f32,
        c_state: []f32,
    ) void {
        const h_dim = self.hidden_size;
        std.debug.assert(in.len == h_dim * t_len);
        std.debug.assert(out.len == h_dim * t_len);
        std.debug.assert(h_state.len == h_dim and c_state.len == h_dim);

        const w_ih = self.w_ih;
        const w_hh = self.w_hh;
        const b_ih = self.b_ih;
        const b_hh = self.b_hh;

        @memset(h_state, 0.0);
        @memset(c_state, 0.0);

        for (0..t_len) |t| {
            for (0..h_dim) |d| {
                var pre_i = b_ih[d] + b_hh[d];
                var pre_f = b_ih[h_dim + d] + b_hh[h_dim + d];
                var pre_g = b_ih[2 * h_dim + d] + b_hh[2 * h_dim + d];
                var pre_o = b_ih[3 * h_dim + d] + b_hh[3 * h_dim + d];

                const row_ih_i = w_ih[d * h_dim ..];
                const row_ih_f = w_ih[(h_dim + d) * h_dim ..];
                const row_ih_g = w_ih[(2 * h_dim + d) * h_dim ..];
                const row_ih_o = w_ih[(3 * h_dim + d) * h_dim ..];

                const row_hh_i = w_hh[d * h_dim ..];
                const row_hh_f = w_hh[(h_dim + d) * h_dim ..];
                const row_hh_g = w_hh[(2 * h_dim + d) * h_dim ..];
                const row_hh_o = w_hh[(3 * h_dim + d) * h_dim ..];

                for (0..h_dim) |k| {
                    const in_val = in[k * t_len + t];
                    const prev_h = h_state[k];

                    pre_i += row_ih_i[k] * in_val + row_hh_i[k] * prev_h;
                    pre_f += row_ih_f[k] * in_val + row_hh_f[k] * prev_h;
                    pre_g += row_ih_g[k] * in_val + row_hh_g[k] * prev_h;
                    pre_o += row_ih_o[k] * in_val + row_hh_o[k] * prev_h;
                }

                const gate_i = activation.sigmoid(pre_i);
                const gate_f = activation.sigmoid(pre_f);
                const gate_g = activation.tanh(pre_g);
                const gate_o = activation.sigmoid(pre_o);

                const next_c = gate_f * c_state[d] + gate_i * gate_g;
                const next_h = gate_o * activation.tanh(next_c);

                out[d * t_len + t] = next_h;
                c_state[d] = next_c;
            }

            for (0..h_dim) |d| {
                h_state[d] = out[d * t_len + t];
            }
        }
    }
};

/// Stack of `num_layers` LSTM layers, each fed by the previous one, with a residual
/// connection around the whole stack (EnCodec's SLSTM: output = lstm(x) + x).
pub fn Lstm(comptime num_layers: usize) type {
    return struct {
        const Self = @This();

        layers: [num_layers]LstmLayer,

        pub fn init(params: [num_layers]LstmParams) !Self {
            var self: Self = undefined;
            for (&self.layers, params) |*layer, p| {
                layer.* = try LstmLayer.init(p);
                if (layer.hidden_size != self.layers[0].hidden_size) return error.ShapeMismatch;
            }
            return self;
        }

        pub fn hiddenSize(self: Self) usize {
            return self.layers[0].hidden_size;
        }

        /// `in` and `out` are [hidden, t_len] and must not alias.
        pub fn forward(self: Self, allocator: Allocator, in: []const f32, t_len: usize, out: []f32) !void {
            const h_dim = self.hiddenSize();

            const state = try allocator.alloc(f32, 2 * h_dim);
            defer allocator.free(state);
            const h_state = state[0..h_dim];
            const c_state = state[h_dim..];

            // Ping-pong buffers for the outputs of all but the last layer
            const scratch_count = @min(num_layers - 1, 2);
            const scratch = try allocator.alloc(f32, scratch_count * h_dim * t_len);
            defer allocator.free(scratch);

            var src = in;
            for (self.layers, 0..) |layer, i| {
                const dst = if (i == num_layers - 1)
                    out
                else
                    scratch[(i % 2) * h_dim * t_len ..][0 .. h_dim * t_len];
                layer.forward(src, t_len, dst, h_state, c_state);
                src = dst;
            }

            // Residual skip
            for (out, in) |*o, x| {
                o.* += x;
            }
        }
    };
}

test "LstmLayer with zero weights follows the gate equations" {
    // hidden = 1; gates i, f, g, o get pre-activations from biases only.
    const layer = try LstmLayer.init(.{
        .weight_ih = &.{ 0.0, 0.0, 0.0, 0.0 },
        .weight_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
        .bias_ih = &.{ 1.0, 0.0, 0.5, 2.0 },
        .bias_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
    });

    const in = [_]f32{ 0.0, 0.0 };
    var out: [2]f32 = undefined;
    var h: [1]f32 = undefined;
    var c: [1]f32 = undefined;
    layer.forward(&in, 2, &out, &h, &c);

    const i = activation.sigmoid(1.0);
    const f = activation.sigmoid(0.0);
    const g = activation.tanh(0.5);
    const o = activation.sigmoid(2.0);
    const c0 = i * g;
    const c1 = f * c0 + i * g;
    try std.testing.expectApproxEqAbs(o * activation.tanh(c0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(o * activation.tanh(c1), out[1], 1e-6);
}

test "Lstm stacks layers and adds the input" {
    const p: LstmParams = .{
        .weight_ih = &.{ 0.0, 0.0, 0.0, 1.0 },
        .weight_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
        .bias_ih = &.{ 1.0, 0.0, 0.5, 0.0 },
        .bias_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
    };
    const stack = try Lstm(3).init(.{ p, p, p });

    const in = [_]f32{ 0.3, -0.2 };
    var stacked: [2]f32 = undefined;
    try stack.forward(std.testing.allocator, &in, 2, &stacked);

    // Same result as running the layers one by one, plus the input
    var a: [2]f32 = undefined;
    var b: [2]f32 = undefined;
    var h: [1]f32 = undefined;
    var c: [1]f32 = undefined;
    stack.layers[0].forward(&in, 2, &a, &h, &c);
    stack.layers[1].forward(&a, 2, &b, &h, &c);
    stack.layers[2].forward(&b, 2, &a, &h, &c);
    for (&a, in) |*o, x| o.* += x;
    try std.testing.expectEqualSlices(f32, &a, &stacked);
}

test "LstmLayer rejects non-square input weights" {
    try std.testing.expectError(error.ShapeMismatch, LstmLayer.init(.{
        .weight_ih = &.{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        .weight_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
        .bias_ih = &.{ 0.0, 0.0, 0.0, 0.0 },
        .bias_hh = &.{ 0.0, 0.0, 0.0, 0.0 },
    }));
}
