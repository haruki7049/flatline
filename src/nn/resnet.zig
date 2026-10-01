const std = @import("std");
const Allocator = std.mem.Allocator;
const activation = @import("activation.zig");
const Conv1d = @import("conv.zig").Conv1d;

/// SEANet residual block with pre-activation:
/// out = shortcut(x) + conv2(elu(conv1(elu(x)))).
/// Tensors are channel-major: [channels, time].
pub const ResnetBlock = struct {
    /// block.1: [dim] -> [hidden], kernel 3 (block.0 is ELU)
    conv1: Conv1d,
    /// block.3: [hidden] -> [dim], kernel 1 (block.2 is ELU)
    conv2: Conv1d,
    /// [dim] -> [dim], kernel 1
    shortcut: Conv1d,

    pub fn init(conv1: Conv1d, conv2: Conv1d, shortcut: Conv1d) !ResnetBlock {
        if (conv1.c_out != conv2.c_in) return error.ShapeMismatch;
        if (conv1.c_in != shortcut.c_in) return error.ShapeMismatch;
        if (conv2.c_out != shortcut.c_out) return error.ShapeMismatch;
        if (shortcut.c_in != shortcut.c_out) return error.ShapeMismatch;
        return .{ .conv1 = conv1, .conv2 = conv2, .shortcut = shortcut };
    }

    pub fn deinit(self: ResnetBlock, allocator: Allocator) void {
        self.conv1.deinit(allocator);
        self.conv2.deinit(allocator);
        self.shortcut.deinit(allocator);
    }

    pub fn channels(self: ResnetBlock) usize {
        return self.shortcut.c_out;
    }

    /// `in` and `out` are [channels, t_len] and must not alias.
    pub fn forward(self: ResnetBlock, allocator: Allocator, in: []const f32, t_len: usize, out: []f32) !void {
        const mid = try allocator.alloc(f32, self.conv1.c_out * t_len);
        defer allocator.free(mid);
        // Holds elu(x) first, then the residual branch output (both [channels, t_len]).
        const res = try allocator.alloc(f32, self.conv2.c_out * t_len);
        defer allocator.free(res);

        self.shortcut.forward(in, t_len, out);

        @memcpy(res, in);
        activation.eluInPlace(res);
        self.conv1.forward(res, t_len, mid);
        activation.eluInPlace(mid);
        self.conv2.forward(mid, t_len, res);

        for (out, res) |*o, r| {
            o.* += r;
        }
    }
};

test "ResnetBlock adds the residual branch to the shortcut" {
    const allocator = std.testing.allocator;

    // dim = 1, hidden = 1. shortcut doubles, conv1 is a causal identity, conv2 negates.
    const conv1 = try Conv1d.initWeightNorm(allocator, .{
        .weight_v = &.{ 0.0, 0.0, 1.0 },
        .weight_g = &.{1.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 3 },
    }, 1);
    const conv2 = try Conv1d.initWeightNorm(allocator, .{
        .weight_v = &.{-1.0},
        .weight_g = &.{1.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 1 },
    }, 1);
    const shortcut = try Conv1d.initWeightNorm(allocator, .{
        .weight_v = &.{1.0},
        .weight_g = &.{2.0},
        .bias = &.{0.0},
        .shape = .{ 1, 1, 1 },
    }, 1);
    const block = try ResnetBlock.init(conv1, conv2, shortcut);
    defer block.deinit(allocator);

    const in = [_]f32{ 1.0, -1.0 };
    var out: [2]f32 = undefined;
    try block.forward(allocator, &in, 2, &out);

    // x = 1: 2 - 1 = 1; x = -1: -2 - elu(elu(-1))
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(-2.0 - activation.elu(activation.elu(-1.0)), out[1], 1e-6);
}
