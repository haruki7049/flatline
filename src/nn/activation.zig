const std = @import("std");

pub fn sigmoid(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return 0.0;
    return 1.0 / (1.0 + @exp(-x));
}

pub fn tanh(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return -1.0;
    return std.math.tanh(x);
}

// ELU activation: x if x > 0 else (exp(x) - 1)
pub fn elu(x: f32) f32 {
    if (x > 0.0) return x;
    return @exp(x) - 1.0;
}

pub fn eluInPlace(xs: []f32) void {
    for (xs) |*x| {
        x.* = elu(x.*);
    }
}

test "sigmoid saturates and is centered at 0.5" {
    try std.testing.expectEqual(@as(f32, 0.5), sigmoid(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), sigmoid(20.0));
    try std.testing.expectEqual(@as(f32, 0.0), sigmoid(-20.0));
}

test "tanh saturates" {
    try std.testing.expectEqual(@as(f32, 0.0), tanh(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), tanh(25.0));
    try std.testing.expectEqual(@as(f32, -1.0), tanh(-25.0));
}

test "elu keeps positives and bends negatives" {
    var xs = [_]f32{ 2.0, 0.0, -1.0 };
    eluInPlace(&xs);
    try std.testing.expectEqual(@as(f32, 2.0), xs[0]);
    try std.testing.expectEqual(@as(f32, 0.0), xs[1]);
    try std.testing.expectApproxEqAbs(@as(f32, @exp(@as(f32, -1.0)) - 1.0), xs[2], 1e-7);
}
