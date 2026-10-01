//! Loaders for the SEANet building blocks shared by the EnCodec encoder and decoder.

const std = @import("std");
const Allocator = std.mem.Allocator;
const nn = @import("nn.zig");
const SafeTensors = @import("model.zig").SafeTensors;

/// Weight-normalized convolution parameters under `prefix` (e.g. "decoder.layers.0.conv").
pub fn loadWeightNorm(weights: *const SafeTensors, comptime prefix: []const u8, args: anytype) !nn.WeightNormParams {
    const v = try weights.tensorFmt(prefix ++ ".weight_v", args);
    const g = try weights.tensorFmt(prefix ++ ".weight_g", args);
    const bias = try weights.tensorFmt(prefix ++ ".bias", args);
    if (v.rank != 3) return error.ShapeMismatch;
    return .{
        .weight_v = v.data,
        .weight_g = g.data,
        .bias = bias.data,
        .shape = v.shape()[0..3].*,
    };
}

pub fn loadConv1d(
    allocator: Allocator,
    weights: *const SafeTensors,
    stride: usize,
    comptime prefix: []const u8,
    args: anytype,
) !nn.Conv1d {
    return nn.Conv1d.initWeightNorm(allocator, try loadWeightNorm(weights, prefix, args), stride);
}

/// SEANetResnetBlock under `prefix` (e.g. "decoder.layers.4").
pub fn loadResnetBlock(allocator: Allocator, weights: *const SafeTensors, comptime prefix: []const u8, args: anytype) !nn.ResnetBlock {
    return nn.ResnetBlock.init(
        try loadConv1d(allocator, weights, 1, prefix ++ ".block.1.conv", args),
        try loadConv1d(allocator, weights, 1, prefix ++ ".block.3.conv", args),
        try loadConv1d(allocator, weights, 1, prefix ++ ".shortcut.conv", args),
    );
}

/// 2-layer SLSTM under `prefix` (e.g. "decoder.layers.1.lstm").
pub fn loadLstm(weights: *const SafeTensors, comptime prefix: []const u8) !nn.Lstm(2) {
    var params: [2]nn.LstmParams = undefined;
    for (&params, 0..) |*p, l| {
        p.* = .{
            .weight_ih = (try weights.tensorFmt(prefix ++ ".weight_ih_l{d}", .{l})).data,
            .weight_hh = (try weights.tensorFmt(prefix ++ ".weight_hh_l{d}", .{l})).data,
            .bias_ih = (try weights.tensorFmt(prefix ++ ".bias_ih_l{d}", .{l})).data,
            .bias_hh = (try weights.tensorFmt(prefix ++ ".bias_hh_l{d}", .{l})).data,
        };
    }
    return nn.Lstm(2).init(params);
}
