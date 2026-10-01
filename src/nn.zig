//! Neural network layers. Tensors are flat f32 slices in channel-major [channels, time] layout.

pub const activation = @import("nn/activation.zig");
pub const conv = @import("nn/conv.zig");
pub const lstm = @import("nn/lstm.zig");
pub const resnet = @import("nn/resnet.zig");

pub const Conv1d = conv.Conv1d;
pub const ConvTranspose1d = conv.ConvTranspose1d;
pub const WeightNormParams = conv.WeightNormParams;
pub const Lstm = lstm.Lstm;
pub const LstmParams = lstm.LstmParams;
pub const ResnetBlock = resnet.ResnetBlock;

test {
    _ = activation;
    _ = conv;
    _ = lstm;
    _ = resnet;
}
