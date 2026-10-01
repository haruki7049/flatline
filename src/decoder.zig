//! EnCodec 24 kHz decoder: RVQ tokens -> latents -> causal SEANet decoder -> PCM.
//! Follows encodec.modules.seanet.SEANetDecoder with causal=True, pad_mode='reflect'.

const std = @import("std");
const Allocator = std.mem.Allocator;
const nn = @import("nn.zig");
const rvq = @import("rvq.zig");
const seanet = @import("seanet.zig");
const SafeTensors = @import("model.zig").SafeTensors;

pub const sample_rate = 24000;

/// Upsampling ratio of each decoder stage.
const ratios = [_]usize{ 8, 5, 4, 2 };

const UpsampleStage = struct {
    conv_tr: nn.ConvTranspose1d,
    resnet: nn.ResnetBlock,
};

pub const Decoder = struct {
    /// Owns every normalized weight buffer.
    arena: std.heap.ArenaAllocator,
    quantizer: rvq.Quantizer,
    /// decoder.layers.0
    conv_in: nn.Conv1d,
    /// decoder.layers.1
    lstm: nn.Lstm(2),
    /// decoder.layers.3 - decoder.layers.14 (ConvTranspose1d, ResnetBlock, ELU)
    stages: [ratios.len]UpsampleStage,
    /// decoder.layers.15
    conv_out: nn.Conv1d,

    /// Builds the decoder from `weights`, which must outlive it.
    pub fn load(child_allocator: Allocator, weights: *const SafeTensors) !Decoder {
        var arena: std.heap.ArenaAllocator = .init(child_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const quantizer = try rvq.Quantizer.load(weights);

        const conv_in = try seanet.loadConv1d(allocator, weights, 1, "decoder.layers.0.conv", .{});
        if (conv_in.c_in != quantizer.dim) return error.ShapeMismatch;

        const lstm = try seanet.loadLstm(weights, "decoder.layers.1.lstm");
        if (lstm.hiddenSize() != conv_in.c_out) return error.ShapeMismatch;

        var stages: [ratios.len]UpsampleStage = undefined;
        var channels = conv_in.c_out;
        for (&stages, ratios, 0..) |*stage, ratio, i| {
            const conv_tr_index = 3 + 3 * i;
            const resnet_index = conv_tr_index + 1;

            const conv_tr = try nn.ConvTranspose1d.initWeightNorm(
                allocator,
                try seanet.loadWeightNorm(weights, "decoder.layers.{d}.conv", .{conv_tr_index}),
                ratio,
            );
            if (conv_tr.c_in != channels) return error.ShapeMismatch;

            const resnet = try seanet.loadResnetBlock(allocator, weights, "decoder.layers.{d}", .{resnet_index});
            if (resnet.channels() != conv_tr.c_out) return error.ShapeMismatch;

            stage.* = .{ .conv_tr = conv_tr, .resnet = resnet };
            channels = resnet.channels();
        }

        const conv_out = try seanet.loadConv1d(allocator, weights, 1, "decoder.layers.15.conv", .{});
        if (conv_out.c_in != channels) return error.ShapeMismatch;

        return .{
            .arena = arena,
            .quantizer = quantizer,
            .conv_in = conv_in,
            .lstm = lstm,
            .stages = stages,
            .conv_out = conv_out,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.arena.deinit();
    }

    /// Decodes RVQ frames into mono PCM samples. Caller owns the returned slice.
    pub fn decode(self: *const Decoder, allocator: Allocator, frames: []const rvq.Frame) ![]f32 {
        var t_len = frames.len;

        // RVQ: [frames] -> [128, T]
        const latents = try allocator.alloc(f32, self.quantizer.dim * t_len);
        defer allocator.free(latents);
        try self.quantizer.decode(frames, latents);

        // Layer 0 (causal Conv1d): [128, T] -> [512, T]
        const conv_in_out = try allocator.alloc(f32, self.conv_in.c_out * t_len);
        defer allocator.free(conv_in_out);
        self.conv_in.forward(latents, t_len, conv_in_out);

        // Layer 1 (2-Layer LSTM + skip) and Layer 2 (ELU): [512, T] -> [512, T]
        var x = try allocator.alloc(f32, self.lstm.hiddenSize() * t_len);
        defer allocator.free(x);
        try self.lstm.forward(allocator, conv_in_out, t_len, x);
        nn.activation.eluInPlace(x);

        // Upsampling stages: ConvTranspose1d -> ResnetBlock -> ELU
        for (self.stages) |stage| {
            const up_len = stage.conv_tr.outputLen(t_len);

            const up = try allocator.alloc(f32, stage.conv_tr.c_out * up_len);
            defer allocator.free(up);
            stage.conv_tr.forward(x, t_len, up);

            const res = try allocator.alloc(f32, stage.resnet.channels() * up_len);
            errdefer allocator.free(res);
            try stage.resnet.forward(allocator, up, up_len, res);
            nn.activation.eluInPlace(res);

            allocator.free(x);
            x = res;
            t_len = up_len;
        }

        // Layer 15 (Final causal Conv1d): [32, T'] -> [1, T']
        const audio = try allocator.alloc(f32, self.conv_out.c_out * t_len);
        self.conv_out.forward(x, t_len, audio);
        return audio;
    }
};
