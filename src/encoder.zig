//! EnCodec 24 kHz encoder: PCM -> causal SEANet encoder -> latents -> RVQ tokens.
//! Follows encodec.modules.seanet.SEANetEncoder with causal=True, pad_mode='reflect'.

const std = @import("std");
const Allocator = std.mem.Allocator;
const nn = @import("nn.zig");
const rvq = @import("rvq.zig");
const seanet = @import("seanet.zig");
const SafeTensors = @import("model.zig").SafeTensors;

pub const sample_rate = 24000;

/// Downsampling ratio of each encoder stage (the decoder ratios reversed).
const ratios = [_]usize{ 2, 4, 5, 8 };

const DownsampleStage = struct {
    resnet: nn.ResnetBlock,
    conv: nn.Conv1d,
};

pub const Encoder = struct {
    /// Owns every normalized weight buffer.
    arena: std.heap.ArenaAllocator,
    /// encoder.layers.0
    conv_in: nn.Conv1d,
    /// encoder.layers.1 - encoder.layers.12 (ResnetBlock, ELU, strided Conv1d)
    stages: [ratios.len]DownsampleStage,
    /// encoder.layers.13
    lstm: nn.Lstm(2),
    /// encoder.layers.15
    conv_out: nn.Conv1d,
    quantizer: rvq.Quantizer,

    /// Builds the encoder from `weights`, which must outlive it.
    pub fn load(child_allocator: Allocator, weights: *const SafeTensors) !Encoder {
        var arena: std.heap.ArenaAllocator = .init(child_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const conv_in = try seanet.loadConv1d(allocator, weights, 1, "encoder.layers.0.conv", .{});
        if (conv_in.c_in != 1) return error.ShapeMismatch;

        var stages: [ratios.len]DownsampleStage = undefined;
        var channels = conv_in.c_out;
        for (&stages, ratios, 0..) |*stage, ratio, i| {
            const resnet_index = 1 + 3 * i;
            const conv_index = resnet_index + 2;

            const resnet = try seanet.loadResnetBlock(allocator, weights, "encoder.layers.{d}", .{resnet_index});
            if (resnet.channels() != channels) return error.ShapeMismatch;

            const conv = try seanet.loadConv1d(allocator, weights, ratio, "encoder.layers.{d}.conv", .{conv_index});
            if (conv.c_in != channels or conv.kernel_size != 2 * ratio) return error.ShapeMismatch;

            stage.* = .{ .resnet = resnet, .conv = conv };
            channels = conv.c_out;
        }

        const lstm = try seanet.loadLstm(weights, "encoder.layers.13.lstm");
        if (lstm.hiddenSize() != channels) return error.ShapeMismatch;

        const conv_out = try seanet.loadConv1d(allocator, weights, 1, "encoder.layers.15.conv", .{});
        if (conv_out.c_in != channels) return error.ShapeMismatch;

        const quantizer = try rvq.Quantizer.load(weights);
        if (quantizer.dim != conv_out.c_out) return error.ShapeMismatch;

        return .{
            .arena = arena,
            .conv_in = conv_in,
            .stages = stages,
            .lstm = lstm,
            .conv_out = conv_out,
            .quantizer = quantizer,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.arena.deinit();
    }

    /// Number of latent frames for `num_samples` input samples (ceil at every downsampling).
    pub fn numFrames(self: *const Encoder, num_samples: usize) usize {
        var t_len = num_samples;
        for (self.stages) |stage| t_len = stage.conv.outputLen(t_len);
        return t_len;
    }

    /// Runs the SEANet encoder on mono PCM. Returns latents [128, numFrames(samples.len)];
    /// caller owns the returned slice.
    pub fn encodeLatents(self: *const Encoder, allocator: Allocator, samples: []const f32) ![]f32 {
        if (samples.len == 0) return error.EmptyInput;
        var t_len = samples.len;

        // Layer 0 (causal Conv1d): [1, T] -> [32, T]
        var x = try allocator.alloc(f32, self.conv_in.c_out * t_len);
        defer allocator.free(x);
        try self.conv_in.forward(allocator, samples, t_len, x);

        // Downsampling stages: ResnetBlock -> ELU -> strided causal Conv1d
        for (self.stages) |stage| {
            const res = try allocator.alloc(f32, stage.resnet.channels() * t_len);
            defer allocator.free(res);
            try stage.resnet.forward(allocator, x, t_len, res);
            nn.activation.eluInPlace(res);

            const down_len = stage.conv.outputLen(t_len);
            const down = try allocator.alloc(f32, stage.conv.c_out * down_len);
            errdefer allocator.free(down);
            try stage.conv.forward(allocator, res, t_len, down);

            allocator.free(x);
            x = down;
            t_len = down_len;
        }

        // Layer 13 (2-Layer LSTM + skip) and Layer 14 (ELU): [512, F] -> [512, F]
        const y = try allocator.alloc(f32, self.lstm.hiddenSize() * t_len);
        defer allocator.free(y);
        try self.lstm.forward(allocator, x, t_len, y);
        nn.activation.eluInPlace(y);

        // Layer 15 (Final causal Conv1d): [512, F] -> [128, F]
        const latents = try allocator.alloc(f32, self.conv_out.c_out * t_len);
        errdefer allocator.free(latents);
        try self.conv_out.forward(allocator, y, t_len, latents);
        return latents;
    }

    /// Encodes mono PCM into RVQ frames. Caller owns the returned slice.
    pub fn encode(self: *const Encoder, allocator: Allocator, samples: []const f32) ![]rvq.Frame {
        const latents = try self.encodeLatents(allocator, samples);
        defer allocator.free(latents);

        const frames = try allocator.alloc(rvq.Frame, self.numFrames(samples.len));
        errdefer allocator.free(frames);
        try self.quantizer.encode(allocator, latents, frames);
        return frames;
    }
};
