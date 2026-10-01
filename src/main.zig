const std = @import("std");

// Writes float PCM [-1.0, 1.0] to a 16-bit PCM mono WAV file
pub fn writeWav16(
    io: std.Io,
    file_path: []const u8,
    samples: []const f32,
    sample_rate: u32,
) !void {
    const file = try std.Io.Dir.cwd().createFile(io, file_path, .{});
    defer file.close(io);

    var buffer: [1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const w = &writer.interface;

    const num_samples = @as(u32, @intCast(samples.len));
    const byte_rate = sample_rate * 2; // 1 channel * 2 bytes/sample
    const block_align: u16 = 2; // 1 channel * 2 bytes
    const bits_per_sample: u16 = 16;
    const data_bytes = num_samples * 2;
    const riff_chunk_size = 36 + data_bytes;

    // 1. RIFF Header
    try w.writeAll("RIFF");
    try w.writeInt(u32, riff_chunk_size, .little);
    try w.writeAll("WAVE");

    // 2. fmt Sub-chunk
    try w.writeAll("fmt ");
    try w.writeInt(u32, 16, .little); // Subchunk1Size (16 for PCM)
    try w.writeInt(u16, 1, .little); // AudioFormat (1 for PCM)
    try w.writeInt(u16, 1, .little); // NumChannels (1 = mono)
    try w.writeInt(u32, sample_rate, .little);
    try w.writeInt(u32, byte_rate, .little);
    try w.writeInt(u16, block_align, .little);
    try w.writeInt(u16, bits_per_sample, .little);

    // 3. data Sub-chunk
    try w.writeAll("data");
    try w.writeInt(u32, data_bytes, .little);

    // Convert f32 [-1.0, 1.0] to i16 with clamping
    for (samples) |sample| {
        const clamped = std.math.clamp(sample, -1.0, 1.0);
        const sample_i16 = @as(i16, @intFromFloat(clamped * 32767.0));
        try w.writeInt(i16, sample_i16, .little);
    }

    try w.flush();
}

// Decodes a sequence of RVQ frames into a contiguous latent matrix [128, T]
pub fn decodeRVQSequence(
    codebooks: []const [*]const f32,
    tokens: []const [8]u16,
    num_frames: usize,
    out_latents: []f32,
) void {
    for (0..128) |c| {
        for (0..num_frames) |t| {
            out_latents[c * num_frames + t] = 0.0;
        }
    }

    for (0..num_frames) |t| {
        const frame_tokens = tokens[t];
        for (frame_tokens, 0..) |token, stage| {
            const base_ptr = codebooks[stage] + (@as(usize, token) * 128);
            for (0..128) |c| {
                out_latents[c * num_frames + t] += base_ptr[c];
            }
        }
    }
}

// 1D Convolution with precomputed normalized weights
pub fn conv1dSame(
    in: []const f32,
    c_in: usize,
    t_len: usize,
    weight: []const f32,
    bias: []const f32,
    c_out: usize,
    kernel_size: usize,
    out: []f32,
) void {
    const pad = kernel_size / 2;

    for (0..c_out) |co| {
        const bias_val = bias[co];
        const w_co = weight[co * (c_in * kernel_size) .. (co + 1) * (c_in * kernel_size)];

        for (0..t_len) |t| {
            var sum: f32 = bias_val;

            for (0..c_in) |ci| {
                const in_channel = in[ci * t_len .. (ci + 1) * t_len];
                const w_channel = w_co[ci * kernel_size .. (ci + 1) * kernel_size];

                for (0..kernel_size) |k| {
                    const in_t = @as(isize, @intCast(t)) + @as(isize, @intCast(k)) - @as(isize, @intCast(pad));
                    if (in_t >= 0 and in_t < @as(isize, @intCast(t_len))) {
                        sum += in_channel[@as(usize, @intCast(in_t))] * w_channel[k];
                    }
                }
            }

            out[co * t_len + t] = sum;
        }
    }
}

fn sigmoid(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return 0.0;
    return 1.0 / (1.0 + @exp(-x));
}

fn tanhActivation(x: f32) f32 {
    if (x >= 20.0) return 1.0;
    if (x <= -20.0) return -1.0;
    return std.math.tanh(x);
}

// ELU activation: x if x > 0 else (exp(x) - 1)
fn elu(x: f32) f32 {
    if (x > 0.0) return x;
    return @exp(x) - 1.0;
}

// Unidirectional LSTM across time dimension
pub fn runLstmLayer(
    in: []const f32,
    out: []f32,
    h_dim: usize,
    t_len: usize,
    w_ih: [*]const f32,
    w_hh: [*]const f32,
    b_ih: [*]const f32,
    b_hh: [*]const f32,
    h_state: []f32,
    c_state: []f32,
) void {
    @memset(h_state, 0.0);
    @memset(c_state, 0.0);

    for (0..t_len) |t| {
        for (0..h_dim) |d| {
            var pre_i = b_ih[d] + b_hh[d];
            var pre_f = b_ih[h_dim + d] + b_hh[h_dim + d];
            var pre_g = b_ih[2 * h_dim + d] + b_hh[2 * h_dim + d];
            var pre_o = b_ih[3 * h_dim + d] + b_hh[3 * h_dim + d];

            const row_ih_i = w_ih + d * h_dim;
            const row_ih_f = w_ih + (h_dim + d) * h_dim;
            const row_ih_g = w_ih + (2 * h_dim + d) * h_dim;
            const row_ih_o = w_ih + (3 * h_dim + d) * h_dim;

            const row_hh_i = w_hh + d * h_dim;
            const row_hh_f = w_hh + (h_dim + d) * h_dim;
            const row_hh_g = w_hh + (2 * h_dim + d) * h_dim;
            const row_hh_o = w_hh + (3 * h_dim + d) * h_dim;

            for (0..h_dim) |k| {
                const in_val = in[k * t_len + t];
                const prev_h = h_state[k];

                pre_i += row_ih_i[k] * in_val + row_hh_i[k] * prev_h;
                pre_f += row_ih_f[k] * in_val + row_hh_f[k] * prev_h;
                pre_g += row_ih_g[k] * in_val + row_hh_g[k] * prev_h;
                pre_o += row_ih_o[k] * in_val + row_hh_o[k] * prev_h;
            }

            const gate_i = sigmoid(pre_i);
            const gate_f = sigmoid(pre_f);
            const gate_g = tanhActivation(pre_g);
            const gate_o = sigmoid(pre_o);

            const next_c = gate_f * c_state[d] + gate_i * gate_g;
            const next_h = gate_o * tanhActivation(next_c);

            out[d * t_len + t] = next_h;
            c_state[d] = next_c;
        }

        for (0..h_dim) |d| {
            h_state[d] = out[d * t_len + t];
        }
    }
}

// 1D Transposed Convolution (Upsampling)
pub fn convTranspose1d(
    in: []const f32,
    c_in: usize,
    t_in: usize,
    weight: []const f32,
    bias: []const f32,
    c_out: usize,
    kernel_size: usize,
    stride: usize,
    out: []f32,
) void {
    const t_out = t_in * stride;
    const pad = (kernel_size - stride) / 2;

    for (0..c_out) |co| {
        const bias_val = bias[co];
        for (0..t_out) |to| {
            out[co * t_out + to] = bias_val;
        }
    }

    for (0..c_in) |ci| {
        for (0..c_out) |co| {
            const w_base = ci * (c_out * kernel_size) + co * kernel_size;
            const w_slice = weight[w_base .. w_base + kernel_size];

            for (0..t_in) |ti| {
                const in_val = in[ci * t_in + ti];
                if (in_val == 0.0) continue;

                for (0..kernel_size) |k| {
                    const out_time = @as(isize, @intCast(ti * stride + k)) - @as(isize, @intCast(pad));
                    if (out_time >= 0 and out_time < @as(isize, @intCast(t_out))) {
                        out[co * t_out + @as(usize, @intCast(out_time))] += in_val * w_slice[k];
                    }
                }
            }
        }
    }
}

// Normalizes weights: W = g * (v / ||v||) for Conv1d
fn normalizeConv1dWeights(
    allocator: std.mem.Allocator,
    weight_v: [*]const f32,
    weight_g: [*]const f32,
    c_out: usize,
    c_in: usize,
    kernel_size: usize,
) ![]f32 {
    const total_len = c_out * c_in * kernel_size;
    const norm = try allocator.alloc(f32, total_len);
    const slice_len = c_in * kernel_size;

    for (0..c_out) |co| {
        var norm_sq: f32 = 0.0;
        const base = co * slice_len;

        for (0..slice_len) |idx| {
            const val = weight_v[base + idx];
            norm_sq += val * val;
        }

        const scale = weight_g[co] / @sqrt(norm_sq);
        for (0..slice_len) |idx| {
            norm[base + idx] = weight_v[base + idx] * scale;
        }
    }

    return norm;
}

// Normalizes weights: W = g * (v / ||v||) for ConvTranspose1d
fn normalizeConvTranspose1dWeights(
    allocator: std.mem.Allocator,
    weight_v: [*]const f32,
    weight_g: [*]const f32,
    c_in: usize,
    c_out: usize,
    kernel_size: usize,
) ![]f32 {
    const total_len = c_in * c_out * kernel_size;
    const norm = try allocator.alloc(f32, total_len);
    const slice_len = c_out * kernel_size;

    for (0..c_in) |ci| {
        var norm_sq: f32 = 0.0;
        const base = ci * slice_len;

        for (0..slice_len) |idx| {
            const val = weight_v[base + idx];
            norm_sq += val * val;
        }

        const scale = weight_g[ci] / @sqrt(norm_sq);
        for (0..slice_len) |idx| {
            norm[base + idx] = weight_v[base + idx] * scale;
        }
    }

    return norm;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const file_path = "src/weights/encodec_24khz.safetensors";

    // 1. Open weight file
    const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    const file_size = (try file.stat(io)).size;
    if (file_size < 8) return error.InvalidFile;

    // 2. Map file into memory space (zero-copy)
    const mapped = try std.posix.mmap(
        null,
        file_size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mapped);

    // 3. Slice out header JSON
    const header_len = std.mem.readInt(u64, mapped[0..8], .little);
    if (header_len > mapped.len - 8) return error.InvalidFile;
    const header_json = mapped[8 .. 8 + header_len];

    var args = init.minimal.args;
    var args_iterator = args.iterate();
    _ = args_iterator.next();
    if (args_iterator.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump-header")) {
            var buffer: [1024]u8 = undefined;
            var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
            const stdout = &stdout_writer.interface;
            try stdout.writeAll(header_json);
            try stdout.writeByte('\n');
            try stdout.flush();
            return;
        }
    }

    const payload_start = 8 + header_len;

    // 4. Parse JSON metadata
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        header_json,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value.object;

    // Get RVQ codebooks
    const NUM_STAGES = 8;
    var codebooks: [NUM_STAGES][*]const f32 = undefined;
    for (0..NUM_STAGES) |stage| {
        var key_buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "quantizer.layers.{d}.codebook.embed", .{stage});
        const tensor_entry = root.get(key) orelse return error.TensorNotFound;
        const offsets = tensor_entry.object.get("data_offsets") orelse return error.InvalidFormat;
        const start_offset = @as(usize, @intCast(offsets.array.items[0].integer));
        codebooks[stage] = @ptrCast(@alignCast(mapped[payload_start + start_offset ..].ptr));
    }

    const get_offset = struct {
        fn run(r: anytype, k: []const u8) !usize {
            const entry = r.get(k) orelse return error.TensorNotFound;
            const offsets = entry.object.get("data_offsets") orelse return error.InvalidFormat;
            return @as(usize, @intCast(offsets.array.items[0].integer));
        }
    }.run;

    // Layer 0 weights (Conv1d)
    const conv0_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.0.conv.bias") ..].ptr));
    const conv0_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.0.conv.weight_v") ..].ptr));
    const conv0_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.0.conv.weight_g") ..].ptr));

    // Layer 1 weights (2-Layer LSTM)
    const l0_w_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_ih_l0") ..].ptr)));
    const l0_w_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_hh_l0") ..].ptr)));
    const l0_b_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_ih_l0") ..].ptr)));
    const l0_b_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_hh_l0") ..].ptr)));

    const l1_w_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_ih_l1") ..].ptr)));
    const l1_w_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.weight_hh_l1") ..].ptr)));
    const l1_b_ih = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_ih_l1") ..].ptr)));
    const l1_b_hh = @as([*]const f32, @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.1.lstm.bias_hh_l1") ..].ptr)));

    // Layer 3 weights (ConvTranspose1d)
    const conv3_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.3.conv.bias") ..].ptr));
    const conv3_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.3.conv.weight_v") ..].ptr));
    const conv3_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.3.conv.weight_g") ..].ptr));

    // Layer 4 weights (ResNet block)
    const l4_b1_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.1.conv.bias") ..].ptr));
    const l4_b1_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.1.conv.weight_v") ..].ptr));
    const l4_b1_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.1.conv.weight_g") ..].ptr));

    const l4_b3_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.3.conv.bias") ..].ptr));
    const l4_b3_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.3.conv.weight_v") ..].ptr));
    const l4_b3_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.block.3.conv.weight_g") ..].ptr));

    const l4_sc_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.shortcut.conv.bias") ..].ptr));
    const l4_sc_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.shortcut.conv.weight_v") ..].ptr));
    const l4_sc_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.4.shortcut.conv.weight_g") ..].ptr));

    // Layer 6 weights (ConvTranspose1d)
    const conv6_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.6.conv.bias") ..].ptr));
    const conv6_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.6.conv.weight_v") ..].ptr));
    const conv6_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.6.conv.weight_g") ..].ptr));

    // Layer 7 weights (ResNet block)
    const l7_b1_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.1.conv.bias") ..].ptr));
    const l7_b1_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.1.conv.weight_v") ..].ptr));
    const l7_b1_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.1.conv.weight_g") ..].ptr));

    const l7_b3_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.3.conv.bias") ..].ptr));
    const l7_b3_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.3.conv.weight_v") ..].ptr));
    const l7_b3_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.block.3.conv.weight_g") ..].ptr));

    const l7_sc_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.shortcut.conv.bias") ..].ptr));
    const l7_sc_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.shortcut.conv.weight_v") ..].ptr));
    const l7_sc_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.7.shortcut.conv.weight_g") ..].ptr));

    // Layer 9 weights (ConvTranspose1d)
    const conv9_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.9.conv.bias") ..].ptr));
    const conv9_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.9.conv.weight_v") ..].ptr));
    const conv9_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.9.conv.weight_g") ..].ptr));

    // Layer 10 weights (ResNet block)
    const l10_b1_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.1.conv.bias") ..].ptr));
    const l10_b1_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.1.conv.weight_v") ..].ptr));
    const l10_b1_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.1.conv.weight_g") ..].ptr));

    const l10_b3_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.3.conv.bias") ..].ptr));
    const l10_b3_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.3.conv.weight_v") ..].ptr));
    const l10_b3_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.block.3.conv.weight_g") ..].ptr));

    const l10_sc_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.shortcut.conv.bias") ..].ptr));
    const l10_sc_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.shortcut.conv.weight_v") ..].ptr));
    const l10_sc_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.10.shortcut.conv.weight_g") ..].ptr));

    // Layer 12 weights (ConvTranspose1d)
    const conv12_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.12.conv.bias") ..].ptr));
    const conv12_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.12.conv.weight_v") ..].ptr));
    const conv12_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.12.conv.weight_g") ..].ptr));

    // Layer 13 weights (ResNet block)
    const l13_b1_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.1.conv.bias") ..].ptr));
    const l13_b1_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.1.conv.weight_v") ..].ptr));
    const l13_b1_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.1.conv.weight_g") ..].ptr));

    const l13_b3_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.3.conv.bias") ..].ptr));
    const l13_b3_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.3.conv.weight_v") ..].ptr));
    const l13_b3_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.block.3.conv.weight_g") ..].ptr));

    const l13_sc_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.shortcut.conv.bias") ..].ptr));
    const l13_sc_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.shortcut.conv.weight_v") ..].ptr));
    const l13_sc_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.13.shortcut.conv.weight_g") ..].ptr));

    // Layer 15 weights (Final Conv1d)
    const conv15_bias: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.15.conv.bias") ..].ptr));
    const conv15_weight_v: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.15.conv.weight_v") ..].ptr));
    const conv15_weight_g: [*]const f32 = @ptrCast(@alignCast(mapped[payload_start + try get_offset(root, "decoder.layers.15.conv.weight_g") ..].ptr));

    // Normalize weights
    const conv0_weight_norm = try normalizeConv1dWeights(allocator, conv0_weight_v, conv0_weight_g, 512, 128, 7);
    const conv3_weight_norm = try normalizeConvTranspose1dWeights(allocator, conv3_weight_v, conv3_weight_g, 512, 256, 16);
    const l4_b1_norm = try normalizeConv1dWeights(allocator, l4_b1_v, l4_b1_g, 128, 256, 3);
    const l4_b3_norm = try normalizeConv1dWeights(allocator, l4_b3_v, l4_b3_g, 256, 128, 1);
    const l4_sc_norm = try normalizeConv1dWeights(allocator, l4_sc_v, l4_sc_g, 256, 256, 1);

    const conv6_weight_norm = try normalizeConvTranspose1dWeights(allocator, conv6_weight_v, conv6_weight_g, 256, 128, 10);
    const l7_b1_norm = try normalizeConv1dWeights(allocator, l7_b1_v, l7_b1_g, 64, 128, 3);
    const l7_b3_norm = try normalizeConv1dWeights(allocator, l7_b3_v, l7_b3_g, 128, 64, 1);
    const l7_sc_norm = try normalizeConv1dWeights(allocator, l7_sc_v, l7_sc_g, 128, 128, 1);

    const conv9_weight_norm = try normalizeConvTranspose1dWeights(allocator, conv9_weight_v, conv9_weight_g, 128, 64, 8);
    const l10_b1_norm = try normalizeConv1dWeights(allocator, l10_b1_v, l10_b1_g, 32, 64, 3);
    const l10_b3_norm = try normalizeConv1dWeights(allocator, l10_b3_v, l10_b3_g, 64, 32, 1);
    const l10_sc_norm = try normalizeConv1dWeights(allocator, l10_sc_v, l10_sc_g, 64, 64, 1);

    const conv12_weight_norm = try normalizeConvTranspose1dWeights(allocator, conv12_weight_v, conv12_weight_g, 64, 32, 4);
    const l13_b1_norm = try normalizeConv1dWeights(allocator, l13_b1_v, l13_b1_g, 16, 32, 3);
    const l13_b3_norm = try normalizeConv1dWeights(allocator, l13_b3_v, l13_b3_g, 32, 16, 1);
    const l13_sc_norm = try normalizeConv1dWeights(allocator, l13_sc_v, l13_sc_g, 32, 32, 1);

    const conv15_weight_norm = try normalizeConv1dWeights(allocator, conv15_weight_v, conv15_weight_g, 1, 32, 7);

    // 5. Decode tokens into latents [128, 4]
    var sequence_tokens = [_][8]u16{
        [_]u16{ 120, 450, 89, 730, 210, 95, 600, 314 },
        [_]u16{ 121, 448, 88, 729, 212, 94, 599, 310 },
        [_]u16{ 125, 440, 92, 735, 205, 99, 605, 320 },
        [_]u16{ 130, 435, 95, 740, 200, 102, 610, 325 },
    };
    const num_frames = sequence_tokens.len;

    var latents: [128 * 4]f32 = undefined;
    decodeRVQSequence(&codebooks, &sequence_tokens, num_frames, &latents);

    // 6. Layer 0 (Conv1d): [128, 4] -> [512, 4]
    const conv_out = try allocator.alloc(f32, 512 * num_frames);
    conv1dSame(&latents, 128, num_frames, conv0_weight_norm, conv0_bias[0..512], 512, 7, conv_out);

    // 7. Layer 1 (2-Layer LSTM): [512, 4] -> [512, 4]
    const lstm_l0_out = try allocator.alloc(f32, 512 * num_frames);
    const lstm_l1_out = try allocator.alloc(f32, 512 * num_frames);
    const h_buf = try allocator.alloc(f32, 512);
    const c_buf = try allocator.alloc(f32, 512);

    runLstmLayer(conv_out, lstm_l0_out, 512, num_frames, l0_w_ih, l0_w_hh, l0_b_ih, l0_b_hh, h_buf, c_buf);
    @memset(h_buf, 0.0);
    @memset(c_buf, 0.0);
    runLstmLayer(lstm_l0_out, lstm_l1_out, 512, num_frames, l1_w_ih, l1_w_hh, l1_b_ih, l1_b_hh, h_buf, c_buf);

    // 8. Layer 2 (ELU)
    for (lstm_l1_out) |*val| {
        val.* = elu(val.*);
    }

    // 9. Layer 3 (ConvTranspose1d): [512, 4] -> [256, 32] (Stride 8)
    const t_up1 = num_frames * 8;
    const conv3_out = try allocator.alloc(f32, 256 * t_up1);
    convTranspose1d(lstm_l1_out, 512, num_frames, conv3_weight_norm, conv3_bias[0..256], 256, 16, 8, conv3_out);

    // 10. Layer 4 (ResNet Block): [256, 32] -> [256, 32]
    const l4_mid = try allocator.alloc(f32, 128 * t_up1);
    const l4_res = try allocator.alloc(f32, 256 * t_up1);
    const l4_sc = try allocator.alloc(f32, 256 * t_up1);
    const l4_out = try allocator.alloc(f32, 256 * t_up1);

    conv1dSame(conv3_out, 256, t_up1, l4_sc_norm, l4_sc_bias[0..256], 256, 1, l4_sc);
    conv1dSame(conv3_out, 256, t_up1, l4_b1_norm, l4_b1_bias[0..128], 128, 3, l4_mid);
    for (l4_mid) |*val| {
        val.* = elu(val.*);
    }
    conv1dSame(l4_mid, 128, t_up1, l4_b3_norm, l4_b3_bias[0..256], 256, 1, l4_res);

    for (0..256 * t_up1) |idx| {
        l4_out[idx] = l4_sc[idx] + l4_res[idx];
    }

    // 11. Layer 5 (ELU)
    for (l4_out) |*val| {
        val.* = elu(val.*);
    }

    // 12. Layer 6 (ConvTranspose1d): [256, 32] -> [128, 160] (Stride 5)
    const t_up2 = t_up1 * 5;
    const conv6_out = try allocator.alloc(f32, 128 * t_up2);
    convTranspose1d(l4_out, 256, t_up1, conv6_weight_norm, conv6_bias[0..128], 128, 10, 5, conv6_out);

    // 13. Layer 7 (ResNet Block): [128, 160] -> [128, 160]
    const l7_mid = try allocator.alloc(f32, 64 * t_up2);
    const l7_res = try allocator.alloc(f32, 128 * t_up2);
    const l7_sc = try allocator.alloc(f32, 128 * t_up2);
    const l7_out = try allocator.alloc(f32, 128 * t_up2);

    conv1dSame(conv6_out, 128, t_up2, l7_sc_norm, l7_sc_bias[0..128], 128, 1, l7_sc);
    conv1dSame(conv6_out, 128, t_up2, l7_b1_norm, l7_b1_bias[0..64], 64, 3, l7_mid);
    for (l7_mid) |*val| {
        val.* = elu(val.*);
    }
    conv1dSame(l7_mid, 64, t_up2, l7_b3_norm, l7_b3_bias[0..128], 128, 1, l7_res);

    for (0..128 * t_up2) |idx| {
        l7_out[idx] = l7_sc[idx] + l7_res[idx];
    }

    // 14. Layer 8 (ELU)
    for (l7_out) |*val| {
        val.* = elu(val.*);
    }

    // 15. Layer 9 (ConvTranspose1d): [128, 160] -> [64, 640] (Stride 4)
    const t_up3 = t_up2 * 4;
    const conv9_out = try allocator.alloc(f32, 64 * t_up3);
    convTranspose1d(l7_out, 128, t_up2, conv9_weight_norm, conv9_bias[0..64], 64, 8, 4, conv9_out);

    // 16. Layer 10 (ResNet Block): [64, 640] -> [64, 640]
    const l10_mid = try allocator.alloc(f32, 32 * t_up3);
    const l10_res = try allocator.alloc(f32, 64 * t_up3);
    const l10_sc = try allocator.alloc(f32, 64 * t_up3);
    const l10_out = try allocator.alloc(f32, 64 * t_up3);

    conv1dSame(conv9_out, 64, t_up3, l10_sc_norm, l10_sc_bias[0..64], 64, 1, l10_sc);
    conv1dSame(conv9_out, 64, t_up3, l10_b1_norm, l10_b1_bias[0..32], 32, 3, l10_mid);
    for (l10_mid) |*val| {
        val.* = elu(val.*);
    }
    conv1dSame(l10_mid, 32, t_up3, l10_b3_norm, l10_b3_bias[0..64], 64, 1, l10_res);

    for (0..64 * t_up3) |idx| {
        l10_out[idx] = l10_sc[idx] + l10_res[idx];
    }

    // 17. Layer 11 (ELU)
    for (l10_out) |*val| {
        val.* = elu(val.*);
    }

    // 18. Layer 12 (ConvTranspose1d): [64, 640] -> [32, 1280] (Stride 2)
    const t_up4 = t_up3 * 2;
    const conv12_out = try allocator.alloc(f32, 32 * t_up4);
    convTranspose1d(l10_out, 64, t_up3, conv12_weight_norm, conv12_bias[0..32], 32, 4, 2, conv12_out);

    // 19. Layer 13 (ResNet Block): [32, 1280] -> [32, 1280]
    const l13_mid = try allocator.alloc(f32, 16 * t_up4);
    const l13_res = try allocator.alloc(f32, 32 * t_up4);
    const l13_sc = try allocator.alloc(f32, 32 * t_up4);
    const l13_out = try allocator.alloc(f32, 32 * t_up4);

    conv1dSame(conv12_out, 32, t_up4, l13_sc_norm, l13_sc_bias[0..32], 32, 1, l13_sc);
    conv1dSame(conv12_out, 32, t_up4, l13_b1_norm, l13_b1_bias[0..16], 16, 3, l13_mid);
    for (l13_mid) |*val| {
        val.* = elu(val.*);
    }
    conv1dSame(l13_mid, 16, t_up4, l13_b3_norm, l13_b3_bias[0..32], 32, 1, l13_res);

    for (0..32 * t_up4) |idx| {
        l13_out[idx] = l13_sc[idx] + l13_res[idx];
    }

    // 20. Layer 14 (ELU)
    for (l13_out) |*val| {
        val.* = elu(val.*);
    }

    // 21. Layer 15 (Final Conv1d): [32, 1280] -> [1, 1280]
    const final_audio = try allocator.alloc(f32, 1 * t_up4);
    conv1dSame(l13_out, 32, t_up4, conv15_weight_norm, conv15_bias[0..1], 1, 7, final_audio);

    // 22. Inspect final audio waveform
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Complete EnCodec Decoder Pipeline finished successfully!\n", .{});
    try stdout.print("Decoded PCM Samples Count: {d} (Sample Rate: 24000Hz, Duration: {d:.4}s)\n", .{
        t_up4,
        @as(f32, @floatFromInt(t_up4)) / 24000.0,
    });
    try stdout.print("First 16 PCM waveform samples:\n", .{});
    for (0..16) |i| {
        try stdout.print("{d:.5} ", .{final_audio[i]});
    }
    try stdout.writeByte('\n');
    try stdout.flush();

    // 23. Export to WAV file
    const output_wav_path = "output.wav";
    try writeWav16(io, output_wav_path, final_audio, 24000);
    try stdout.print("Successfully exported decoded audio to {s}\n", .{output_wav_path});
}
