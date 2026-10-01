const std = @import("std");
const Model = @import("model.zig").Model;
const decoder = @import("decoder.zig");
const rvq = @import("rvq.zig");
const wav = @import("wav.zig");

const weights_path = "src/weights/encodec_24khz.safetensors";
const output_wav_path = "output.wav";

// Sample RVQ tokens: 4 frames x 8 stages
const sequence_tokens = [_]rvq.Frame{
    .{ 120, 450, 89, 730, 210, 95, 600, 314 },
    .{ 121, 448, 88, 729, 212, 94, 599, 310 },
    .{ 125, 440, 92, 735, 205, 99, 605, 320 },
    .{ 130, 435, 95, 740, 200, 102, 610, 325 },
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    // 1. Map and parse weight file
    var model = try Model.open(io, allocator, weights_path);
    defer model.close();

    var args = init.minimal.args.iterate();
    _ = args.next();
    if (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump-header")) {
            try stdout.writeAll(model.weights.header_json);
            try stdout.writeByte('\n');
            try stdout.flush();
            return;
        }
    }

    // 2. Build decoder layers from weights
    var dec = try decoder.Decoder.load(allocator, &model.weights);
    defer dec.deinit();

    // 3. Decode tokens into PCM
    const audio = try dec.decode(allocator, &sequence_tokens);
    defer allocator.free(audio);

    // 4. Inspect final audio waveform
    try stdout.print("Complete EnCodec Decoder Pipeline finished successfully!\n", .{});
    try stdout.print("Decoded PCM Samples Count: {d} (Sample Rate: {d}Hz, Duration: {d:.4}s)\n", .{
        audio.len,
        decoder.sample_rate,
        @as(f32, @floatFromInt(audio.len)) / @as(f32, @floatFromInt(decoder.sample_rate)),
    });
    try stdout.print("First 16 PCM waveform samples:\n", .{});
    for (audio[0..@min(16, audio.len)]) |sample| {
        try stdout.print("{d:.5} ", .{sample});
    }
    try stdout.writeByte('\n');
    try stdout.flush();

    // 5. Export to WAV file
    try wav.writeMono16File(io, output_wav_path, audio, decoder.sample_rate);
    try stdout.print("Successfully exported decoded audio to {s}\n", .{output_wav_path});
    try stdout.flush();
}

test {
    _ = @import("nn.zig");
    _ = @import("rvq.zig");
    _ = @import("model.zig");
    _ = @import("wav.zig");
    _ = decoder;
}
