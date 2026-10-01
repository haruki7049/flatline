const std = @import("std");
const Model = @import("model.zig").Model;
const SafeTensors = @import("model.zig").SafeTensors;
const decoder = @import("decoder.zig");
const encoder = @import("encoder.zig");
const rvq = @import("rvq.zig");
const wav = @import("wav.zig");

const weights_path = "src/weights/encodec_24khz.safetensors";
const default_output_wav_path = "output.wav";
const default_output_tokens_path = "tokens.bin";
const max_token_file_size = 64 * 1024 * 1024;

// Sample RVQ tokens: 4 frames x 8 stages
const sequence_tokens = [_]rvq.Frame{
    .{ 120, 450, 89, 730, 210, 95, 600, 314 },
    .{ 121, 448, 88, 729, 212, 94, 599, 310 },
    .{ 125, 440, 92, 735, 205, 99, 605, 320 },
    .{ 130, 435, 95, 740, 200, 102, 610, 325 },
};

const usage =
    \\Usage: hoge [options]                   Decode RVQ tokens into a WAV file
    \\       hoge --encode <input.wav> [-o <tokens.bin>]
    \\                                        Encode a 24 kHz WAV file into RVQ tokens
    \\
    \\Options:
    \\  -t, --tokens <path>   Raw RVQ tokens to decode: little-endian u16, [num_frames, 8]
    \\                        (default: built-in 4-frame sample)
    \\  -e, --encode <path>   WAV to encode (16-bit PCM or 32-bit float, mono or stereo, 24 kHz)
    \\  -o, --output <path>   Output path (default: output.wav, or tokens.bin with --encode)
    \\      --dump-header     Print the SafeTensors JSON header and exit
    \\  -h, --help            Show this help
    \\
;

const Options = struct {
    tokens_path: ?[]const u8 = null,
    encode_path: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    dump_header: bool = false,
    help: bool = false,
};

const ArgsError = error{ MissingValue, UnknownArgument, ConflictingArguments };

/// Parses command line arguments, excluding the program name.
fn parseArgs(args: []const [:0]const u8, failed_arg: *[]const u8) ArgsError!Options {
    var options: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        failed_arg.* = arg;
        if (eqlAny(arg, &.{ "-t", "--tokens" })) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.tokens_path = args[i];
        } else if (eqlAny(arg, &.{ "-e", "--encode" })) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.encode_path = args[i];
        } else if (eqlAny(arg, &.{ "-o", "--output" })) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.output_path = args[i];
        } else if (std.mem.eql(u8, arg, "--dump-header")) {
            options.dump_header = true;
        } else if (eqlAny(arg, &.{ "-h", "--help" })) {
            options.help = true;
        } else {
            return error.UnknownArgument;
        }
    }
    if (options.encode_path != null and options.tokens_path != null) {
        failed_arg.* = "--encode";
        return error.ConflictingArguments;
    }
    return options;
}

fn eqlAny(arg: []const u8, names: []const []const u8) bool {
    for (names) |name| {
        if (std.mem.eql(u8, arg, name)) return true;
    }
    return false;
}

fn loadTokens(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]rvq.Frame {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_token_file_size));
    defer allocator.free(bytes);
    return rvq.framesFromBytes(allocator, bytes);
}

fn writeTokens(io: std.Io, path: []const u8, frames: []const rvq.Frame) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try rvq.writeFrames(&writer.interface, frames);
    try writer.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    // 1. Parse command line
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var failed_arg: []const u8 = "";
    const options = parseArgs(argv[1..], &failed_arg) catch |err| {
        var stderr_buffer: [256]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
        const stderr = &stderr_writer.interface;
        switch (err) {
            error.MissingValue => try stderr.print("error: {s} requires a value\n\n", .{failed_arg}),
            error.UnknownArgument => try stderr.print("error: unknown argument: {s}\n\n", .{failed_arg}),
            error.ConflictingArguments => try stderr.print("error: --encode cannot be combined with --tokens\n\n", .{}),
        }
        try stderr.writeAll(usage);
        try stderr.flush();
        std.process.exit(2);
    };
    if (options.help) {
        try stdout.writeAll(usage);
        try stdout.flush();
        return;
    }

    // 2. Map and parse weight file
    var model = try Model.open(io, allocator, weights_path);
    defer model.close();

    if (options.dump_header) {
        try stdout.writeAll(model.weights.header_json);
        try stdout.writeByte('\n');
        try stdout.flush();
        return;
    }

    if (options.encode_path) |input_path| {
        try runEncode(io, allocator, stdout, &model.weights, input_path, options.output_path orelse default_output_tokens_path);
    } else {
        try runDecode(io, allocator, stdout, &model.weights, options.tokens_path, options.output_path orelse default_output_wav_path);
    }
}

fn runDecode(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    weights: *const SafeTensors,
    tokens_path: ?[]const u8,
    output_path: []const u8,
) !void {
    // 1. Load RVQ tokens
    const external_tokens = if (tokens_path) |path|
        loadTokens(io, allocator, path) catch |err| {
            std.log.err("failed to load tokens from {s}: {t}", .{ path, err });
            return err;
        }
    else
        null;
    defer if (external_tokens) |tokens| allocator.free(tokens);
    const tokens: []const rvq.Frame = external_tokens orelse &sequence_tokens;

    // 2. Build decoder layers from weights
    var dec = try decoder.Decoder.load(allocator, weights);
    defer dec.deinit();

    // 3. Decode tokens into PCM
    const audio = try dec.decode(allocator, tokens);
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
    try wav.writeMono16File(io, output_path, audio, decoder.sample_rate);
    try stdout.print("Successfully exported decoded audio to {s}\n", .{output_path});
    try stdout.flush();
}

fn runEncode(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    weights: *const SafeTensors,
    input_path: []const u8,
    output_path: []const u8,
) !void {
    // 1. Load input audio
    const audio = wav.readMonoFile(io, allocator, input_path) catch |err| {
        std.log.err("failed to read {s}: {t}", .{ input_path, err });
        return err;
    };
    defer audio.deinit(allocator);
    if (audio.sample_rate != encoder.sample_rate) {
        std.log.err("{s}: sample rate {d} Hz is not supported (expected {d} Hz)", .{ input_path, audio.sample_rate, encoder.sample_rate });
        return error.UnsupportedSampleRate;
    }

    // 2. Build encoder layers and quantizer from weights
    var enc = try encoder.Encoder.load(allocator, weights);
    defer enc.deinit();

    // 3. Encode PCM into RVQ tokens
    const frames = try enc.encode(allocator, audio.samples);
    defer allocator.free(frames);

    try stdout.print("Complete EnCodec Encoder Pipeline finished successfully!\n", .{});
    try stdout.print("Encoded {d} samples ({d:.4}s) into {d} frames x {d} stages\n", .{
        audio.samples.len,
        @as(f32, @floatFromInt(audio.samples.len)) / @as(f32, @floatFromInt(encoder.sample_rate)),
        frames.len,
        rvq.num_stages,
    });
    try stdout.print("First frame tokens: {any}\n", .{frames[0]});
    try stdout.flush();

    // 4. Export tokens
    try writeTokens(io, output_path, frames);
    try stdout.print("Successfully exported tokens to {s}\n", .{output_path});
    try stdout.flush();
}

test "parseArgs defaults" {
    var failed_arg: []const u8 = "";
    const options = try parseArgs(&.{}, &failed_arg);
    try std.testing.expectEqual(@as(?[]const u8, null), options.tokens_path);
    try std.testing.expectEqual(@as(?[]const u8, null), options.encode_path);
    try std.testing.expectEqual(@as(?[]const u8, null), options.output_path);
    try std.testing.expect(!options.dump_header);
}

test "parseArgs accepts long and short forms" {
    var failed_arg: []const u8 = "";
    const long = try parseArgs(&.{ "--tokens", "a.bin", "--output", "a.wav" }, &failed_arg);
    try std.testing.expectEqualStrings("a.bin", long.tokens_path.?);
    try std.testing.expectEqualStrings("a.wav", long.output_path.?);

    const short = try parseArgs(&.{ "-o", "b.wav", "-t", "b.bin", "--dump-header" }, &failed_arg);
    try std.testing.expectEqualStrings("b.bin", short.tokens_path.?);
    try std.testing.expectEqualStrings("b.wav", short.output_path.?);
    try std.testing.expect(short.dump_header);

    const encode = try parseArgs(&.{ "--encode", "in.wav", "-o", "out.bin" }, &failed_arg);
    try std.testing.expectEqualStrings("in.wav", encode.encode_path.?);
    try std.testing.expectEqualStrings("out.bin", encode.output_path.?);
}

test "parseArgs rejects missing values, unknown and conflicting arguments" {
    var failed_arg: []const u8 = "";
    try std.testing.expectError(error.MissingValue, parseArgs(&.{"--tokens"}, &failed_arg));
    try std.testing.expectEqualStrings("--tokens", failed_arg);
    try std.testing.expectError(error.MissingValue, parseArgs(&.{"--encode"}, &failed_arg));
    try std.testing.expectError(error.UnknownArgument, parseArgs(&.{"--bogus"}, &failed_arg));
    try std.testing.expectEqualStrings("--bogus", failed_arg);
    try std.testing.expectError(error.ConflictingArguments, parseArgs(&.{ "-e", "a.wav", "-t", "a.bin" }, &failed_arg));
}

test {
    _ = @import("nn.zig");
    _ = @import("rvq.zig");
    _ = @import("model.zig");
    _ = @import("wav.zig");
    _ = @import("seanet.zig");
    _ = decoder;
    _ = encoder;
}
