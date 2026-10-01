const std = @import("std");
const Model = @import("model.zig").Model;
const decoder = @import("decoder.zig");
const rvq = @import("rvq.zig");
const wav = @import("wav.zig");

const weights_path = "src/weights/encodec_24khz.safetensors";
const default_output_wav_path = "output.wav";
const max_token_file_size = 64 * 1024 * 1024;

// Sample RVQ tokens: 4 frames x 8 stages
const sequence_tokens = [_]rvq.Frame{
    .{ 120, 450, 89, 730, 210, 95, 600, 314 },
    .{ 121, 448, 88, 729, 212, 94, 599, 310 },
    .{ 125, 440, 92, 735, 205, 99, 605, 320 },
    .{ 130, 435, 95, 740, 200, 102, 610, 325 },
};

const usage =
    \\Usage: hoge [options]
    \\
    \\Options:
    \\  -t, --tokens <path>   Raw RVQ tokens: little-endian u16, [num_frames, 8]
    \\                        (default: built-in 4-frame sample)
    \\  -o, --output <path>   Output WAV path (default: output.wav)
    \\      --dump-header     Print the SafeTensors JSON header and exit
    \\  -h, --help            Show this help
    \\
;

const Options = struct {
    tokens_path: ?[]const u8 = null,
    output_path: []const u8 = default_output_wav_path,
    dump_header: bool = false,
    help: bool = false,
};

const ArgsError = error{ MissingValue, UnknownArgument };

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

    // 2. Load RVQ tokens
    const external_tokens = if (options.tokens_path) |path|
        loadTokens(io, allocator, path) catch |err| {
            std.log.err("failed to load tokens from {s}: {t}", .{ path, err });
            return err;
        }
    else
        null;
    defer if (external_tokens) |tokens| allocator.free(tokens);
    const tokens: []const rvq.Frame = external_tokens orelse &sequence_tokens;

    // 3. Map and parse weight file
    var model = try Model.open(io, allocator, weights_path);
    defer model.close();

    if (options.dump_header) {
        try stdout.writeAll(model.weights.header_json);
        try stdout.writeByte('\n');
        try stdout.flush();
        return;
    }

    // 4. Build decoder layers from weights
    var dec = try decoder.Decoder.load(allocator, &model.weights);
    defer dec.deinit();

    // 5. Decode tokens into PCM
    const audio = try dec.decode(allocator, tokens);
    defer allocator.free(audio);

    // 6. Inspect final audio waveform
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

    // 7. Export to WAV file
    try wav.writeMono16File(io, options.output_path, audio, decoder.sample_rate);
    try stdout.print("Successfully exported decoded audio to {s}\n", .{options.output_path});
    try stdout.flush();
}

test "parseArgs defaults" {
    var failed_arg: []const u8 = "";
    const options = try parseArgs(&.{}, &failed_arg);
    try std.testing.expectEqual(@as(?[]const u8, null), options.tokens_path);
    try std.testing.expectEqualStrings(default_output_wav_path, options.output_path);
    try std.testing.expect(!options.dump_header);
}

test "parseArgs accepts long and short forms" {
    var failed_arg: []const u8 = "";
    const long = try parseArgs(&.{ "--tokens", "a.bin", "--output", "a.wav" }, &failed_arg);
    try std.testing.expectEqualStrings("a.bin", long.tokens_path.?);
    try std.testing.expectEqualStrings("a.wav", long.output_path);

    const short = try parseArgs(&.{ "-o", "b.wav", "-t", "b.bin", "--dump-header" }, &failed_arg);
    try std.testing.expectEqualStrings("b.bin", short.tokens_path.?);
    try std.testing.expectEqualStrings("b.wav", short.output_path);
    try std.testing.expect(short.dump_header);
}

test "parseArgs rejects missing values and unknown arguments" {
    var failed_arg: []const u8 = "";
    try std.testing.expectError(error.MissingValue, parseArgs(&.{"--tokens"}, &failed_arg));
    try std.testing.expectEqualStrings("--tokens", failed_arg);
    try std.testing.expectError(error.UnknownArgument, parseArgs(&.{"--bogus"}, &failed_arg));
    try std.testing.expectEqualStrings("--bogus", failed_arg);
}

test {
    _ = @import("nn.zig");
    _ = @import("rvq.zig");
    _ = @import("model.zig");
    _ = @import("wav.zig");
    _ = decoder;
}
