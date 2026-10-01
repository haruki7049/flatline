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

const usage =
    \\Usage: hoge [options]                    Decode RVQ tokens into a WAV file
    \\       hoge --stream                     Stream tokens from stdin to stdout (raw f32 PCM)
    \\       hoge --encode <input.wav> [-o <tokens.bin>]
    \\
    \\Options:
    \\  -t, --tokens <path>    Raw RVQ tokens to decode: little-endian u16, [num_frames, 8]
    \\  -s, --stream           Read tokens from stdin and write raw f32 PCM to stdout
    \\  -e, --encode <path>    WAV to encode (24 kHz)
    \\  -o, --output <path>    Output path
    \\      --dump-header      Print the SafeTensors JSON header and exit
    \\  -h, --help             Show this help
    \\
;

const Options = struct {
    tokens_path: ?[]const u8 = null,
    encode_path: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    stream: bool = false,
    dump_header: bool = false,
    help: bool = false,
};

const ArgsError = error{ MissingValue, UnknownArgument, ConflictingArguments };

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
        } else if (eqlAny(arg, &.{ "-s", "--stream" })) {
            options.stream = true;
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

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var failed_arg: []const u8 = "";
    const options = parseArgs(argv[1..], &failed_arg) catch {
        var stderr_buffer: [256]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
        const stderr = &stderr_writer.interface;
        try stderr.print("error: {s}\n\n{s}", .{ failed_arg, usage });
        try stderr.flush();
        std.process.exit(2);
    };

    if (options.help) {
        var stdout_buffer: [1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
        try stdout_writer.interface.writeAll(usage);
        try stdout_writer.interface.flush();
        return;
    }

    var model = try Model.open(io, allocator, weights_path);
    defer model.close();

    if (options.stream) {
        try runStreamDecode(io, allocator, &model.weights);
    } else if (options.encode_path) |input_path| {
        var buffer: [1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
        try runEncode(io, allocator, &stdout_writer.interface, &model.weights, input_path, options.output_path orelse default_output_tokens_path);
    } else {
        var buffer: [1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
        try runDecode(io, allocator, &stdout_writer.interface, &model.weights, options.tokens_path, options.output_path orelse default_output_wav_path);
    }
}

/// Streams raw RVQ frames from stdin and writes raw f32-le PCM audio to stdout.
fn runStreamDecode(
    io: std.Io,
    allocator: std.mem.Allocator,
    weights: *const SafeTensors,
) !void {
    var dec = try decoder.Decoder.load(allocator, weights);
    defer dec.deinit();

    var stdin_buffer: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buffer);
    const stdin = &stdin_reader.interface;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const chunk_frames = 16; // 16 frames = ~213ms
    var chunk_buf: [chunk_frames]rvq.Frame = undefined;

    while (true) {
        const bytes_to_read = chunk_frames * @sizeOf(rvq.Frame);
        const chunk_slice_u8 = std.mem.sliceAsBytes(chunk_buf[0..chunk_frames]);

        var read_count: usize = 0;
        while (read_count < bytes_to_read) {
            const n = stdin.readSliceShort(chunk_slice_u8[read_count..]) catch |err| {
                return err;
            };
            if (n == 0) break;
            read_count += n;
        }

        if (read_count == 0) break; // EOF
        const frames_read = read_count / @sizeOf(rvq.Frame);
        if (frames_read == 0) break;

        // Decode only the incoming chunk (no overlapping context re-computation)
        const pcm_chunk = try dec.decode(allocator, chunk_buf[0..frames_read]);
        defer allocator.free(pcm_chunk);

        try stdout.writeAll(std.mem.sliceAsBytes(pcm_chunk));
        try stdout.flush();
    }
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

fn runDecode(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    weights: *const SafeTensors,
    tokens_path: ?[]const u8,
    output_path: []const u8,
) !void {
    const external_tokens = if (tokens_path) |path|
        loadTokens(io, allocator, path) catch |err| {
            std.log.err("failed to load tokens from {s}: {t}", .{ path, err });
            return err;
        }
    else
        null;
    defer if (external_tokens) |tokens| allocator.free(tokens);
    const tokens: []const rvq.Frame = external_tokens orelse &[_]rvq.Frame{};

    var dec = try decoder.Decoder.load(allocator, weights);
    defer dec.deinit();

    const audio = try dec.decode(allocator, tokens);
    defer allocator.free(audio);

    try stdout.print("Complete EnCodec Decoder Pipeline finished successfully!\n", .{});
    try stdout.print("Decoded PCM Samples Count: {d} (Sample Rate: {d}Hz, Duration: {d:.4}s)\n", .{
        audio.len,
        decoder.sample_rate,
        @as(f32, @floatFromInt(audio.len)) / @as(f32, @floatFromInt(decoder.sample_rate)),
    });
    try stdout.flush();

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
    const audio = wav.readMonoFile(io, allocator, input_path) catch |err| {
        std.log.err("failed to read {s}: {t}", .{ input_path, err });
        return err;
    };
    defer audio.deinit(allocator);
    if (audio.sample_rate != encoder.sample_rate) {
        std.log.err("{s}: sample rate {d} Hz is not supported", .{ input_path, audio.sample_rate });
        return error.UnsupportedSampleRate;
    }

    var enc = try encoder.Encoder.load(allocator, weights);
    defer enc.deinit();

    const frames = try enc.encode(allocator, audio.samples);
    defer allocator.free(frames);

    try stdout.print("Encoded {d} samples into {d} frames\n", .{ audio.samples.len, frames.len });
    try stdout.flush();

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
    try std.testing.expect(!options.stream);
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

    try std.testing.expect((try parseArgs(&.{"--stream"}, &failed_arg)).stream);
    try std.testing.expect((try parseArgs(&.{"-s"}, &failed_arg)).stream);
}

test "parseArgs rejects missing values, unknown and conflicting arguments" {
    var failed_arg: []const u8 = "";
    try std.testing.expectError(error.MissingValue, parseArgs(&.{"--tokens"}, &failed_arg));
    try std.testing.expectEqualStrings("--tokens", failed_arg);
    try std.testing.expectError(error.MissingValue, parseArgs(&.{"--encode"}, &failed_arg));
    try std.testing.expectError(error.UnknownArgument, parseArgs(&.{"--bogus"}, &failed_arg));
    try std.testing.expectEqualStrings("--bogus", failed_arg);
    try std.testing.expectError(error.ConflictingArguments, parseArgs(&.{ "-e", "a.wav", "-t", "a.bin" }, &failed_arg));
    try std.testing.expectEqualStrings("--encode", failed_arg);
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
