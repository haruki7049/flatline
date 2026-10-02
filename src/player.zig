//! Native response-asset player. Replaces python/pipeline/stream_decoder.py: reads
//! the 0xAA55 "utterance committed" packets that receiver.zig forwards, discards
//! their payload (the trigger is all that matters here, not the user's speech),
//! and plays back a randomly-picked pre-built response from assets/responses/
//! (built offline by python/experiments/build_response_assets.py) via zaudio.
//!
//! No Python, torch or Bark in this process: every response WAV is decoded and
//! handed to zaudio once at startup, so a commit only has to pick one and call
//! Sound.start().

const std = @import("std");
const zaudio = @import("zaudio");
const wav = @import("wav.zig");

const MAGIC_NUMBER: u16 = 0xAA55;
const ASSETS_DIR = "assets/responses/";
const MANIFEST_PATH = ASSETS_DIR ++ "manifest.json";
const MANIFEST_MAX_SIZE = 1 << 20;

const Header = extern struct {
    magic: u16,
    version: u8,
    reserved: u8,
    payload_len: u32,
};

const ManifestEntry = struct {
    category: []const u8,
    id: u32,
    text: []const u8,
    bin: []const u8,
    wav: []const u8,
    num_frames: u32,
    duration_s: f64,
};

const Manifest = struct {
    sample_rate: u32,
    frame_rate_hz: u32,
    codebooks: u32,
    responses: []ManifestEntry,
};

const ResponseAsset = struct {
    category: []const u8,
    text: []const u8,
    duration_s: f64,
    sound: *zaudio.Sound,
    audio_buffer: *zaudio.AudioBuffer,
};

// Direct POSIX exact read to bypass runtime abstraction issues on pipes (same
// technique as receiver.zig).
fn readExact(fd: c_int, buffer: []u8) !bool {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const rc = std.c.read(fd, buffer.ptr + total_read, buffer.len - total_read);
        if (rc == 0) {
            if (total_read == 0) return false;
            return error.UnexpectedEof;
        }
        if (rc < 0) {
            return error.ReadFailed;
        }
        total_read += @intCast(rc);
    }
    return true;
}

fn loadResponseAssets(
    io: std.Io,
    allocator: std.mem.Allocator,
    engine: *zaudio.Engine,
) ![]ResponseAsset {
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(io, MANIFEST_PATH, allocator, .limited(MANIFEST_MAX_SIZE));
    defer allocator.free(manifest_bytes);

    const parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{});
    defer parsed.deinit();
    const manifest = parsed.value;

    const assets = try allocator.alloc(ResponseAsset, manifest.responses.len);
    for (assets, manifest.responses) |*asset, entry| {
        const wav_path = try std.fmt.allocPrint(allocator, "{s}{s}", .{ ASSETS_DIR, entry.wav });
        defer allocator.free(wav_path);

        const audio = try wav.readMonoFile(io, allocator, wav_path);

        var buffer_config = zaudio.AudioBuffer.Config.init(.float32, 1, audio.samples.len, audio.samples.ptr);
        buffer_config.sample_rate = audio.sample_rate;
        const audio_buffer = try zaudio.AudioBuffer.create(buffer_config);

        const sound = try engine.createSoundFromDataSource(audio_buffer.asDataSourceMut(), .{}, null);

        asset.* = .{
            .category = try allocator.dupe(u8, entry.category),
            .text = try allocator.dupe(u8, entry.text),
            .duration_s = entry.duration_s,
            .sound = sound,
            .audio_buffer = audio_buffer,
        };
    }
    return assets;
}

fn destroyAssets(assets: []const ResponseAsset) void {
    for (assets) |asset| {
        asset.sound.destroy();
        asset.audio_buffer.destroy();
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    zaudio.init(allocator);
    defer zaudio.deinit();

    const engine = try zaudio.Engine.create(null);
    defer engine.destroy();
    try engine.start();

    std.debug.print("[flatline-player] Loading response assets from {s}...\n", .{ASSETS_DIR});
    const assets = loadResponseAssets(io, allocator, engine) catch |err| {
        std.debug.print(
            "[flatline-player] Failed to load {s}: {t}. Run python/experiments/build_response_assets.py first.\n",
            .{ MANIFEST_PATH, err },
        );
        std.process.exit(1);
    };
    defer destroyAssets(assets);
    std.debug.print("[flatline-player] Loaded {d} response assets.\n", .{assets.len});
    std.debug.print("[flatline-player] Ready. Listening for committed utterances...\n", .{});

    const seed_ns = std.Io.Timestamp.now(io, .real).nanoseconds;
    var prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @intCast(seed_ns))));
    const random = prng.random();

    while (true) {
        var header_buf: [@sizeOf(Header)]u8 = undefined;
        const has_header = try readExact(0, &header_buf);
        if (!has_header) break;

        const header: *const Header = @ptrCast(@alignCast(&header_buf));
        if (header.magic != MAGIC_NUMBER) {
            std.debug.print("[flatline-player] Invalid magic number received\n", .{});
            return error.InvalidPacket;
        }

        // The committed utterance's raw PCM is only a trigger here: its content is
        // discarded. We are not transcribing it (yet) - see docs/architecture.md.
        const payload = try allocator.alloc(u8, header.payload_len);
        defer allocator.free(payload);
        if (!try readExact(0, payload)) return error.IncompletePayload;

        const commit_time = std.Io.Timestamp.now(io, .awake);

        const asset = &assets[random.uintLessThan(usize, assets.len)];
        try asset.sound.seekToPcmFrame(0);
        try asset.sound.start();

        const playback_time = std.Io.Timestamp.now(io, .awake);
        const latency_ms = @as(f64, @floatFromInt(commit_time.durationTo(playback_time).nanoseconds)) / 1_000_000.0;
        std.debug.print(
            "[flatline-player] Utterance committed. Routing to \"{s}\" ({s}). Commit-to-playback latency: {d:.1}ms\n",
            .{ asset.text, asset.category, latency_ms },
        );

        while (!asset.sound.isAtEnd()) {
            try io.sleep(.fromMilliseconds(5), .awake);
        }
        std.debug.print("[flatline-player] Playback finished ({d:.2}s).\n", .{asset.duration_s});
    }
}
