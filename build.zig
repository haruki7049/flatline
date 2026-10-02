const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hoge",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the decoder");
    const run_cmd = b.addRunArtifact(exe);
    // Weights and output.wav are resolved relative to the project root.
    run_cmd.setCwd(b.path("."));
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_tests.step);

    // Native audio I/O binaries, replacing the Python ends of the pipeline. Both
    // link zaudio/miniaudio, with no Python/torch in the runtime path.
    const zaudio_dep = b.dependency("zaudio", .{
        .target = target,
        .optimize = optimize,
    });

    const player_exe = addZaudioExecutable(b, target, optimize, zaudio_dep, "player", "src/player.zig");
    b.installArtifact(player_exe);

    const run_player_step = b.step("run-player", "Run the response-asset player");
    const run_player_cmd = b.addRunArtifact(player_exe);
    run_player_cmd.setCwd(b.path("."));
    run_player_cmd.step.dependOn(b.getInstallStep());
    run_player_step.dependOn(&run_player_cmd.step);

    // Native microphone capture + RMS-based VAD flagging. Replaces
    // stream_mic_encoder.py; writes the same 0xAA55 protocol that receiver.zig
    // and player.zig already speak.
    const capture_exe = addZaudioExecutable(b, target, optimize, zaudio_dep, "capture", "src/capture.zig");
    b.installArtifact(capture_exe);

    const run_capture_step = b.step("run-capture", "Run the microphone capture binary");
    const run_capture_cmd = b.addRunArtifact(capture_exe);
    run_capture_cmd.setCwd(b.path("."));
    run_capture_cmd.step.dependOn(b.getInstallStep());
    run_capture_step.dependOn(&run_capture_cmd.step);
}

/// Builds an executable that links zaudio/miniaudio, with the macOS framework
/// search path fallback both player.zig and capture.zig need (see below).
fn addZaudioExecutable(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    zaudio_dep: *std.Build.Dependency,
    name: []const u8,
    root_source_file: []const u8,
) *std.Build.Step.Compile {
    const module = b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("zaudio", zaudio_dep.module("root"));
    module.linkLibrary(zaudio_dep.artifact("miniaudio"));

    // zaudio's own build.zig resolves macOS frameworks through a lazy "system_sdk"
    // dependency that doesn't reliably link its framework search path through to
    // consumers in every environment (seen here: "-framework CoreAudio" is passed
    // but with no "-F" path, so the linker can't find it even though the
    // frameworks exist on-disk via the active SDK). Point the module at that SDK's
    // Frameworks directory directly as a fallback.
    if (target.result.os.tag == .macos) {
        const sdk_frameworks = b.run(&.{ "sh", "-c", "xcrun --show-sdk-path 2>/dev/null" });
        const sdk_path = std.mem.trimEnd(u8, sdk_frameworks, "\n");
        if (sdk_path.len > 0) {
            module.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" }) });
        }
    }

    return b.addExecutable(.{
        .name = name,
        .root_module = module,
    });
}
