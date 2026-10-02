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

    // Native response-asset player: replaces stream_decoder.py. Reads the 0xAA55
    // commit protocol from stdin and plays back pre-built assets/responses/ via
    // zaudio/miniaudio, with no Python/torch in the runtime path.
    const zaudio_dep = b.dependency("zaudio", .{
        .target = target,
        .optimize = optimize,
    });

    const player_module = b.createModule(.{
        .root_source_file = b.path("src/player.zig"),
        .target = target,
        .optimize = optimize,
    });
    player_module.addImport("zaudio", zaudio_dep.module("root"));
    player_module.linkLibrary(zaudio_dep.artifact("miniaudio"));

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
            player_module.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" }) });
        }
    }

    const player_exe = b.addExecutable(.{
        .name = "player",
        .root_module = player_module,
    });
    b.installArtifact(player_exe);

    const run_player_step = b.step("run-player", "Run the response-asset player");
    const run_player_cmd = b.addRunArtifact(player_exe);
    run_player_cmd.setCwd(b.path("."));
    run_player_cmd.step.dependOn(b.getInstallStep());
    run_player_step.dependOn(&run_player_cmd.step);
}
