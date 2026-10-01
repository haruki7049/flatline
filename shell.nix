{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  },
}:

let
  isLinux = pkgs.stdenv.hostPlatform.isLinux;
  isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

  # ROCm packages pinned only for Linux
  pkgs2411 = if isLinux then import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.11.tar.gz") {} else null;
  rocmDevLibs = if isLinux then pkgs2411.rocmPackages.rocm-device-libs else null;

  commonInputs = [
    pkgs.zig_0_16
    pkgs.zls_0_16
    pkgs.jq
    pkgs.python311
    pkgs.zstd
    pkgs.zlib
  ];

  linuxInputs = if isLinux then [
    pkgs.rocmPackages.clr
    pkgs.rocmPackages.rocminfo
    pkgs.rocmPackages.rocm-smi
    pkgs.rocmPackages.rocm-runtime
    pkgs.stdenv.cc.cc.lib
    pkgs.numactl
    pkgs.libxml2
    pkgs.libdrm
    pkgs.elfutils
    pkgs.pciutils
  ] else [];

in
pkgs.mkShell {
  nativeBuildInputs = commonInputs ++ linuxInputs;

  shellHook = if isLinux then ''
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export HIP_VISIBLE_DEVICES=0
    export ROCM_PATH=${pkgs.rocmPackages.clr}
    export DEVICE_LIB_PATH="${rocmDevLibs}/amdgcn/bitcode"
    export HIP_DEVICE_LIB_PATH="${rocmDevLibs}/amdgcn/bitcode"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zstd
      pkgs.zlib
      pkgs.numactl
      pkgs.libxml2
      pkgs.libdrm
      pkgs.elfutils
      pkgs.pciutils
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      "/run/opengl-driver"
    ]}:$LD_LIBRARY_PATH"
  '' else ''
    # macOS Darwin environment initialized
  '';
}

