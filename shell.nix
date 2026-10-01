{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  },
}:

let
  # Pin NixOS 24.11 channel to supply LLVM 18-compatible ROCm device-libs
  pkgs2411 = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.11.tar.gz") {};
  rocmDevLibs = pkgs2411.rocmPackages.rocm-device-libs;
in
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.zig_0_16
    pkgs.zls_0_16
    pkgs.jq
    pkgs.python311
    pkgs.rocmPackages.clr
    pkgs.rocmPackages.rocminfo
    pkgs.rocmPackages.rocm-smi
    pkgs.rocmPackages.rocm-runtime
    pkgs.stdenv.cc.cc.lib
    pkgs.zstd
    pkgs.zlib
    pkgs.numactl
    pkgs.libxml2
    pkgs.libdrm
    pkgs.elfutils
    pkgs.pciutils
  ];

  shellHook = ''
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
  '';
}
