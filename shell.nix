{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.zig_0_16
    pkgs.zls_0_16
  ];
}
