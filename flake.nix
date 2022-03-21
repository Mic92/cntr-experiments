{
  description = "Spawn debug shells in virtual machines";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, }:
    (flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        packages = rec {
          # see justfile/nixos-image
          phoronix-image = pkgs.callPackage ./nix/phoronix-image.nix {};

          phoronix-test-suite = pkgs.callPackage ./nix/phoronix.nix {};
        };
      }));
}
