with import <nixpkgs> {};
mkShell {
  nativeBuildInputs = [
    bashInteractive
    python3.pkgs.lxml
    python3.pkgs.pandas
    python3.pkgs.psutil
    python3.pkgs.pytest
    cargo
    rustc
    mypy
  ];
}
