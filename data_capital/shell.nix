# nix-shell 로 진입: $ nix-shell
# 패키지 설치 후 테스트: $ cd data_capital && python3 -m pytest tests/ -v
{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonEnv = python.withPackages (ps: with ps; [
    pandas
    numpy
    scipy
    pytest
    requests
    # pykrx는 nixpkgs에 없으므로 pip로 설치 필요:
    #   pip install pykrx
  ]);
in pkgs.mkShell {
  buildInputs = [ pythonEnv pkgs.python312Packages.pip ];

  shellHook = ''
    echo "DATA CAPITAL 개발 환경"
    echo "pykrx 미설치 시: pip install pykrx --user"
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
}
