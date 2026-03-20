{
  description = "RAG Hoopla dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              pkgs.python312
              pkgs.gcc
              pkgs.zlib
              pkgs.stdenv.cc.cc
            ];

            shellHook =
              let
                libPath = pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                  pkgs.zlib
                ];
              in
              ''
              export LD_LIBRARY_PATH=${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

              # Prevent uv from downloading its own Python — NixOS can't run them
              export UV_PYTHON_DOWNLOADS=never
              export UV_PYTHON_PREFERENCE=only-system

              # --- Virtual Environment Setup ---
              if [ ! -d ".venv" ]; then
                  echo "Creating Python virtual environment (.venv) with UV..."
                  uv venv --python ${pkgs.python312}/bin/python3
                  source .venv/bin/activate
                  uv add google-genai==1.12.1
                  uv add python-dotenv==1.1.0
                  uv add nltk==3.9.1
                  uv add sentence-transformers
                  uv add numpy
              else
                  source .venv/bin/activate
              fi

              export PS1="\n\[\033[1;32m\][RAG_Hoopla_env:\w]\$\[\033[0m\]"
            '';
          };
        });
    };
}
