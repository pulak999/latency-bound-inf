#!/usr/bin/env bash
# Plan section 2, A1: llama.cpp + Gemma 3 4B QAT Q4_0 on the Jetson AGX Thor.
# Idempotent: re-running skips finished stages. Model download needs `hf auth login` first.
set -euo pipefail

ROOT=/data/projects/pulak
SRC=$ROOT/src/llama.cpp
MODEL_DIR=$ROOT/models/gemma-3-4b-it-qat-q4_0
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"

mkdir -p $ROOT/{models,envs,src,runs}

# uv and the hf CLI live in ~/.local/bin; system pip is PEP 668 locked.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v hf >/dev/null || uv tool install huggingface_hub

[ -d $SRC ] || git clone https://github.com/ggml-org/llama.cpp $SRC
if [ ! -x $SRC/build/bin/llama-server ]; then
  cmake -S $SRC -B $SRC/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build $SRC/build -j"$(nproc)" --target llama-mtmd-cli llama-server
fi
git -C $SRC rev-parse HEAD > $SRC/build/COMMIT

if hf auth whoami >/dev/null 2>&1; then
  hf download google/gemma-3-4b-it-qat-q4_0-gguf --local-dir $MODEL_DIR
else
  echo "Not logged in to Hugging Face: run 'hf auth login', then re-run this script." >&2
  exit 2
fi
