#!/usr/bin/env bash
# Plan section 3, B1: LeRobot env for SmolVLA and pi0.5 on synthetic observations.
# torch 2.10 (cu130, proven on this box). No `dataset` extra, so no torchcodec
# (which wants torch >= 2.11 on aarch64); synthetic inputs need no video decoding.
# Idempotent. Run when the GPU is free: the final check initialises CUDA.
set -euo pipefail

ROOT=/data/projects/pulak
ENV=$ROOT/envs/lerobot
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

[ -d $ENV ] || uv venv $ENV --python 3.12
uv pip install --python $ENV/bin/python --torch-backend cu130 \
  "torch==2.10.0" "torchvision==0.25.0" "lerobot[smolvla,pi]==0.6.1"
uv pip install --python $ENV/bin/python -e "$REPO"   # the harness itself

hf download lerobot/smolvla_base
hf download lerobot/pi05_libero_finetuned_v044
# SmolVLA and pi0.5 also pull their VLM backbones' tokenizer/config on first load.

$ENV/bin/python - <<'PY'
import torch, lerobot
print("lerobot", lerobot.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("gpu ok:", torch.cuda.is_available(), "| arches:", torch.cuda.get_arch_list())
assert torch.cuda.is_available() and "sm_110" in torch.cuda.get_arch_list()
PY
