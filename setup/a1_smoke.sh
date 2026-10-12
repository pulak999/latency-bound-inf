#!/usr/bin/env bash
# Chunk 1 smoke test: one seeded synthetic image -> Gemma 3 4B QAT Q4_0 via llama-mtmd-cli -> one label.
# Passes if a non-empty label comes back and the VDD_GPU rail rises above its idle reading.
set -euo pipefail

ROOT=/data/projects/pulak
BIN=$ROOT/src/llama.cpp/build/bin
M=$ROOT/models/gemma-3-4b-it-qat-q4_0
OUT=$ROOT/runs/smoke-a1-$(date +%Y%m%d-%H%M%S)
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $OUT

# Seeded random-noise RGB image, 896x896 (Gemma 3's native vision resolution).
uv run --quiet --with pillow --with numpy python - "$OUT/synthetic.png" <<'PY'
import sys, numpy as np
from PIL import Image
rng = np.random.default_rng(0)
Image.fromarray(rng.integers(0, 256, (896, 896, 3), dtype=np.uint8)).save(sys.argv[1])
PY

tegrastats --interval 100 --logfile $OUT/tegrastats.log &
TEGRA=$!
trap 'kill $TEGRA 2>/dev/null || true' EXIT
sleep 1

$BIN/llama-mtmd-cli -m $M/gemma-3-4b-it-q4_0.gguf --mmproj $M/mmproj-model-f16-4B.gguf \
  --image $OUT/synthetic.png -n 4 --temp 0 -ngl 99 \
  -p "Classify this satellite scene with one word from: forest, water, urban, cloud, field." \
  > $OUT/stdout.txt 2> $OUT/stderr.txt
sleep 1
kill $TEGRA; trap - EXIT

LABEL=$(tr -d '\n' < $OUT/stdout.txt | sed 's/^ *//;s/ *$//')
read IDLE PEAK < <(grep -o 'VDD_GPU [0-9]*' $OUT/tegrastats.log | awk '{print $2}' | awk 'NR==1{i=$1} {if($1>p)p=$1} END{print i, p}')
echo "label='$LABEL'  VDD_GPU idle=${IDLE}mW peak=${PEAK}mW  run=$OUT"
grep -E "encoding done|eval time" $OUT/stderr.txt || true

[ -n "$LABEL" ] || { echo "FAIL: empty label"; exit 1; }
[ "$PEAK" -gt $((IDLE + 1000)) ] || { echo "FAIL: GPU rail did not rise"; exit 1; }
echo PASS
