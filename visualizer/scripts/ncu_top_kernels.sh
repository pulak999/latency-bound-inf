#!/usr/bin/env bash
# STARTER (not yet run): Nsight Compute on the top kernels of a captured bundle.
#
#   sudo -E visualizer/scripts/ncu_top_kernels.sh <bundle dir> [top N kernels, default 6]
#   uv run python -m visualizer.bundle <bundle dir>      # then rebuild to pull the results in
#
# Needs root: the Thor restricts GPU performance counters to admins
# (RmProfilingAdminOnly: 1). Replays one frame with llama-mtmd-cli (same
# libraries as llama-server, one image, then exit) instead of the server, so
# ncu does not have to find one frame inside a long-running process. For each
# of the top N kernel names (by time in the bundle) it profiles the first two
# launches with the full section set, warp states over time, and per-source-line
# stall samples, and writes <bundle>/ncu/<kernel>.ncu-rep and .raw.csv.
#
# Source-level stalls only map to ggml source lines if llama.cpp was built
# with -lineinfo: set LLAMA_BIN to that build's bin/ directory.
set -euo pipefail

BUNDLE=${1:?usage: ncu_top_kernels.sh <bundle dir> [top N]}
TOP=${2:-6}
NCU=${NCU:-/opt/nvidia/nsight-compute/2026.1.1/ncu}
LLAMA_BIN=${LLAMA_BIN:-/data/projects/rohan/src/llama.cpp/build/bin}
MODEL_DIR=${MODEL_DIR:-/data/projects/rohan/models/gemma-3-4b-it-qat-q4_0}
TILES=${TILES:-/data/projects/rohan/tiles}
PROMPT=${PROMPT:-"Decide whether to keep or drop this satellite image for downlink. Drop it if it is mostly cloud or open water. Answer with exactly one word: keep or drop."}

mkdir -p "$BUNDLE/ncu"

# Top kernel names by total time, and the server args the bundle was captured with.
readarray -t KERNELS < <(python3 - "$BUNDLE" "$TOP" <<'EOF'
import json, sys, collections
bundle = json.load(open(f"{sys.argv[1]}/bundle.json"))
k, total = bundle["kernels"], collections.Counter()
for name, dur in zip(k["name"], k["dur_us"]):
    total[bundle["names"][name]["short"]] += dur
for name, _ in total.most_common(int(sys.argv[2])):
    print(name)
EOF
)
readarray -t EXTRA < <(python3 -c "import json,sys; [print(a) for a in json.load(open(sys.argv[1]))['server_args']]" "$BUNDLE/meta.json")
TILE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tile'])" "$BUNDLE/meta.json")

for KERNEL in "${KERNELS[@]}"; do
  OUT="$BUNDLE/ncu/${KERNEL//[^A-Za-z0-9_]/_}"
  echo "== $KERNEL -> $OUT"
  "$NCU" --target-processes all --graph-profiling node \
    --kernel-name "regex:^${KERNEL}$" --launch-count 2 \
    --set full --section PmSampling_WarpStates --import-source yes \
    --force-overwrite --export "$OUT" \
    "$LLAMA_BIN/llama-mtmd-cli" -m "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" \
    --mmproj "$MODEL_DIR/mmproj-model-f16-4B.gguf" -ngl 99 -n 1 --temp 0 \
    --image "$TILES/$TILE.png" -p "$PROMPT" "${EXTRA[@]}" > "$OUT.log" 2>&1
  "$NCU" --import "$OUT.ncu-rep" --page raw --csv > "$OUT.raw.csv"
  "$NCU" --import "$OUT.ncu-rep" --page source --csv > "$OUT.source.csv" || true
done
chown -R "${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}" "$BUNDLE/ncu"
echo "done; now: uv run python -m visualizer.bundle $BUNDLE"
