# TODO.md — system-setup-v1

Ordered by dependency. Source: `control-folder/plans/system-setup-v1.md`.

**Current scope (user, 2026-10-12):** synthetic inputs only, model running alone, both stacks.
Stack A: synthetic image in, label out. Stack B: synthetic observation (images + robot state) in, action chunk out.
Report latency percentiles (p50, p99) and throughput (requests/s, a plain rate; p99 is a latency term only).
Real data (Sentinel-2, LIBERO sim), co-located load and power-mode sweeps come later.

## Step 0: blockers (people, not code)
- [ ] Agree GPU booking convention: `/data/gpu-lock` (owner, purpose, ETA) + post in `#latency-bound-inf`
- [ ] Ask admin for sudo on `nvpmodel`, `jetson_clocks`, MIG (or a booked window)
- [x] `hf auth login` with a token that has Gemma 3 access
- [x] Fix the two command bugs in the plan (`--no-cont-batching`; no `pip install --user`)

## Step 1: A1 llama.cpp + Gemma 3 4B QAT Q4_0
- [x] Install `uv` (standalone installer) and `hf` CLI (`uv tool install`)
- [x] Create `/data/projects/pulak/{models,envs,src,runs}`
- [x] Clone and build llama.cpp with CUDA for `sm_110` (`llama-mtmd-cli`, `llama-server`)
- [x] Download `google/gemma-3-4b-it-qat-q4_0-gguf`
- [x] Smoke test: one synthetic image, one label, GPU rail moves in tegrastats
- [x] Start llama-server single slot on port 8081

## Step 2: harness v1 on A1
- [x] Synthetic image generator (fixed size, seeded, reproducible)
- [x] Record schema + JSONL writer
- [x] Fixed-rate feeder on a monotonic clock (latency mode)
- [x] Closed-loop / saturation mode (throughput mode)
- [x] llama-server client adapter (streaming, engine `timings`)
- [x] Co-tenant snapshot and power-mode capture
- [x] tegrastats logger
- [x] Reporter: latency p50, p99, p99/p50, miss rate, Wilson CI, CDF plot; throughput in requests/s; energy per request
- [x] Run 100 synthetic requests at T = 30 s, GPU idle; reproduce report from log alone (2026-10-12; latency inflated ~27 ms by sleep overshoot, since fixed)
- [x] Re-run the 100-request A1 run with the sleep fix (a1-latency-20261012-210527: p50 1659 ms, p99 1670 ms, 0/100 missed)
- [ ] Investigate idle-gap slowdown: SmolVLA 289 ms at 1 Hz vs 188 ms back to back (GPU clocks?)
- [ ] Throughput run at saturation; report requests/s

## Step 4: B1 LeRobot on synthetic observations
- [x] Decide torch 2.10 vs 2.11: torch 2.10, no `dataset` extra, so no torchcodec
- [x] pi0.5 LeRobot checkpoint: `lerobot/pi05_libero_finetuned_v044` (old name `lerobot/pi05_libero` redirects)
- [x] Synthetic observation generator (camera images + state, shapes from the policy config, seeded)
- [x] In-process policy adapter (preprocess, predict chunk, postprocess; per-stage timings)
- [x] Shared run plumbing (`harness/runner.py`) used by both stacks
- [x] LeRobot env: `setup/b1_lerobot.sh` (lerobot 0.6.1, torch 2.10.0+cu130, sm_110 OK)
- [x] Run unit tests, lint, re-check A1 after the runner refactor (short A1 run)
- [x] SmolVLA: short latency and throughput runs
- [x] pi0.5: short latency and throughput runs (torch.compile needs system ptxas for sm_110a)
- [ ] LeRobot async policy server (gRPC) adapter, to compare with in-process

## Step 5: A2 vLLM
- [ ] Find NGC Thor vLLM tag, or build from source for `sm_110`
- [ ] bf16, AWQ, NVFP4 runs at `--max-num-seqs 1`; harness adapter for vLLM

## Step 6: B2 openpi (PyTorch path)
- [ ] Override torch in pyproject, sync, copy `transformers_replace`, convert `pi05_libero`
- [ ] Serve and run LIBERO client natively

## Step 7: B3 EagleVLA-Edge
- [ ] Build fork for `sm_110`, follow `master` README for pi0.5
- [ ] Per-inference tail with and without graph reuse

## Step 8: as time allows
- [ ] A3 TensorRT-LLM, A0 transformers baseline
- [ ] B4 VLASH and ETH baselines
- [ ] MIG partitioning, Titan connected cell

## Deferred (later scope)

- LIBERO simulator closed loop (B1 sanity eval, openpi LIBERO client)

## Step 3: A1 under load (slide 7)
- [ ] Same run with co-located load
- [ ] Same run at 70 W (needs sudo)
- [ ] Three CDFs on shared axes
