# CLAUDE.md

## Project

Thesis work on latency-bound inference: when an edge GPU is given a deadline, where does inference miss it,
why does its latency vary, and how much determinism can the GPU borrow before work must move to an FPGA.
Core questions: `control-folder/notes/core-questions.md`. Current plan: `control-folder/plans/system-setup-v1.md`.

Two use cases on one NVIDIA Jetson AGX Thor (JetPack 7, CUDA 13.2, `sm_110`, 122 GB unified memory):
- **Stack A:** Gemma 3 4B VLM labels satellite images, one image every T seconds.
- **Stack B:** pi0.5 / SmolVLA VLA policies on the LIBERO simulator.

## Current scope

`DESIGN.md` is the spec for Stack A: a fixed-rate frame stream through a depth-1 mailbox,
one-token keep/drop answers, miss rate at deadline D applied to the log afterwards.
Build order and gates are in DESIGN.md section 4; gate reports go in `reports/`.
Co-runners and power modes are later extensions; the config loader refuses cells that ask for them.
Stack B (`run_b1.py`) keeps the older fixed-rate runner and shares records and reporting.

## Layout

- `control-folder/` — notes, slides, plans. `DESIGN.md` at the root is the current spec.
- `harness/` — feeder and mailbox, engines, device sampling, controller, reporter, gate checker. Python 3.12.
- `configs/` — TOML cell matrices; every tunable lives here, not in code.
- `setup/` — scripts that build engines and envs, and cut Sentinel-2 tiles.
- Heavy artifacts never go in git. On the Thor they live under `/data/projects/rohan/{models,src,runs,tiles}`
  (`/data/projects/pulak/` is the colleague's tree: usable, not ours to change).

## Build and environment

- Python envs with `uv`, one env per stack, never shared. System pip is PEP 668 locked; do not `pip install --user`.
- Torch pin: `torch==2.10.0` from `https://download.pytorch.org/whl/cu130`.
- CUDA builds target `CMAKE_CUDA_ARCHITECTURES=110` / `TORCH_CUDA_ARCH_LIST="11.0"`.
- Docker only where NVIDIA ships the image (NGC, needs `docker login nvcr.io`).
- HF token at `~/.cache/huggingface/token` (`hf auth login`). Gemma 3 is gated.

## Running

```bash
uv run pytest && uv run ruff check && uv run ruff format --check     # local checks
uv run python -m harness.controller configs/gate1.toml --engine dummy  # gate 1, no GPU needed
uv run python -m harness.gate1 runs/gate1-T1-<timestamp>              # gate 1 checks
uv run python -m harness.controller configs/stack_a.toml --cells alone-T20 --max-minutes 90  # Stack A
/data/projects/pulak/envs/lerobot/bin/python -m harness.run_b1 --policy smolvla latency  # Stack B
uv run python -m harness.report <cell dir>                            # rebuild a report from logs
uv run python -m harness.compare <cell dirs...> -o reports/<name>     # cells on one axis
```

`test_sleep_until_wakes_on_time_after_a_long_wait` can fail on macOS (timer jitter); it is a Linux check.

Readability matters for `harness/` (reviewed and reused). `setup/` scripts only need to work.

## Conventions

- Every cell writes `manifest.json`, `records.jsonl`, `power.csv`, `gpu_util.csv`, `tegrastats.log` and the report into its own directory; a rerun gets a new directory, nothing is overwritten.
- Every record carries engine, model, quant, power mode, co-tenant GPU processes at send time, and an `outcome` (`ok`, `overwritten`, `error`).
- The feeder never waits for the model. Frames fire on a monotonic clock into a depth-1 mailbox; a replaced frame is a miss.
- Stack A answers with one token (`max_tokens = 1`, temperature 0), so each request is pure prefill.
- tegrastats on JetPack 7 has no GPU clock or load; the clock comes from `/sys/class/devfreq/gpu-gpc-0/cur_freq`, load from `nvidia-smi`.

## Key design decisions

- The GPU is shared with labmates. Check `nvidia-smi --query-compute-apps` and `nvpmodel -q` before trusting any run.
- Power mode and clocks need sudo (`nvpmodel`, `jetson_clocks`); not available yet.
- Record both client-side and engine-side timings; the gap is the transport/queue/Python share of latency.
- Git push uses SSH (`git@github.com:pulak999/latency-bound-inf.git`); the HTTPS remote has no credentials.

## Tests

Unit tests under `harness/tests/` (`uv run pytest`). CI (`.github/workflows/ci.yml`) lints, installs and runs every `--help`.
