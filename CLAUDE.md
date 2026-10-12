# CLAUDE.md

## Project

Thesis work on latency-bound inference: when an edge GPU is given a deadline, where does inference miss it,
why does its latency vary, and how much determinism can the GPU borrow before work must move to an FPGA.
Core questions: `control-folder/notes/core-questions.md`. Current plan: `control-folder/plans/system-setup-v1.md`.

Two use cases on one NVIDIA Jetson AGX Thor (JetPack 7, CUDA 13.2, `sm_110`, 122 GB unified memory):
- **Stack A:** Gemma 3 4B VLM labels satellite images, one image every T seconds.
- **Stack B:** pi0.5 / SmolVLA VLA policies on the LIBERO simulator.

## Current scope

Synthetic inputs only, model running alone on an idle GPU, both stacks.
Stack A: one synthetic image in, one label out. Stack B: one synthetic observation in, one action chunk out.
Metrics: latency percentiles (p50, p99) and throughput as requests/s. p99 is a latency term only.
Real data (Sentinel-2, LIBERO), co-located workloads and power-mode sweeps are later.

## Layout

- `control-folder/` — notes, slides, plans. Plans are the source of truth for what to build.
- `harness/` (planned) — shared fixed-rate feeder, JSONL records, tegrastats capture, reporting. Python 3.12.
- `setup/` (planned) — scripts that build engines and envs under `/data/projects/pulak/`.
- Heavy artifacts never go in git. They live under `/data/projects/pulak/{models,envs,src,runs}`.

## Build and environment

- Python envs with `uv`, one env per stack, never shared. System pip is PEP 668 locked; do not `pip install --user`.
- Torch pin: `torch==2.10.0` from `https://download.pytorch.org/whl/cu130`.
- CUDA builds target `CMAKE_CUDA_ARCHITECTURES=110` / `TORCH_CUDA_ARCH_LIST="11.0"`.
- Docker only where NVIDIA ships the image (NGC, needs `docker login nvcr.io`).
- HF token at `~/.cache/huggingface/token` (`hf auth login`). Gemma 3 is gated.

## Conventions

- Every measurement run writes one JSONL of per-request records plus a `tegrastats` log into `/data/projects/pulak/runs/<run>/`.
- Every record carries engine, model, quant, power mode and co-tenant GPU processes at send time.
- The feeder never waits for the model. Requests fire on a monotonic clock.
- Generation output is fixed-length (temperature 0, fixed `-n`) so latency reflects the system, not output length.

## Key design decisions

- The GPU is shared with labmates. Check `nvidia-smi --query-compute-apps` and `nvpmodel -q` before trusting any run.
- Power mode and clocks need sudo (`nvpmodel`, `jetson_clocks`); not available yet.
- Record both client-side and engine-side timings; the gap is the transport/queue/Python share of latency.
- Git push uses SSH (`git@github.com:pulak999/latency-bound-inf.git`); the HTTPS remote has no credentials.

## Tests

None yet. Harness logic gets pytest unit tests under `harness/tests/`. No CI workflow exists.
