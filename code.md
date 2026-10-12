# code.md

## Code Review — commit dea61ae2f3a8cf8c7bcd00d9d809d5279fab4fcc (2026-10-12)

### Phase 1: Repo overview

1. **Purpose.** Research control folder for a thesis on latency-bound inference: where inference misses
   deadlines on an edge GPU (Jetson Thor), why latency varies, and how much FPGA-style determinism the GPU can borrow.
   There is no source code yet. The repo holds notes, slides and one implementation plan.
2. **Structure.**
   - `control-folder/notes/` — core questions, two LaTeX measurement notes (phys-AI, satellite), per-paper notes in `usecases-notes/`.
   - `control-folder/slides/` — kickoff Beamer deck (`main.tex`, `kickoff.pdf`).
   - `control-folder/plans/` — `system-setup-v1.md`, the plan being implemented.
   - `README.md` — one line.
3. **Entry points.** None. Nothing executes.
4. **Key data structures.** None in code. The plan defines one: the per-request JSONL record (plan section 5).
5. **Patterns.** Plan-driven research repo. Heavy artifacts (models, builds, venvs) are planned to live outside the repo under `/data/projects/pulak/`.

Structural oddities:
- LaTeX build artifacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, `.nav`, `.snm`, `.toc`) are committed. There is no `.gitignore`.
- No tests, no CI (`.github/` absent).

### Phase 2: File-by-file

- `control-folder/plans/system-setup-v1.md` — the plan. Sets up Stack A (Gemma 3 4B VLM: llama.cpp, vLLM, TRT-LLM, transformers) and
  Stack B (pi0.5 / SmolVLA VLA: LeRobot, openpi, EagleVLA-Edge, VLASH/ETH baselines) on the Thor, plus a shared fixed-rate harness.
  Issues found:
  - A1 `llama-server ... --cont-batching off`: `--cont-batching` is a boolean flag; the off form is `--no-cont-batching`. Will fail or misparse.
  - Conventions and A1 use `pip install --user`. Ubuntu 24.04 system Python is PEP 668 externally-managed, so this errors.
  - Section 5 records `t_first_token`; llama-server only gives a distinct first-token time when the request streams.
  - Step 1 (100 requests at T = 30 s, about 50 min) barely fits the one-hour idle window requested in step 0.
  - B1 pins torch 2.10 but LeRobot's `torchcodec` on aarch64 wants torch >= 2.11 (listed as open item; unresolved).
  - B2 converts from `gs://openpi-assets/...`; whether the converter reads `gs://` directly or needs a prior download is unverified.
- `control-folder/slides/main.tex` — kickoff deck. Slide 6 (measurement setup: W, NM, M, D) and slide 7 (first experiment).
  **Slide 7 names "Jetson Orin"; the plan runs everything on the Jetson AGX Thor.**
- `control-folder/notes/core-questions.md` — Q1 where it fails, Q2 why it varies, Q3 what fixes it. The plan's harness serves Q1/Q2; B3 is the first Q3 experiment.
- `control-folder/notes/usecases-notes/*.md` — paper notes; source for the plan's model and engine choices. No issues relevant to implementation.
- `control-folder/notes/*-measurement-v1/main.tex` — measurement notes. The satellite note carries the Sentinel-2 dataset/licence TODO the plan references.

### Phase 3: Cross-cutting

1. **Inconsistencies.** Slide 7 says Orin, plan says Thor. Plan conventions say `pip install --user`, which the OS refuses.
2. **Missing pieces.** No place in the repo for the harness code; the plan puts sources in `/data/projects/pulak/src/` (outside git).
   No test images or Sentinel-2 feeder data. No HF token on the box (`~/.cache/huggingface/` absent), and Gemma 3 is gated.
   No `uv`, no `hf` CLI installed.
3. **Most likely failure point.** Measurement contamination: the GPU is shared, there is no `/data/gpu-lock`, and no sudo for
   `nvpmodel`/`jetson_clocks`. A run that looks fine can carry co-tenant load silently unless the harness records it (plan does require this).
4. **Warn a new engineer.** Every number depends on GPU idleness and power mode. Check `nvidia-smi --query-compute-apps` and `nvpmodel -q` before trusting any run.

### Plan vs codebase cross-reference (implement-from-plan step 0.3)

Machine state checked 2026-10-12:

| Item | State |
|---|---|
| `/data/projects/pulak/` | Does not exist |
| `/data` | Exists, group `thorusers`, setgid; also has `checkpoints/`, `datasets/`, `shared/` (not ours) |
| `uv`, `hf` | Not installed |
| HF token | Absent |
| GPU | Idle at check time (0%, no compute apps) |
| Power mode | 120 W (default) |
| `/data/gpu-lock` | Does not exist |
| `gh` | Installed, not logged in |
| Git push | Remote is HTTPS with no credential helper, so push fails. SSH key authenticates to GitHub as `pulak999` |
| CPU / RAM | 14 cores, 122 GB |

- **Already implemented:** nothing. Plan sections 1 (machine facts) are descriptive and were rechecked above.
- **Preconditions before code:** step 0 blockers (GPU booking convention, sudo for `nvpmodel`/`jetson_clocks`) need people, not code.
  HF login needs the user's token. `uv` must be installed without system pip.
- **Conflicts:** slide 7 (Orin) vs plan (Thor). Plan's two command bugs above. Plan stores source under `/data`, outside this repo.
- **Tests / CI:** none exist. Any harness code needs new tests (pytest) and, if wanted, a new GitHub Actions workflow.
