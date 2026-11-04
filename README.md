# latency-bound-inf

Deadline-miss measurements for edge inference on an NVIDIA Jetson AGX Thor.
The design is in `DESIGN.md`; the thesis notes and plans are in `control-folder/`.

Stack A: a 4-bit Gemma 3 4B VLM labels one satellite tile per frame on a fixed
cadence T, through a depth-1 mailbox, and the harness logs when each answer
arrived against the deadline. Stack B (robot policies, `harness/run_b1.py`) uses
the same records and reporter.

## From SSH login to a running dummy cell

The dummy cell needs no GPU and runs anywhere, the Thor included.

```bash
git clone git@github.com:pulak999/latency-bound-inf.git ~/latency-bound-inf
cd ~/latency-bound-inf
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh   # ~/.local/bin
uv sync
uv run pytest -q                                           # ~5 s
uv run python -m harness.controller configs/gate1.toml --engine dummy   # ~4 min, 200 frames
uv run python -m harness.gate1 runs/gate1-T1-<timestamp>  # checks (a) to (d) of DESIGN.md gate 1
```

The cell's directory holds `manifest.json`, `records.jsonl` (one line per frame),
`power.csv` (power and GPU clock at 10 ms), `gpu_util.csv`, `tegrastats.log`,
`summary.json` and three plots: `latency_cdf.png`, `miss_vs_deadline.png`, `timeline.png`.

## The real model

One-off setup on the Thor (llama.cpp for sm_110 plus the Gemma GGUF; needs `hf auth login`):

```bash
ROOT=/data/projects/rohan setup/a1_llamacpp.sh
```

Then, with the GPU otherwise idle (`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`):

```bash
uv run python -m harness.controller configs/stack_a.toml --cells alone-T20 --max-minutes 90
uv run python -m harness.report  /data/projects/rohan/runs/alone-T20-<timestamp>     # rebuild from logs
uv run python -m harness.compare /data/projects/rohan/runs/alone-T* -o reports/alone  # cells on one axis
```

Every tunable (prompt, token cap, cadences, frame counts, warm-up, paths) is in
the config file. Nothing numeric that affects a result lives in code. A cell that
asks for a co-runner or a power mode is refused until those are implemented.

## Tiles

Real inputs are Sentinel-2 L2A true-colour crops at 896x896 with their SCL band,
cut on any machine with internet and copied to the Thor:

```bash
uv run --group tiles python setup/tiles_sentinel2.py --out tiles --n-tiles 1000
rsync -a tiles/ jetson-thor:/data/projects/rohan/tiles/
```

If `[tiles] dir` in the config does not exist, synthetic noise images are used instead.

## Layout

- `harness/` — feeder and mailbox, engines (`llama_server`, `dummy`), device sampling,
  controller, reporter, cross-cell comparison, gate-1 checker, tests.
- `configs/` — cell matrices. `setup/` — build and data scripts. `reports/` — gate reports.
- Runs live outside git: `/data/projects/rohan/runs/` on the Thor, `runs/` locally.

## Timing semantics

Latency is `t_end - t_obs` (DESIGN.md): from the frame's arrival, so time waiting in
the mailbox counts. A frame misses deadline D if it finished later than D after
arrival, or was overwritten by the next frame before the model took it. D is
applied in the reporter, so one log gives the miss rate at every deadline.

Record field names map to DESIGN.md as `t_arrival` = `t_obs`, `t_send` = `t_start`,
`t_last_token` = `t_end`, `engine_prompt_ms` = `prompt_ms`, `output_text` = `label`.
