# Results

One folder per question, named `<YYYY-MM-DD>-<question>` so they sort by date.
Each folder holds raw measurements, never statistics: p50, p99, miss rates and energy
are computed from the tables when a figure or table is made, so any other statistic
can be computed later from the same data.

| Folder | Question | Answer so far | Status |
|---|---|---|---|
| [2026-11-04-gate1-dummy](2026-11-04-gate1-dummy/) | Does the harness keep time and count misses as the dummy model predicts? | Yes on the Thor (all four checks pass); macOS descheduling breaks (a) | done; raw cells lost, only checker output kept |
| [2026-11-04-gate2-real-model](2026-11-04-gate2-real-model/) | Does Gemma 3 4B answer keep/drop in one token, correctly, at steady latency? | One token and 96% agreement yes; latency spread failed because a labmate shared the GPU | done |
| [2026-11-19-space-multitenancy-initial](2026-11-19-space-multitenancy-initial/) | Do other VLM tenants on the GPU make tenant 0 miss its deadline? | Yes: two busy neighbours make every frame ~2.9x slower; equal MPS partitions isolate but are always slow | first look, 12 frames per cell |

## What a folder holds

```
README.md        question, answer up front, figures, caveats, how to rebuild
cells.csv        one row per cell: settings and provenance (raw_dir points at the logs)
requests.csv     one row per request; `tenant` 0 is the measured one, 1, 2, ... its neighbours
co_tenants.csv   other GPU processes seen when each measured request was sent
power.csv        GPU and board watts and GPU clock, one row per sample (100 Hz)
gpu_util.csv     GPU load, one row per sample (10 Hz)
manifests/       each cell's manifest.json, verbatim
figures/         drawn from the tables by a harness command; never edited by hand
```

Every table has a `cell` column to join on. Times are the harness's raw monotonic
seconds: subtract a cell's first `t_arrival` to plot. An empty field means the value
was not recorded. Units are in column names (`_s`, `_ms`, `_w`, `_mhz`, `_pct`).

## Making one

```bash
# 1. run cells (raw logs go to /data/projects/rohan/runs/<experiment>/<cell>-<timestamp>/)
uv run python -m harness.controller configs/<experiment>.toml
# 2. copy the cells that answer the question into a results folder
uv run python -m harness.collect /data/projects/rohan/runs/<experiment>/<cells...> -o results/<date>-<question>
# 3. compute and draw
uv run python -m harness.compare results/<date>-<question>          # any cells: table + miss-rate curves
uv run python -m harness.plot_tenancy results/<date>-<question>     # multi-tenant experiments
# 4. write README.md: the question, the answer, the figures
```

To compute something new in Python:

```python
from pathlib import Path
from harness import results
from harness.report import miss_rate_at

r = results.load(Path("results/2026-11-19-space-multitenancy-initial"))
for cell in r.cells:
    records = r.records(cell["cell"])  # RequestRecord objects; .latency_s, .missed
    print(cell["sharing"], cell["load"], miss_rate_at(records, 2.5))
```

The tables are plain CSV, so `csv`, numpy or pandas read them directly too.
Keep a folder under about 50 MB; past that, switch `power.csv` to Parquet.
