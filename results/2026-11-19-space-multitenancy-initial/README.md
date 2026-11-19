# Initial naive results: space use case, 1 vs 3 VLM tenants (2026-11-19)

Quick and small: 12 frames per cell, so each miss rate has a 95% interval of about ±25 points.
Treat this as a first look, not a result.

Setup: Gemma 3 4B QAT Q4_0 on llama-server (our sm_110 build), Jetson Thor at 120 W,
GPU clock not pinned. Every tenant is its own container. The measured tenant gets one
Sentinel-2 tile every 6 s; 6 s is longer than any service time, so frames never queue
and any deadline can be applied to the logs afterwards. The other two tenants run the
same model, either bursty or saturating:

- **bursty**: each neighbour sends bursts of 5 requests back to back, then goes quiet. Bursts
  start at random times, one every 15 s on average, so about one request per 3 s overall.
- **saturating**: each neighbour sends its next request the moment the last one returns,
  so it never goes quiet.

Each neighbour has one request in flight at a time (one server slot) and its own client process.
Config: `configs/space_multitenancy_now.toml`.
Raw cells: `/data/projects/rohan/runs/space-multitenancy-v1/now-n*` (column `raw_dir` of `cells.csv`).

| cell | tenants | sharing | neighbours busy | p50 (s) | p99 (s) | missed at 3 s |
|---|---|---|---|---|---|---|
| alone | 1 | - | - | 1.55 | 1.56 | 0% |
| bursty, time-slicing | 3 | no MPS | 38% | 2.98 | 4.01 | 50% |
| saturating, time-slicing | 3 | no MPS | 100% | 4.54 | 4.73 | 100% |
| bursty, MPS partitions | 3 | 6 SMs each (2 idle) | 62% | 4.43 | 4.47 | 100% |
| saturating, MPS partitions | 3 | 6 SMs each (2 idle) | 100% | 4.45 | 4.52 | 100% |

"Neighbours busy" is the share of the cell each neighbour spent inside a request. It is
higher under partitions for the same bursts because each request takes longer on 6 SMs,
so it does not compare load across the two kinds of sharing.

What it says so far:

- Bursty neighbours under time-slicing spread the latency out: a frame takes 1.6 s to 4 s
  depending on whether it lands in a burst. Half the frames miss 3 s although the
  neighbours are only 38% busy.
- Under partitions, bursty and saturating neighbours give the same 4.4 s: the partition
  isolates tenant 0 completely, at the price of being slow all the time.
- Two busy neighbours make every frame about 2.9x slower (1.55 s to ~4.5 s), close to
  the 3x that fair time-slicing between three tenants would give.
- Equal MPS partitions are no faster on average here (4.45 vs 4.54 s), but the tail is tighter
  (p99 4.52 vs 4.73 s). With 6 of 20 SMs a frame takes about as long as sharing the
  whole GPU three ways. The partition also costs that time when the neighbours are idle;
  time-slicing does not.
- At a 3 s deadline everything misses with 3 busy tenants, under either kind of sharing.

## Why time-slicing is poor

`timeline_bursty.png`: the two bursty cells over time, neighbours' requests above tenant 0's
frames. Under time-slicing the frames that land in a burst miss 3 s and the rest take 1.5 s
(6 of 12 missed); under partitions every frame takes 4.4 s (12 of 12 missed). The two panels
are separate runs, so the bursts do not line up between them.

Under time-slicing a frame takes about alone × (1 + busy neighbours). The GPU takes turns between whoever has work,
so one busy neighbour halves tenant 0's share. Twice the alone time is 3.06 s, so a 3 s
deadline cannot absorb even one busy neighbour. Under equal partitions the frame takes
4.39 to 4.47 s whatever the neighbours do. Why exactly is issue #14.

Caveats: the cells are short (about 80 s), so a bursty neighbour only fires 1 to 5 bursts
and the load differs between cells (one neighbour fired a single burst in both bursty cells).

Not yet: steady neighbours, the full 60-frame cells, 2 and 5 tenants (which split the SMs
exactly), and a 3 s camera rate (T = 3 s), where slow frames would also delay the next one.

## Data and how to rebuild

Raw measurements only, as described in `harness/results.py`: `cells.csv` (what each cell ran),
`requests.csv` (every request of tenant 0 and of both neighbours, raw monotonic timestamps),
`co_tenants.csv`, `power.csv`, `gpu_util.csv` and `manifests/`. Every number above is computed
from them; none is stored.

```bash
uv run python -m harness.collect /data/projects/rohan/runs/space-multitenancy-v1/now-n* \
    -o results/2026-11-19-space-multitenancy-initial        # raw cells -> tables
uv run python -m harness.plot_tenancy results/2026-11-19-space-multitenancy-initial  # figures + table
```
