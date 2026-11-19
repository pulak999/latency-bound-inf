# Initial naive results: space use case, 1 vs 3 VLM tenants (2026-11-19)

Quick and small: 12 frames per cell, so each miss rate has a 95% interval of about ±25 points.
Treat this as a first look, not a result.

Setup: Gemma 3 4B QAT Q4_0 on llama-server (our sm_110 build), Jetson Thor at 120 W,
GPU clock not pinned. Every tenant is its own container. The measured tenant gets one
Sentinel-2 tile every 6 s; 6 s is longer than any service time, so frames never queue
and any deadline can be applied to the logs afterwards. The other two tenants run the
same model, back to back (saturating). Config: `configs/space_multitenancy_now.toml`.
Raw cells: `/data/projects/rohan/runs/space-multitenancy-v1/now-*`.

| cell | tenants | sharing | neighbours busy | p50 (s) | p99 (s) | missed at 3 s |
|---|---|---|---|---|---|---|
| alone | 1 | - | - | 1.55 | 1.56 | 0% |
| saturating, time-slicing | 3 | no MPS | 100% | 4.54 | 4.73 | 100% |
| saturating, MPS partitions | 3 | 6 SMs each (2 idle) | 100% | 4.45 | 4.52 | 100% |

What it says so far:

- Two busy neighbours make every frame about 2.9x slower (1.55 s to ~4.5 s), close to
  the 3x that fair time-slicing between three tenants would give.
- Equal MPS partitions are no faster on average here (4.45 vs 4.54 s), but the tail is tighter
  (p99 4.52 vs 4.73 s). With 6 of 20 SMs a frame takes about as long as sharing the
  whole GPU three ways. The partition also costs that time when the neighbours are idle;
  time-slicing does not.
- At a 3 s deadline everything misses with 3 busy tenants, under either kind of sharing.

Not yet: steady and bursty neighbours (bursty, time-slicing is running), the full 60-frame cells,
and a 3 s camera rate (T = 3 s), where slow frames would also delay the next one.
