# Gate 2: real model, no cadence (Thor, 2026-11-04) — one-token rule and accuracy pass, latency spread fails

Engine: our own llama.cpp build (`/data/projects/rohan/src/llama.cpp`, sm_110), `llama-server`
single slot, Gemma 3 4B QAT Q4_0 + f16 mmproj (symlinked from the colleague's download).
Prompt (DESIGN.md's proposal, in `configs/stack_a.toml`): "Decide whether to keep or drop this
satellite image for downlink. Drop it if it is mostly cloud or open water. Answer with exactly one
word: keep or drop." `max_tokens = 1`, temperature 0. 50 Sentinel-2 tiles (896x896, six regions,
cut by `setup/tiles_sentinel2.py`), sent back to back by `harness.probe`. Per-request log:
`gate2-probe-thor.jsonl`.

| DESIGN.md gate 2 criterion | Result | Pass |
|---|---|---|
| Every answer is exactly one of the two words | 50/50 are `keep` or `drop`, verbatim, one token | yes |
| Agreement with SCL cloud fraction above 80% | 48/50 = 96% (truth: drop if SCL cloud or water > 50%; 21 drops) | yes |
| Latency of requests 11 to 50 within a factor of two | 1448 to 3143 ms, ratio 2.17 | no |

Where the spread comes from: 40 requests took 1555 ms (median; prefill 1383 ms, 301 prompt
tokens, 174 ms outside the engine) and 9 took 2.3 to 3.1 s. In the slow ones the engine's own
`prompt_ms` rose to 2.1 to 2.9 s while the outside-engine share stayed at 175 ms, so the extra
time is on the GPU. During the probe three labmate processes held about 85 GB of GPU memory
and one of them (`run_formal_robotwin.py`) drove utilisation to 97 to 98% for about a second
every 14 s; nine collisions in a three-minute probe matches that cadence. The GPU clock governor
(`nvhost_podgov`) is dynamic, 765 MHz when sampled against a 1575 MHz maximum, which the cells'
sampler will record per request.

Verdict: the prompt and the one-token stopping rule work as proposed and need no change.
The factor-of-two check must be repeated with the GPU idle; as run it measured the
co-tenant, which is the step 4 experiment, not step 2.

Earlier probes on the first ten tiles (all Iberia land, cloud at most 41%) gave `keep` 10/10,
and two other prompt shapes (`yes`/`no` for "mostly cloud or water", and `cloud`/`water`/`land`)
also answered with exactly one configured token every time, so the model is not sensitive to the
exact phrasing on this point.

Decisions for a human: (1) adopt this prompt; (2) when the GPU can be idle for the alone cells
(two to three hours for T = 20, 5, 2, 1.5 s with the config's frame counts).
