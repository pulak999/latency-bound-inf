# Cloud, Edge, or Split? Profiling Onboard and Split Vision-Language Model Deployment for Drone AI — Zoha Azimi, Reza Farahani, Schahram Dustdar, Christian Timmerer; arXiv preprint (cs.DC, v1 dated 21 Sep 2026; no venue stated in the text), arXiv 2609.25415, https://arxiv.org/abs/2609.25415

Affiliations: CD Lab ATHENA / University of Klagenfurt (Azimi, Timmerer); DSG, TU Wien (Farahani, Dustdar).

## One-paragraph summary
The paper benchmarks three ways of running a small vision-language model (SmolVLM-256M) for drone inspection tasks: fully onboard on a Jetson, fully in the cloud on an RTX A6000, and split (vision encoder plus projection onboard, language decoder in the cloud, INT8 visual tokens sent uplink). Using power-line and bridge-defect images, 18 inspection prompts and a 4G/LTE throughput trace cut into low, moderate and high bandwidth windows, it reports compute time, utilization, GPU memory, energy and transfer latency per strategy. Headline finding: no strategy wins everywhere. Onboard is fastest end to end at low bandwidth (2.77 s), cloud is fastest at moderate and high bandwidth (0.83 s and 0.77 s), and split never wins on latency but cuts onboard GPU memory (about 59 % allocated), onboard energy (about 42 %) and uplink payload (about 54 % vs JPEG). The authors propose an adaptive runtime orchestrator as future work. The evidence is about placement under network variability; it contains no deadline analysis, no latency distributions, and no contention measurements.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Model: SmolVLM-256M (SigLIP vision encoder, pixel-shuffle plus MLP projection, SmolLM2-135M decoder). Default preprocessing held fixed (longest edge 2048, 512x512 tiles plus a thumbnail), so tile count T = 4*ceil(4r)+1 depends on aspect ratio, not pixel count (Sec. II, IV-A).
- Split point: after the projection module. Tensor [T, 64, 576] is sent instead of [T, 1024, 768], about 21x fewer values (Sec. II). Sent as INT8-quantized tokens (Sec. V-B1).
- Edge: "an NVIDIA Jetson" with 8-core Cortex-A78AE and Ampere GPU with 1024 CUDA cores and 32 Tensor Cores. Exact module, power mode and software stack are not named in the text (Sec. IV-D).
- Cloud: Ubuntu 22.04, Intel Xeon Gold 5218, NVIDIA RTX A6000 (Sec. IV-D).
- Workloads: 70 frames from a simulated UAV power-line video dataset and 150 CODEBRIM bridge-defect images (Sec. IV-E); 18 prompts in four categories (component enumeration, context awareness, defect detection, localization); same image-prompt pairs for every strategy (Sec. IV-F).
- Network: a 500 s 4G/LTE HTTP throughput trace, three windows: low (220-250 s, mean 2.1 Mbps, min 0, max 10.86), moderate (50-80 s, mean 43.5, min 26.7, max 61.5), high (380-480 s, mean 65, min 34.6, max 103.03) (Sec. IV-G, Table I).
- Measured: Jetson CPU/GPU utilization, memory and module power via tegrastats at 100 ms; per-request energy by integrating power over the inference interval. Cloud CPU/DRAM energy from RAPL, GPU energy by integrating PyNVML power samples (Sec. IV-D).
- Latency definition: "inference latency" excludes network transfer; communication is reported separately as payload size and transfer latency (Sec. IV-H).
- Measured vs injected: compute times are measured on real hardware. The network is "emulated" from the trace (Sec. IV-G). The text does not say how transfer latency is computed (real shaped link vs payload divided by trace throughput). Mean transfer times are consistent with payload / mean throughput (my inference, e.g. 821.45 kB at 2.1 Mbps is about 3.1 s vs the reported 3.2 s). End-to-end latency (Table III) is described as onboard compute plus transfer plus remote compute (Sec. V-B2).
- Statistics: results are "mean, and the standard deviation across distinct image-prompt pairs" (Sec. IV-H), but no standard deviations or percentiles appear anywhere in the extracted text.

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Quantity | Value | Source |
|---|---|---|
| Compute latency, fully onboard | about 2.76 s | Sec. V-A1, Fig. 6 |
| Compute latency, fully cloud | about 0.72 s (3.8x faster than onboard) | Sec. V-A1, Fig. 6 |
| Split: onboard encoder+projection / cloud decoder / aggregate | about 1.41 s / 0.68 s / 2.09 s (24 % below onboard) | Sec. V-A1, Fig. 6 |
| Onboard utilization (CPU / GPU / RAM) | about 11 % / 55 % / 31 % | Sec. V-A2, Fig. 7 |
| Cloud utilization (CPU / GPU / RAM) | about 2 % / 15 % / 2 % | Sec. V-A2, Fig. 7 |
| Split onboard stage: mean GPU / RAM | about 69 % / 22 % | Sec. V-A2, Fig. 7 |
| Onboard GPU memory, allocated / peak | full: 572 MB / 854 MB; split edge: 236 MB / 472 MB (about 59 % and 45 % lower) | Sec. V-A2, Fig. 8 |
| UAV energy per request | full onboard about 26 J; split onboard stage about 15 J (about 42 % lower) | Sec. V-A3, Fig. 9 |
| Cloud energy per request | full cloud about 150 J; split decoder stage about 130 J | Sec. V-A3, Fig. 9 |
| Mean uplink payload | full cloud 821.45 kB (JPEG); split 375.84 kB (INT8 tokens); onboard about 10 kB (text); split about 54 % smaller than JPEG | Sec. V-B1, Table II |
| Transfer latency, low / moderate / high BW | onboard 0.04 / 0.002 / 0.001 s; cloud 3.2 / 0.155 / 0.104 s; split 1.47 / 0.071 / 0.047 s | Table II |
| End-to-end latency, low BW | onboard 2.77 s; split 3.59 s; cloud 3.87 s | Table III |
| End-to-end latency, moderate BW | cloud 0.83 s; split 2.19 s; onboard 2.73 s | Table III |
| End-to-end latency, high BW | cloud 0.77 s; split 2.17 s; onboard 2.73 s | Table III |
| Visual-token payload per image (formula) | BF16 73.7T kB, INT8 36.9T kB; T in {5, 9, 13, 17}; BF16 about 370/660/960/1250 kB, INT8 half of that | Sec. V-C |
| JPEG vs tokens at extremes | JPEG can be smaller below 0.2 MP; JPEG exceeds 20 MB for largest images while tokens stay at or below 1.2 MB (BF16) / 0.6 MB (INT8) | Sec. V-C, Fig. 10 |

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Partial, indirect answer. The paper shows that end-to-end latency for a small VLM varies by roughly 5x across placements and bandwidth (0.77 s to 3.87 s, Table III), and that the best placement flips with bandwidth. The onboard Jetson takes about 2.76 s per request, so a tight deadline would be missed on the edge, while cloud misses it at low bandwidth (3.87 s). However, the paper never defines a deadline, never counts misses, and does not examine any inference framework. Where latency goes is shown at the stage level: onboard vision pipeline is the dominant part of split latency (1.41 s of 2.09 s, Sec. V-A1), and image upload dominates cloud latency at low bandwidth (3.2 s of 3.87 s).

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Only weakly relevant, and not about shared accelerators. Variability sources identified are the network (the low-bandwidth window has min 0 Mbps, Table I) and the input (JPEG size varies with resolution and compressibility; token payload takes discrete values via tile count, Sec. V-C). The paper notes that on a UAV "GPU memory is shared with other perception, navigation, and control workloads" (Sec. V-A2) as motivation for memory savings, but it does not measure any contention or co-running workloads. Compute-latency spread is not reported.

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
Nothing relevant.

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- No deadlines, no tail latencies, no standard deviations or percentiles in the extracted text, despite Sec. IV-H saying std is reported. Only means. This limits its use as evidence about deadline misses or predictability.
- Single model (256M parameters), single Jetson (module and power mode unspecified), single cloud GPU, one network trace (4G/LTE, 2016 dataset). Whether results transfer to larger VLMs or to satellite or other links is unknown.
- Runs appear to be isolated, one request at a time, with no co-located workloads, so nothing on shared-accelerator interference (my inference from the absence of any such setup).
- Network is emulated, and the method for turning the trace into per-request transfer latency is not described. Cloud-side and uplink-queueing effects are not discussed.
- End-to-end numbers do not exactly equal the sum of compute plus transfer from Figs. 6 and Table II (my inference, e.g. onboard 2.76 + 0.04 = 2.80 vs 2.77 in Table III; cloud 0.72 + 3.2 = 3.92 vs 3.87). Small differences, possibly from per-condition averaging, but unexplained.
- Cloud energy figures (about 150 J and 130 J) cover a high-power CPU-GPU server; the text does not say whether idle power is subtracted. Energy is also not comparable one to one with UAV battery energy.
- Accuracy and response quality are not evaluated (the authors list this as future work). Generated token counts are not reported, although decode latency depends on them (Sec. II).
- Sensible, deployment-oriented guidelines are given, but they are qualitative and derived from three bandwidth buckets.
- Derived observation (my inference): split decoder on A6000 takes 0.68 s while the full cloud pipeline takes 0.72 s, so the cloud encoder costs only about 0.04 s; if times are additive on the Jetson, its decoder is about 1.35 s (2.76 - 1.41), i.e. about 2x slower than cloud rather than the roughly 35x gap for the encoder. This suggests autoregressive decode benefits little from a bigger GPU, but the paper does not state it and the cloud GPU is only 15 % utilized.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "The latency-optimal configuration, thus, shifts from fully onboard inference under constrained bandwidth to fully cloud-based inference when sufficient bandwidth is available." (Sec. V-B2)
- "Split inference does not minimize end-to-end latency under the evaluated conditions; instead, its principal benefit is balancing lower onboard memory and energy consumption against lower communication overhead than fully cloud-based inference." (Sec. V-B2)
- "The preferred VLM inference deployment strategy is determined by the end-to-end critical path, not by compute or communication latency alone." (Sec. V-D)
- "These reductions are important for UAV platforms, where GPU memory is shared with other perception, navigation, and control workloads." (Sec. V-A2)

## Open questions / things to verify
- Which Jetson module and power mode was used? (Not stated; needed to interpret 2.76 s and 26 J.)
- Does the PDF contain per-request standard deviations or distributions in the figure error bars (Figs. 6-9)? The text extraction gives none; check the original PDF.
- How exactly is transfer latency obtained from the trace (real shaping vs arithmetic)? And how is the low-window 0 Mbps outage handled?
- Why do Table III end-to-end values differ slightly from compute plus transfer sums?
- Is the cloud decoder time (0.68 s) dominated by generated-token count? Output length and decoding settings are not reported.
- Related works cited that may matter for us: AVERY adaptive VLM split computing (arXiv 2511.18151), Zhan et al. on edge VLM energy (arXiv 2607.09520), EdgeVLM (arXiv 2508.12638). Not read here.
