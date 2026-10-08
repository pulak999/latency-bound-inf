# Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference — Zebin Yang, Qi Wang, Yunhe Wang, Xiurui Guo, Bo Yu, Shaoshan Liu, Jiafeng Xu, Hao Dong, Meng Li (Peking University, TLAIC, AIRS, PrimeBot Research Institute); arXiv preprint v5 (cs.RO, 5 Sep 2026), venue not stated in the text; arXiv 2607.12659; https://arxiv.org/abs/2607.12659

## One-paragraph summary
Jetson-PI targets running pi0 / pi0.5 vision-language-action (VLA) models on low-power onboard GPUs (Jetson Orin, Thor) instead of 450 W desktop GPUs. The authors observe that pi0.5 takes about 1.4 s per inference on Orin (about 0.7 Hz), and that asynchronous inference (predict chunk k+1 while executing chunk k) hides some of this but causes two problems: the observation is stale by the time the chunk is used (perception-execution misalignment), and reaction time stays bounded by inference latency. Their fix has three parts. (1) A small (40M parameter) "future correction module" predicts the VLM's final-layer state at the time inference will finish, conditioned on the actions already committed, so the action expert can predict from the future state. (2) A confidence-gated scheduler skips the expensive VLM call when the module is confident and calls only the action expert. (3) A llama.cpp-based engine that exploits the fixed shapes of VLA inference (CUDA graph reuse, GPU-resident intermediate buffers, unrolled flow-matching steps). On Orin, control frequency rises from 0.70 Hz to 6.06 Hz (8.66x), and on LIBERO the method beats VLASH by 14.8% average success rate. It is mainly a robotics algorithm-plus-systems paper; latency is a given input to be hidden or tolerated, not something analysed for variance.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Hardware: Jetson Orin (32GB and 64GB variants in Table 2, several power modes: 30W and 50W in Table 1), Jetson Thor, RTX A6000, RTX 4090 (Table 1, Table 2, Fig. 1b). Table 2 gives TOPS, memory bandwidth (Orin 204.8 GB/s, Thor 273, A6000 768, 4090 1024) and a roofline "balanced computational intensity".
- Models: pi0 and pi0.5 (SigLIP ViT 27 layers, Gemma 2B LLM 18 layers, Gemma 300M action expert 18 layers, 10 flow-matching denoising steps; Appendix A, Table 5). Real-world uses the XR-1 model (Sec. 5.3). Training of the correction module on A100 in two stages, 30,000 and 25,000 steps (Table 6).
- Workloads: LIBERO simulation benchmark (Spatial, Object, Goal, LIBERO-10) with pi0.5 (Sec. 5.1, Table 3). Real robot: PrimeBot X2-W with three 224x224 cameras, folding clothes split into 3 subtasks (picking/folding/placing in Fig. 7), actions at 15 Hz, 10 trials per cell (Sec. 5.3, Fig. 7).
- Measured: per-component latency (ViT, LLM, action expert, total) per device and power mode (Table 1, Table 4); "reaction time" and control frequency per optimization step (Table 4); success rate (Tables 3, Fig. 7, Fig. 8).
- Measured vs injected: Table 1 and Table 4 are on-device latencies. In the LIBERO simulation, the inference delay Delta (in action steps, 1 to 9) is a swept parameter; the action-expert delay is assumed as ceil(Delta/3) ("a common ratio" on 4090 and Orin; Table 3 caption). Inference in the simulation is therefore not run under real Orin timing; Delta is injected (my inference from the sweep design; the text says success is evaluated "with different Delta"). Baselines are VLASH, RTC, and synchronous (delay 0).
- Table 4 baseline "Naive PI05" Orin row (152.3 / 631.0 / 536.8 / 1420.8 ms) equals the Orin-50W pi0.5 row of Table 1, so Table 4 on Orin appears to be at 50W (my inference; the Table 4 caption does not state the power mode).
- Control frequency in Table 4 equals 1 / reaction time in every row (e.g. 1/1.4208 s = 0.70 Hz; 1/0.1651 s = 6.06 Hz). How "reaction time" is obtained for the scheduled rows (e.g. 674.9 ms) is not described in the text beyond "action prediction no longer requires invoking the VLM every time", so it seems to be derived from component latencies and the invocation ratio rather than a timed end-to-end loop (my inference).

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Result | Value | Source |
|---|---|---|
| pi0.5 latency on Jetson Orin, naive PyTorch | about 1.4 s, control frequency about 0.7 Hz | Sec. 1, Fig. 1b |
| pi0 total latency Orin-30W / Orin-50W / Thor / A6000 / 4090 | 2268.8 / 1403.0 / 447.9 / 116.1 / 69.5 ms | Table 1 |
| pi0.5 total latency Orin-30W / Orin-50W / Thor / A6000 / 4090 | 2289.4 / 1420.8 / 457.9 / 128.2 / 76.4 ms | Table 1 |
| pi0.5 on Orin-50W, breakdown ViT / LLM / action expert | 152.3 / 631.0 / 536.8 ms | Table 1 |
| Battery life, RTX 4090 vs Orin | 4090 reduces battery life by 6.0x (also 4.0x and 2.9x shown for the other robots in the figure) | Sec. 1, Fig. 1a |
| Orin control frequency: naive / +schedule / +graph reuse / +buffer & unroll | 0.70 / 1.48 / 4.41 / 6.06 Hz | Table 4 |
| Orin reaction time, same four rows | 1420.8 / 674.9 / 227.0 / 165.1 ms | Table 4 |
| Orin total latency, same four rows | 1420.8 / 1420.8 / 476.1 / 412.9 ms | Table 4 |
| Thor control frequency, same four rows | 2.18 / 4.08 / 5.22 / 7.59 Hz | Table 4 |
| Thor reaction time, same four rows | 457.9 / 244.7 / 191.4 / 131.8 ms | Table 4 |
| Control frequency gain vs naive PyTorch | 8.66x (Orin), 3.48x (Thor) | Sec. 5.2, Abstract |
| Gain vs vla.cpp (893.0 ms on Orin, taken from [31]) | 5.41x control frequency | Sec. 5.2 |
| Scheduling optimization, reaction time reduction | 2.11x (Orin), 1.87x (Thor) | Sec. 5.2 |
| Graph reuse, reaction time reduction on Orin | 2.96x | Sec. 5.2 |
| Buffering + unroll, action expert speedup | 1.50x (Orin), 1.59x (Thor) | Sec. 5.2 |
| LIBERO success, avg over Delta=1..9, Spatial: VLASH / RTC / Ours / +Sched | 74.4 / 92.6 / 97.0 / 97.4 (Sync 97.3) | Table 3 |
| Same, Object | 86.1 / 96.6 / 98.0 / 98.6 (Sync 99.6) | Table 3 |
| Same, Goal | 84.2 / 94.1 / 96.5 / 96.8 (Sync 96.7) | Table 3 |
| Same, LIBERO-10 | 81.3 / 86.4 / 92.2 / 92.5 (Sync 93.5) | Table 3 |
| Delta=9, Spatial: VLASH / RTC / Ours / +Sched | 30.1 / 88.8 / 97.0 / 97.3 | Table 3 |
| Delta=9, LIBERO-10: VLASH / RTC / Ours / +Sched | 59.6 / 81.0 / 92.0 / 92.2 | Table 3 |
| Average gain over VLASH and RTC across suites and Delta | 14.8% and 3.9% | Sec. 5.1 |
| Gain at Delta=9 over VLASH and RTC | 45.6% and 7.0% | Sec. 5.1 |
| Real robot (success out of 10): picking / folding / placing | RTX 4090 baseline 10 / 7 / 10; Orin naive async 6 / 0 / 5; Jetson-PI 10 / 8 / 9 | Fig. 7 |
| Correction module size | 40M parameters, about 1% of the VLA | Sec. 4.1 |

Note: the 14.8% and 3.9% figures match the "+Sched" column averaged over the four suites minus the VLASH and RTC averages from Table 3 (my arithmetic check; they are absolute percentage points, not relative).

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Yes for robot control on onboard GPUs, framed as latency versus control cadence rather than as deadline misses. A 1.4 s pi0.5 inference on Orin gives about 0.7 Hz and long pauses between chunks under synchronous execution (Sec. 1, Sec. 2). Under asynchronous execution, success falls as the inference delay Delta (in action steps) grows: for example VLASH on LIBERO-Spatial drops from 98.8 at Delta=1 to 30.1 at Delta=9, and RTC from 97.1 to 88.8 (Table 3). On the real robot, naive async on Orin completes folding 0/10 versus 7/10 on a 4090 (Fig. 7). Where the time goes: on Orin-50W, the LLM (631.0 ms) and action expert (536.8 ms) dominate ViT (152.3 ms) (Table 1), and per-inference graph construction and KV copies via CPU memory add avoidable overhead (reuse cut total from 1420.8 to 476.1 ms, Table 4; Sec. 4.3 and 5.2). A chunk-level deadline is implicit (Delta against chunk length L), but no deadline-miss rate is reported.

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Mostly nothing relevant: the paper gives only mean-style latencies and no variance, jitter, or tail measurements. Two partial points. First, the same model's latency changes with power mode (pi0.5 total 2289.4 ms at Orin-30W vs 1420.8 ms at Orin-50W, Table 1), and the authors state that VLA inference latency "varies across different hardware platforms and even under different power modes on the same device" (Sec. 4.1). Second, they argue via a roofline analysis that on Orin both VLM and action expert are bandwidth-bound, so running them in parallel causes "severe contention for the shared bandwidth" and slows both (Sec. 4.2, Table 2); this is an argument, and no parallel-run measurement is shown.

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
The paper never mentions FPGAs. On the GPU-determinism half, there is a modest data point: VLA inference has a nearly fixed token length and fixed intermediate sizes, so the authors pad the instruction to a fixed length, reuse one CUDA graph, preallocate fixed GPU buffers, and unroll the 10 denoising steps into one graph (Sec. 4.3). This improves mean latency and, by construction, removes per-call graph construction, but variability is not measured, so it does not show that determinism was gained (my inference). Nothing is said about what belongs on each device.

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- No latency distributions: no variance, p99, jitter, or deadline-miss counts. Table 1 and Table 4 give single numbers per cell, and the text does not say whether they are means, medians, or single runs.
- The LIBERO study injects Delta as a swept parameter with an assumed action-expert ratio ceil(Delta/3); it does not run the policy under real Orin timing or contention (Table 3 caption). Real-world evidence is small: 10 trials per subtask, one robot, a different model (XR-1), and no latency table for that setup (Fig. 7, Sec. 5.3).
- Reaction time and control frequency in Table 4 are not clearly end-to-end measured; the scheduled rows depend on the VLM-skip rate and the method of computing it is not given in the text (my inference).
- The roofline argument against parallelizing VLM and action expert is analytical; the text does not report a measured parallel run on Orin. The comparison to vla.cpp uses a latency (893.0 ms) quoted from [31], not re-measured here (Sec. 5.2).
- Single-tenant setting: no other workloads share the Jetson, so "shared accelerator" interference is not studied. Power modes appear only as a latency table, and thermal throttling is not mentioned.
- The headline gains partly come from algorithmic skipping of the VLM (confidence gating), which trades accuracy risk for latency, not only from systems work; the success rate effect of the system-only part is not isolated. The authors note their own limitation: onboard platforms remain constrained relative to GPU clusters, and gains may not close the gap as models grow (Limitations section).
- Robot manipulation at roughly 15 Hz is a different regime from satellite or hard-real-time settings; the 8.66x and 5.41x claims are control-frequency numbers for one model family.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "Running them in parallel leads to severe contention for the shared bandwidth, ultimately causing both to slow down significantly." (Sec. 4.2)
- "Unlike LLM decoding, where token length grows progressively, given camera configurations and image resolutions, the token length in VLA inference remains largely constant." (Sec. 4.3)
- "the reaction time in terms of action steps ranges from ∆ to ∆+L" (Sec. 3; Delta is inference latency in action steps, L the number of actions executed per chunk)
- "VLA inference latency varies across different hardware platforms and even under different power modes on the same device." (Sec. 4.1, requirement (3))

## Open questions / things to verify
- Which Orin power mode and which statistic (mean, median, single run) back Table 4; confirm it is Orin-50W as the matching numbers with Table 1 suggest.
- How "reaction time" is computed for the scheduled configurations, and what fraction of steps skip the VLM at the chosen threshold (Fig. 8 and Fig. 9 show the threshold effect, with theta = -0.2 in the Fig. 9 example, but the text gives no skip fraction).
- Whether the code release (github.com/PKU-SEC-Lab/Jetson-PI-Edge) allows per-inference latency traces so we can measure jitter ourselves under the fixed-graph design.
- Check the vla-perf paper [17] and vla.cpp [31] cited here for measured latency distributions, which this paper does not give.
- Check the Delta symbol in the Sec. 3 quote against the PDF, since the quote was taken from the text extraction.
