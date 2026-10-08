# VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference — Jiaming Tang, Yufei Sun, et al. (MIT, NVIDIA, Tsinghua, UC Berkeley, UCSD, Caltech; Song Han senior author), robotics paper (cs.RO, arXiv v2 dated 26 Jul 2026; venue not stated in the text), arXiv 2512.01031, https://arxiv.org/abs/2512.01031

## One-paragraph summary
VLASH targets Vision-Language-Action (VLA) robot policies that run under synchronous inference: the robot waits for the model, then executes an action chunk, so it stalls and its worst-case reaction time is long. Asynchronous inference (execute chunk N while computing chunk N+1) hides model latency, but the new chunk is computed from a stale state and applied later, a "prediction-execution misalignment" that causes jerky or inaccurate control. Prior fixes either add inference-time cost (RTC inpainting, which raises latency) or change the architecture (A2C2). VLASH instead rolls the robot state forward through the already-queued actions to estimate the state at the moment the new chunk starts executing, feeds that state to the VLA, and fine-tunes the VLA with a temporal-offset augmentation so it actually uses the state input. A shared-observation attention mask makes this fine-tuning 3.26x cheaper per step. Result: lower worst-case reaction latency (up to 11.8x vs sync, on an H100 using measured pi0.5 inference latency), accuracy equal to or better than sync and other async baselines, and a robot (pi0.5 on an RTX 5090 laptop) that returns ping-pong serves 11/20 times where sync, naive async and RTC all get 0/20. The paper is about hiding and compensating for inference latency in the control algorithm, not about reducing or explaining latency variability in the inference stack.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Models: pi0.5, GR00T N1.6, SmolVLA (Sec. V); a small 4-layer MLP-Mixer flow policy for Kinetix (Sec. V-A1).
- Baselines: Sync, Naive Async (switch chunks directly, as in SmolVLA), RTC (Real-Time Chunking, inference-time inpainting).
- Simulation: Kinetix (12 tasks, 1,024 rollouts per data point, H=8, simulated delays 0 to 4 steps) and LIBERO (Spatial, Object, Goal, Long; 10 tasks each; K=5; delays 0 to 3 steps; 30K fine-tuning iterations, batch 32).
- Real world: Galaxea R1 Lite (dual-arm, 7-DOF arms) and LeRobot SO-101 (6-DOF arm). Tasks: pick and place, stacking, sorting (20 rollouts per method, up to 3 attempts per rollout), ping-pong (20 trials, one serve each), whack-a-mole (5 rounds of up to 30 s). Real-world manipulation uses K=24 at 30 Hz on a laptop with an RTX 5090, inference delay of 2 steps (Sec. V-B1).
- Reaction-speed measurement (Sec. V-B2, Table II): pi0.5 with 1 image input, K=25 at 50 Hz (execution duration D=500 ms), model inference latency L measured on H100, RTX 5090, RTX 4090, with CUDA Graphs, kernel fusion and torch.compile. Maximum reaction latency is then computed from a formula: D+2L for Sync, 2L for async with back-to-back inference. So L is measured; the reaction latency is derived analytically (my inference: the text gives the formula and does not say it was also measured end to end).
- Latency source in other experiments: Kinetix and LIBERO delays are set as a parameter (0 to 4 and 0 to 3 control steps), i.e. injected/simulated. Real-world delay is stated as "an inference delay of 2 steps"; at deployment VLASH "select[s] the smallest offset delta that is greater than the measured inference latency" (Sec. IV-B).
- Training efficiency (Sec. V-C): pi0.5 on LIBERO, 4x H100, DDP, effective batch 64.

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Result | Value | Source |
|---|---|---|
| pi0.5 inference latency L, H100 / RTX 5090 / RTX 4090 (Sync and VLASH) | 23.2 / 29.4 / 32.3 ms | Table II |
| pi0.5 inference latency L with RTC overhead (same GPUs) | 48.4 / 49.6 / 54.2 ms | Table II |
| Max reaction latency, Sync (H100 / 5090 / 4090) | 546.4 / 558.8 / 564.6 ms | Table II |
| Max reaction latency, RTC | 96.8 / 99.2 / 108.4 ms | Table II |
| Max reaction latency, VLASH | 46.4 / 58.8 / 64.6 ms | Table II |
| VLASH speedup vs Sync | 11.8x / 9.5x / 8.7x | Table II |
| VLASH speedup vs RTC | 2.1x / 1.7x / 1.7x | Table II |
| Execution duration D of a chunk | 500 ms (K=25 at 50 Hz) | Sec. V-B2, Table II |
| Kinetix success at delay 4: VLASH vs Naive Async | 81.7% vs 51.2% (30.5% gain) | Sec. V-A1b |
| LIBERO pi0.5 avg SR: Sync (delay 0) | 96.8% | Table I |
| LIBERO pi0.5 avg SR: VLASH delay 1 / 2 / 3 | 97.2 / 97.1 / 94.6% | Table I |
| LIBERO SmolVLA avg SR: Sync | 79.0% | Table I |
| LIBERO SmolVLA avg SR: VLASH delay 1 / 2 / 3 | 77.1 / 78.1 / 79.1% | Table I |
| Ping-pong single-return hit rate (q=2): Sync, Naive Async, RTC, VLASH | 0/20, 0/20, 0/20, 11/20 (55%) | Table III |
| Whack-a-mole avg score (q=2): Sync, Naive Async, RTC, VLASH | 3.2, 8.6, 11.0, 28.8 | Table III |
| Task-completion speedup from action quantization at q in [1.5, 2], no accuracy loss | 1.5 to 2.0x | Sec. V-B1b, abstract |
| Fine-tuning time per step: Original vs VLASH | 420.99 vs 129.29 ms (3.26x) | Table IV |
| LIBERO avg after 10K / 20K / 30K steps: Original | 94.1 / 97.1 / 96.8% | Table IV |
| LIBERO avg after 10K / 20K / 30K steps: VLASH | 87.1 / 94.4 / 96.6% | Table IV |
| Token length with 5 offsets packed (pi0.5) | ~650 to ~850 tokens (+~31%) | Sec. IV-C |
| Demo inference frequency, pi0.5 on RTX 5090 laptop | 15 Hz | Appendix II |

The real-world success rates for pick&place, stacking and sorting appear only in Figs. 7 and 8 (bar charts); the text gives no numbers for them beyond qualitative claims (VLASH highest in all tasks; q=2 matches or exceeds Sync at q=1 on pick&place and sorting).

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Partially, in a control sense. The paper shows that in a robot loop, synchronous VLA inference creates action stalls and a worst-case reaction latency of D+2L (about 547 to 565 ms for pi0.5 on the three GPUs, Table II), and that sync, naive async and even RTC (whose inpainting roughly doubles L, 23.2 to 48.4 ms on H100) get 0/20 on ping-pong at q=2 (Table III). So for a physical-AI task with a reaction deadline, plain inference latency does cause task failure. However, the paper never frames this as missed deadlines of an inference framework: it treats L as a fixed per-GPU number and does not report latency distributions, tail latency, or any deadline-miss rate. It also does not locate where in the stack time goes. It only notes that RTC's added computation raises L.

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Nothing relevant. (The paper reports a single latency L per GPU and treats it as given. It mentions no jitter, no contention, no shared accelerator. It does say deployment picks "the smallest offset delta that is greater than the measured inference latency", which implicitly assumes L is stable enough to bound, my inference.)

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
Nothing relevant. (No FPGA content. The only determinism-flavored item is that CUDA Graphs, kernel fusion and torch.compile were used to get minimal L, Sec. V-B2, but the paper does not discuss determinism.)

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- Latency is reported as a single number per GPU (Table II); no repetitions, percentiles, variance or tail. Nothing is said about how many runs L comes from.
- The headline 11.8x is a computed worst-case (D+2L vs 2L) using the fastest GPU (H100), not a measured end-to-end reaction time. Most of the speedup comes from removing the chunk execution time D=500 ms from the sync worst case, not from lowering L (VLASH and Sync share the same L). It is a scheduling gain, not an inference-speed gain.
- Async only works if L < chunk execution duration D; the paper assumes this ("the execution duration of an action chunk is typically longer than the model inference time") and does not study what happens when L spikes past D or the delay exceeds Delta_max.
- Delays in Kinetix and LIBERO are injected as integer control steps (0 to 4, 0 to 3), not produced by a loaded system. Real-world runs use a fixed delay of 2 steps on a dedicated laptop RTX 5090; there is no co-located workload or shared-GPU scenario.
- Inconsistent latency figures across the text: Table II gives 29.4 ms for pi0.5 on the 5090 (1 image, K=25), while Appendix II says the demo runs at 15 Hz inference frequency (about 67 ms per inference, my arithmetic) and Sec. V-B1 uses a 2-step delay at 30 Hz (about 67 ms). Different configs (images, whole-pipeline vs model-only) probably explain it, but the paper does not say.
- Mitigation is algorithmic (state rollforward plus fine-tuning); it requires fine-tuning the VLA and assumes the future environment can be ignored while the future robot state is predictable from known actions. Open-loop state estimation can drift if the environment is stochastic (Kinetix rollforward noise is sampled independently, so the estimate is "approximate", Sec. V-A1).
- Small real-world samples: 20 rollouts per method, 20 ping-pong trials, 5 whack-a-mole rounds. Ping-pong and whack-a-mole use q=2 for all methods, chosen as best across baselines. The text gives no per-method error bars for Table III.
- LIBERO: delay 3 drops pi0.5 to 94.6% vs 96.8% Sync, and the authors note LIBERO is slowly changing so async methods behave similarly (Sec. V-A2a).
- Fine-tuning ablation (Table IV) shows slower early convergence (87.1% vs 94.1% at 10K).
- Reaction-speed GPUs are all NVIDIA desktop/datacenter parts; no edge, FPGA or satellite hardware.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "asynchronous inference with back-to-back inference captures the change at the next inference start and reacts when that inference completes, giving 2L." (Sec. V-B2a)
- "RTC's inference-time overhead increases the inference latency and widens the prediction-execution gap, causing it to operate under greater state staleness than ours." (Sec. V-B1b)
- "the same VLA may be deployed on hardware with different compute budgets, leading to different inference delays" (Sec. IV-B)
- "we simply select the smallest offset delta that is greater than the measured inference latency" (Sec. IV-B)

## Open questions / things to verify
- Where do the L numbers in Table II come from: model forward pass only, or including preprocessing, image transfer and sampling steps? Check the code at https://github.com/mit-han-lab/vlash.
- How many measurement runs per L, and what is the variance? Would the 2L and D+2L bounds hold at p99 on a shared GPU? (Our question; the paper does not answer.)
- What happens when real latency exceeds Delta_max or the chunk duration D (fallback behavior)? Not described.
- Resolve the 29.4 ms (Table II) vs 15 Hz (Appendix II) inconsistency for pi0.5 on the RTX 5090.
- Read the real-world success rates off Figs. 7 and 8 in the PDF if needed; they are not in the extracted text.
- Check the related work for usable latency baselines for our physical-AI use case: RTC (arXiv 2506.07339), training-time RTC (2512.05964), A2C2 (2509.23224), "Running VLAs at real-time speed" (2510.26742), as cited in the reference list.
