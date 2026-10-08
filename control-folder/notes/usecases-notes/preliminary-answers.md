# Preliminary answers to the core questions, from eight papers and two secondary sources (2026-10-06)

Per-paper notes sit next to this file. Numbers below were spot-checked against the paper text.
Papers: VLASH, Jetson-PI, maritime UAV pose, drone VLM split, DNN time variation (driving), hls4ml, Earth+, NAVI-Orbital.
Secondary sources (news and vendor): the IEEE Spectrum article on Loft YAM-9 and the VOLLO press release with its STAC summary page.
The full STAC audit report is gated behind registration and was not read.

## Q1. Is this a problem? Do inference frameworks miss deadlines, and where?

**Short answer: probably yes, but no paper measures a deadline-miss rate.** The evidence is
latency against cadence, mostly means, and often with latency injected rather than measured.

| Evidence | Source |
|---|---|
| Sync pi0.5 worst-case reaction latency about 547 to 565 ms on H100/5090/4090; async cuts it to 2L | VLASH, Table II |
| pi0.5 about 1.4 s on Orin (0.7 Hz); 2289 ms at 30 W vs 1421 ms at 50 W | Jetson-PI, Sec. 1, Table 1 |
| Naive async on Orin folds 0/10 vs 7/10 on a 4090 | Jetson-PI, Fig. 7 |
| 4 of 7 perception DNNs have a latency range above 100 ms; LaneNet 82 ms mean, 282 ms range, 340 ms worst case, under 160 ms 95% of the time | Driving study, Table I, Sec. II |
| Single pose model 5.5 Hz alone, 4.7 Hz with the flight controller on the same board | Maritime UAV, Table I |
| SmolVLM-256M end to end 0.77 s to 3.87 s depending on placement and bandwidth | Drone VLM, Table III |
| Gemma 3 4B on the flight-class Jetson-based engineering model: 26.6 s average VLM cycle per image (5 images), against a "20-second inter-capture cadence" claimed for flight | NAVI-Orbital, Sec. VI-B and Conclusion |

**Where they fail, per this evidence:** model-to-device (Orin is 5 to 30 times slower than a desktop GPU for the same VLA),
and the system around the model (middleware, CPU, co-running load). Not inside the model.
**Gap that is the thesis:** nobody reports deadline-miss counts under a real, measured latency distribution.
VLASH and Jetson-PI inject delay as a fixed integer. For space, the best case is NAVI-Orbital, the only paper that flew an inference workload. It reports no in-orbit latency, power or thermal telemetry.
Its ground cycle time of 26.6 s per image sits above the 20 s capture cadence it claims for flight, and the paper never reconciles the two
(different runs, so this is a flag to check, not a measured miss). Its retry loop (up to ten attempts, 30 s sleep, for "transient hardware and scheduling anomalies")
means per-image latency is unbounded by design. Only two live captures flew.

## Q2. Why is it hard to predict?

**Short answer: the only paper that dissected variance found it mostly outside the GPU.**
The driving study (6 perspectives, 6 insights) attributes variation to input-dependent
post-processing (correlation 0.9 or higher with latency), ROS middleware, CPU scheduling and co-running DNNs.
Per-module tails reach about 4 s and fused tails over 10 s. Its GPU workstation showed the lowest variation for the one-stage model.
It never isolates GPU or accelerator contention and defers multi-tenant analysis to future work.

Contention evidence is thin: the maritime 5.5 to 4.7 Hz drop (averages only), the Jetson-PI power-mode swing,
and a roofline argument that VLM and action expert contend for bandwidth (no parallel-run measurement).

**Implication for the Asaf debate.** The "closed-source GPU scheduler causes the variance" guess is not
supported by this literature. What is supported is CPU, middleware and input-dependent work.
Measuring GPU-level contention directly is therefore a real gap, but the first measurement should be a breakdown,
not an assumption. Caveat: the driving study's Insight 4 and Table IV have internal inconsistencies flagged in its note.

## Q3. What can we learn from FPGAs?

**Short answer: determinism by construction, with a hard size ceiling, and no one has compared it to a GPU.**
- hls4ml: latency is a closed-form cycle count (about 15 cycles, 75 ns at 200 MHz). Fully pipelined, initiation interval equal to
  the reuse factor, about 16-bit fixed point. Pruning cut one MLP from 4389 to 1338 parameters and DSP use from 3329 to 954 at the same latency.
- Ceiling: weights must live in on-chip logic; block RAM breaks the sub-microsecond budget. The models are thousands of parameters,
  and VLAs are billions. The tipping point is a gap of orders of magnitude, not a threshold to find.
- A third-party-audited data point at the other end: VOLLO on an AMD Versal VP1802 reports sub-2 us, sub-3 us and sub-8 us p99 for LSTM_A, B and C
  on STAC-ML Markets. That is a tail number on a real card, but vendor-sponsored, LSTM-sized, and the full report is gated.
- Caveat: those latencies are HLS synthesis estimates, not board measurements. Place and route needed an 8 ns clock for a 5 ns target.

**What the GPU side shows:** Jetson-PI exploits fixed shapes (CUDA graph reuse, preallocated buffers, unrolled denoising) and cuts one Orin
configuration from 1420.8 to 476.1 ms. That is a mean reduction. No jitter or tail result is reported, so "GPU can borrow determinism" is untested.
**Space angle:** YAM-9 carries FPGAs alongside the Jetson-class GPU (IEEE Spectrum), but NAVI-Orbital runs on the GPU only and says nothing about what the FPGAs do.
Its "deterministic" refers to graph transitions and label parsing, not timing.
**Split inference:** the drone VLM paper splits encoder and decoder across devices and finds no strategy universally best (depends on bandwidth and resolution), but uses no FPGA.

## What this changes for the thesis

1. Q1 needs to be stated as a measurement the literature lacks: deadline misses under measured, not injected, latency.
2. Q2 should start with a breakdown (CPU, middleware, input-dependent work, GPU contention) before blaming the GPU.
3. Q3 has a clean first experiment: run the same small model with and without graph capture and static scheduling on the Orin, and report the tail, not the mean.
4. No evidence for or against the GPU-FPGA split. It stays a hypothesis.

## Corrections made along the way

- The slide originally linked the wrong paper for Earth+ (arXiv 2305.10119 is an Aalborg change-detection paper). Correct: arXiv 2403.11434.
- The "over 90% of images are never downloaded" figure is not in the Earth+ paper. It says only 2% of image data can be downloaded (citing Kodan) and under 12% in an idealized Doves estimate.
  The satellite-measurement-v1 notes repeat the 90% figure and should be corrected or sourced elsewhere.
- Slide links updated: the Loft row now cites the NAVI-Orbital paper (arXiv 2606.18271) and the VOLLO row cites the STAC summary page, so neither rests on news alone.
- NAVI-Orbital has small internal inconsistencies flagged in its note (valid-image count, a 13 minute vs 26.6 s x 5 gap, quantization method).
