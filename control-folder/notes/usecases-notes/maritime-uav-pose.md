# Hardware- and Vision-in-the-Loop Validation of Deep Monocular Pose Estimation for Autonomous Maritime UAV Flight — Maneesha Wickramasuriya, Beomyeol Yu, Jaden Shin, Mason Huslig, Taeyoung Lee, Murray Snyder (George Washington University); arXiv preprint cs.RO, v1 dated 17 Jun 2026 (no venue stated in the text); arXiv 2606.19176; https://arxiv.org/abs/2606.19176

## One-paragraph summary
The paper builds a closed-loop flight test for a quadrotor that must estimate its pose relative to a ship from camera images. Instead of flying at sea, the UAV flies indoors in a Vicon motion-capture room while a workstation renders photorealistic "at-sea" views (3D Gaussian Splatting) from the tracked pose and streams them over Wi-Fi to an onboard Jetson Orin NX. A transformer pose model (TNN-MO, ResNet50 + 6-layer encoder/decoder, TensorRT FP16) produces slow, delayed pose measurements, which a delayed Kalman filter (DKF) fuses with 200 Hz IMU data so a geometric controller can fly takeoff, a 7.5 m out-and-back track, and landing. The inference-latency part is a throughput versus delay trade-off: running 1 to 4 staggered model instances in parallel raises the update rate (4.7 to 10.0 Hz with the flight controller on board) but also raises average delay (0.21 to 0.40 s). The authors absorb latency by estimation-side compensation (DKF), not by making inference faster or deterministic. Flight results are good (position estimation MAE 0.066 m, control MAE 0.089 m) with a single reported flight and a worst-case latency of about 0.55 s.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Onboard compute: NVIDIA Jetson Orin NX, running TNN-MO inference, DKF, and the geometric flight controller (multi-threaded C++) together (Sec. II-A, III-B, III-C, Fig. 1).
- Rendering workstation (off-board): Linux machine with an RTX 2080 GPU renders 3DGS views from Vicon pose (Sec. III-A). Sec. II-B separately states real-time rendering at 640 x 480 and 60-110 FPS on an RTX 3060 laptop. The two GPU statements are not reconciled in the text.
- Sensing: Vicon, 12 Valkyrie VK-8 cameras, 200 Hz; VectorNav VN-100 IMU at 200 Hz. The onboard camera is not used indoors (Fig. 1 caption).
- Link: rendered images go over Wi-Fi to the Jetson; Vicon pose goes to the workstation over Ethernet (Sec. III-A).
- Model: TNN-MO, ResNet50 backbone, 6-layer transformer encoder, 6-layer decoder with seven queries (six ship components plus a no-object token), EPnP per object, Bayesian fusion. Input 480 x 640. Deployed as a TensorRT FP16 .engine instead of the PyTorch .pt (Sec. II-C, III-B).
- Workload: image-stream inference, with 1, 2, 3 or 4 parallel TNN-MO instances each fed a time-shifted image stream (Sec. III-B, Fig. 2), with and without the flight controller running on the Jetson (Table I).
- Mission: autonomous takeoff, 7.5 m out-and-back at 1.0 m altitude (virtual ship frame), landing (Sec. IV-A). Plots span roughly t = 30 to 80 s (Figs. 7-9).
- Metrics: estimation error (DKF estimate vs Vicon) and control error (desired vs measured), as MSE/RMSE/MAE for position, velocity, attitude (Table II). Inference rate and average delay per configuration (Table I).
- Latency: measured and injected.
  - Measured: Table I average inference frequency and delay per instance count; the component breakdown in Fig. 2 (motion capture 0.005 s, rendering 0.016 s, Wi-Fi 0.075 s, TNN-MO 0.27 s); communication latency of 0.08-0.1 s (Sec. III-B); the 0.3 s total end-to-end delay (Sec. II-A).
  - Injected / engineered: explicit wait time inserted so the update rate stays constant (Fig. 2, Sec. III-B); target rate 8.5 Hz with synchronization at fixed time steps; for the flight test, "a conservative fixed inference delay of 0.4 s was imposed" with two parallel instances for a 5 Hz update rate, giving about 0.55 s total latency including transmission (Sec. III-B, IV-B-c).
- It is not stated which latency configuration (0.3 s, 8.5 Hz, or the 0.55 s / 5 Hz worst case) produced the Table II flight numbers. Sec. III-C and IV-B-c both point to 5 Hz pose updates (my inference: the Table II run used the 5 Hz / 0.55 s condition).

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Quantity | Value | Source |
|---|---|---|
| Single TNN-MO, Jetson dedicated (no FC) | 5.5 Hz, 0.18 s avg delay | Table I, Sec. III-B |
| Single TNN-MO with FC on board | 4.7 Hz, 0.21 s | Table I, Sec. III-B |
| Two instances w/o FC / w/ FC | 9.0 Hz, 0.22 s / 8.7 Hz, 0.23 s | Table I |
| Three instances w/o FC / w/ FC | 9.8 Hz, 0.30 s / 9.0 Hz, 0.33 s | Table I |
| Four instances w/o FC / w/ FC | 11.1 Hz, 0.36 s / 10.0 Hz, 0.40 s | Table I |
| Highest-throughput config | 4 instances + FC, about 10 Hz, about 0.4 s delay; "nearly saturates" GPU memory and compute | Sec. III-B |
| Communication latency | 0.08-0.1 s | Sec. III-B |
| Component latencies (Fig. 2) | motion cap 0.005 s, render 0.016 s, Wi-Fi 0.075 s, TNN-MO 0.27 s; staggered by 0.117 s, 8.5 Hz output, 0.345 s per frame | Fig. 2 caption and labels |
| Total end-to-end delay (nominal) | 0.3 s | Sec. II-A, Fig. 2 |
| Rate drift | pipeline averaging 8.7 Hz may drift to 8.55 Hz over a two-hour run; target set to 8.5 Hz | Sec. III-B |
| Conservative worst case | 0.4 s inference delay, about 0.55 s total, 5 Hz | Sec. III-B, IV-B-c |
| 3DGS rendering | 640 x 480, 60-110 FPS on RTX 3060 laptop | Sec. II-B |
| Estimation error (DKF vs Vicon) | position MAE 0.066 m (RMSE 0.0907 m); velocity MAE 0.032 m/s (RMSE 0.043); attitude MAE 2.13 deg (RMSE 2.30) | Table II, Sec. IV-B-a |
| Control error (desired vs measured) | position MAE 0.089 m (RMSE 0.1235); velocity MAE 0.062 m/s (RMSE 0.087); attitude MAE 4.00 deg (RMSE 4.17) | Table II, Sec. IV-B-b |
| DKF max supported delay | tau_max = 5.5 s (Sec. III-C) vs 0.55 s (Sec. IV-B-c); text is inconsistent | Sec. III-C, IV-B-c |
| Mission | 7.5 m out-and-back at 1.0 m altitude | Sec. IV-A |

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Partly relevant. Inference on a Jetson Orin NX is slow relative to control rates: one TNN-MO instance gives only 4.7-5.5 Hz at 0.18-0.21 s (Table I), and total perception latency reaches 0.3 s nominal and about 0.55 s worst case. The paper does not talk about deadline misses or deadlines in the frameworks sense (no per-frame deadline, no miss rate, no tail latency); it treats latency as a delay to be compensated. The visible "miss" is in the control loop: vision cannot keep up with the 200 Hz IMU/control rate, so a DKF bridges the gap. Also, adding parallel instances to raise throughput increases delay (0.21 s to 0.40 s with FC), so throughput and latency trade against each other on the same device (Table I).

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Some relevant evidence, mostly qualitative. The authors state that achievable rates depend on "GPU utilization, CPU scheduling, memory bandwidth, and operating conditions such as temperature and concurrent processes" (Sec. III-B). Quantified pieces: running the flight controller alongside inference cuts a single instance from 5.5 to 4.7 Hz and raises delay from 0.18 to 0.21 s; parallel instances "contend for shared Jetson resources, which can lengthen inference time"; the 4-instance setup nearly saturates GPU memory and compute; a pipeline averaging 8.7 Hz drifted to 8.55 Hz over two hours; Jetson performance is "sensitive to ambient temperature"; Wi-Fi latency fluctuated during flight. They mitigate by commanding a rate below peak and inserting wait time. No variance, percentile, or per-source breakdown of the inference-time jitter is given, only averages.

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
Nothing relevant.

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- Only averages are reported for inference rate and delay (Table I). No jitter, tail latency, deadline-miss rate, or distributions, so it cannot support claims about predictability.
- Latency in the flight test was largely set by the authors (fixed 0.4 s inference delay, enforced wait times, target 8.5 Hz or 5 Hz), not purely observed. The flight results therefore reflect an engineered, constant-delay regime.
- Evidence is from one reported flight (one mission profile, indoor, 7.5 m path). No repeated trials, no statistics across runs, no ablation without the DKF, so the claim that the DKF is "essential" (Sec. V) is asserted, not shown in the text I read.
- No comparison of alternatives to reduce latency: smaller models, INT8, other accelerators, DLA, or GPU scheduling/priority control. FP16 TensorRT is the only optimization mentioned and no PyTorch vs TensorRT numbers are given.
- Latency includes rendering and Wi-Fi that exist only because of the vision-in-the-loop emulation; the real-camera deployment will not have the 0.075 s Wi-Fi component or the render time (my inference). Compute-only latency is therefore only the 0.18-0.4 s Table I figures and the 0.27 s in Fig. 2.
- Internal inconsistencies: tau_max 5.5 s (Sec. III-C) vs 0.55 s (Sec. IV-B-c); Fig. 2 shows 0.27 s TNN-MO time and 0.345 s per frame, while Table I averages are 0.18-0.40 s depending on instance count; Fig. 2 shows three instances yet Table I's three-instance rate is 9.0-9.8 Hz, not 8.5 Hz (a commanded rate); two different GPUs for rendering (RTX 2080 workstation vs RTX 3060 laptop).
- Which latency configuration generated Table II is not stated explicitly.
- Accuracy numbers are against Vicon and for a slow trajectory; the estimator inputs come from rendered images of a 3DGS scene, not real sea imagery, with a slightly blurred water region (Sec. II-B).
- Nothing on power, energy, memory-bandwidth profiling, or FPGAs; no comparison across hardware platforms.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "These results demonstrate that achievable update rates are constrained not only by network architecture but also by GPU utilization, CPU scheduling, memory bandwidth, and operating conditions such as temperature and concurrent processes." (Sec. III-B)
- "Table I shows a clear throughput-latency trade-off, where additional instances increase update frequency but also increase average delay." (Sec. III-B)
- "This configuration, however, nearly saturates available GPU memory and compute capacity." (Sec. III-B)
- "Because the performance of the Jetson is sensitive to ambient temperature, additional tests are conducted under intentionally increased delay conditions." (Sec. III-B)

## Open questions / things to verify
- Which configuration (delay, rate) produced Table II, and is the flight a single run? Check the video (https://youtu.be/dfWmVE0wvoE) and the authors' earlier DKF paper (ref. [14], Control Engineering Practice 2024) for delay-sensitivity or ablation results.
- Is tau_max 5.5 s or 0.55 s? Likely a typo in one of the two places (my inference).
- How is the "average delay" in Table I defined (inference only, or including staggering wait and pre-processing)? Fig. 2 suggests staggering adds to delay.
- Is the observed slowdown with FC due to CPU contention only (the text says "competing for CPU resources") or also GPU contention? Not separated in the paper.
- Whether per-inference variance exists in their logs (the authors noticed Wi-Fi latency fluctuation and thermal drift, so likely); could request or reproduce with TensorRT on Orin NX as a Q2 case study.
- Possible use for the thesis: a concrete example of a physical-AI loop where perception runs at 5-10 Hz and the system design is built around latency compensation, not latency reduction (my inference).
