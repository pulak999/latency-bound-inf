# Understanding Time Variations of DNN Inference in Autonomous Driving — Liangkai Liu (Wayne State), Yanzhi Wang (Northeastern), Weisong Shi (Univ. of Delaware); arXiv preprint (cs.LG, v1, 12 Sep 2022; no venue stated in the text); arXiv 2209.05487; https://arxiv.org/abs/2209.05487

## One-paragraph summary
The paper is a measurement study of why DNN inference time varies on autonomous-driving hardware, aimed at real-time schedulers that set deadlines from the worst observed execution time. It first shows (Table I) that variation is large across eleven models/algorithms: four of seven DNN perception models have a max-min range above 100 ms, and LaneNet reaches 282 ms. It then profiles variation from six angles (data, I/O, model, runtime, hardware, end-to-end system) and states six insights. Mostly on Jetson AGX Xavier with fixed clocks, it finds that variation comes largely from CPU-side work (pre/post-processing whose cost depends on the number of detected objects or lane proposals), from ROS middleware and message synchronization, from CPU scheduling policy and co-running tasks, and from CPU quality. The GPU inference stage dominates variation only for one-stage detectors (YOLOv3, SSD). In the full ROS pipeline, tail delays reach about 4000 ms per module and over 10000 ms for the fused output. All latencies are measured, none injected, and the paper never evaluates a fix. It does not mention FPGAs, deadline-miss rates, or GPU sharing mechanisms.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
Hardware (Sec III-A, Table II):
- NVIDIA Jetson AGX Xavier: 8-core Carmel Arm v8.2 CPU, 512-core Volta GPU with 64 Tensor Cores, 32 GB LPDDR4x, 136.5 GB/s, 32 TOPs.
- Jetson Xavier NX: 6-core Carmel CPU, 384-core Volta GPU with 48 Tensor Cores, 8 GB LPDDR4x, 59.7 GB/s, 21 TOPs.
- "Fog Node" (Intel Fog Reference): 8 x Intel Xeon E3-1275 v5 @ 3.6 GHz, 32 GB DDR4, 34.1 GB/s; the GPU column is "-" (so inference there presumably runs on CPU; my inference).
- GPU workstation: Intel Core i9-9940X, 4 x GeForce RTX 2080 Ti, 64 GB DDR4, 85 GB/s, 312 TOPS.
- Jetson software: JetPack 4.4-DP (L4T R32.4.2), l4t-ml docker image (TensorFlow 1.15, PyTorch 1.5.0, CUDA 10.2, cuDNN 8.0.0, OpenCV 4.1), ROS Melodic and ROS Galactic. The RT-kernel experiments use an RT-patched kernel on the AGX.
- To limit hardware effects, the data, I/O, model and runtime studies were run on Jetson AGX Xavier. DVFS was disabled (jetson_benchmark), MAXN power mode (max CPU and GPU clocks), other user applications turned off, FP32 for all models.

Models: object detection YOLOv3, Faster R-CNN ResNet101, Mask R-CNN Inceptionv2, SSD MobileNetv2 (SSD Inceptionv2 appears only in the Fig 5 correlation table); lane detection LaneNet and PINet; segmentation Deeplabv3 with MobileNetv2; localization AMCL and ORB-SLAM2; planning A* and DWA (run offline in ROS Navigation on recorded sensor data).

Workloads: an image set built from KITTI, three scenarios (city, residential, road), 1,800 images, subsampled at 1/2/5/10 FPS to 60/120/300/600 images per scenario; also a 1920x1080, 30 FPS auto-grade camera. Rain was rendered onto KITTI at six levels. The Table I hardware/input setup is only described as "the same input"; the device is not stated there.

Measured: end-to-end latency split into read, pre-processing, inference, post-processing; range (max-min) and coefficient of variation (cv = sigma/mu); CPU/GPU/memory utilization; correlation coefficients; Linux perf events (cycles, context switches, CPU migrations, cache misses, etc.); nvprof for GPU activity; cProfile for code-level call graphs.

How latency is obtained: all of it is measured on real hardware. Nothing is injected, except synthetic perturbations of the input (all-0, all-255, random pixels; image scaling; rendered rain) and the competing DNN process in the "compete test". Deadlines in the scheduling study are set as SCHED_DEADLINE parameters, not injected latency.

Runtime study (Sec III-E): PINet and YOLOv3 under SCHED_OTHER, SCHED_FIFO, SCHED_RR (priority 99) and SCHED_DEADLINE with two deadlines (Deadline-1 = worst observed time: 300 ms PINet, 225 ms YOLOv3; Deadline-2 = average time: 150 ms PINet, 200 ms YOLOv3). "Single test" has no competition; "compete test" adds another DNN (default SCHED_OTHER).

End-to-end system (Sec IV-V): ROS pipeline on Jetson AGX with RT kernel; /image node feeds three perception nodes (ORB-SLAM2, YOLOv3, Deeplabv3) and a /fusion node using an approximate-time synchronizer (queue 100, 100 ms slop). Planning is excluded as "rule-based with stable execution time".

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Result | Value | Source |
|---|---|---|
| Real-time budget cited for 40 km/h urban driving | task execution under 100 ms, effective every 1 m (citing [27]) | Sec I |
| DNN models with range > 100 ms | 4 of 7 (Faster R-CNN 128, Mask R-CNN 104, LaneNet 282, PINet 263 ms) | Sec II-B, Table I |
| Table I mean / range / range-over-mean | YOLOv3 173/57/32.8%; Faster R-CNN 413/128/31%; Mask R-CNN 266/104/39.1%; SSD MobileNetv2 144/70/48.6%; LaneNet 82/282/344%; PINet 127/263/207.1%; Deeplabv3 149/19/12.8%; AMCL 1.3/8.7/675.5%; ORB-SLAM2 53/56/105.6%; A* 79/97/122.3%; DWA 23/73/323.8% (ms, ms, %) | Table I |
| Static-deadline waste, LaneNet | worst observed 340 ms; under 160 ms over 95% of the time; about 180 ms wasted per job; narrowing range to under 50 ms would save about 110 ms | Sec II-B |
| Insight 1: post-processing time vs number of detected objects (correlation) | Faster R-CNN 0.98, Mask R-CNN 0.98, YOLOv3 0.43, SSD MobileNetv2 0.91, SSD Inceptionv2 0.96 | Sec III-B, Fig 5 |
| Insight 1: pixel size | CDFs similar for scale factors 0.01 to 5; scale 10 is higher mean and variation, attributed to pre-processing transpose/crop | Sec III-B, Fig 6 |
| Insight 1: rain (0 to 200 mm/h) | Faster R-CNN mean 320.7 to 309.1 ms, cv 3.8 to 2.6; PINet mean 228.9 to 190.4 ms, cv 34.4 to 26.6; reported sigma values are 1217.9 to 805.5 (Faster R-CNN) and 7862.6 to 5068.1 (PINet) | Sec III-B, Table IV |
| Insight 2: I/O | range of message latency grows with 1 to 8 subscribers; ROS2 DDS better for 62 KB msg, ROS1 IPC better for 6.2 MB msgs; with msg2/msg3 DDS, four of eight subscribers had much higher latency | Sec III-C, Table V, Fig 9 |
| Insight 3: correlation of end-to-end latency with stage latency | YOLOv3: inference 0.906, post 0.378; SSD MobileNetv2: inference 0.963, post 0.525; Faster R-CNN: inference 0.681, post 0.896; Mask R-CNN: inference 0.619, post 0.946 (read and pre-processing all below 0.43) | Sec III-D, Table VI |
| Insight 3: proposals vs post-processing time | correlation 0.898 (Faster R-CNN), 0.960 (PINet), 0.901 (LaneNet) | Sec III-D, Fig 11 |
| Insight 4: cv under RT kernel (single / compete) | PINet: Deadline-1 0.30/0.33, Deadline-2 0.44/0.44, FIFO 0.33/0.31, OTHER 0.32/0.30, RR 0.33/0.31. YOLOv3: Deadline-1 0.13/0.21, Deadline-2 0.27/0.23, FIFO 0.02/0.14, OTHER 0.02/0.15, RR 0.02/0.15 | Sec III-E, Table VIII |
| Insight 5: hardware | GPU workstation best mean latency; all four devices show long-tail latency; Fog Node has the smallest PINet variation range; GPU workstation has lowest YOLOv3 variation | Sec III-F, Fig 13 |
| Insight 5: perf event ratios (with / without post-processing) | mostly 1.0 to 1.25; larger outliers: cpu-migrations 4.54 (PINet, third column), context-switches 1.36 (YOLOv3 AGX), branch-misses 1.18 (PINet AGX); column mapping AGX/Fog/GPU read from the table layout (my reading) | Sec III-F, Table IX |
| Insight 6: ROS overhead | total delay vs total inference differs greatly for YOLOv3 and Deeplabv3, little for ORB-SLAM2; total-delay cv higher than inference cv for YOLOv3 and Deeplabv3 | Sec V-A, Fig 15 |
| Insight 6: full pipeline | highest per-module total delay almost 4000 ms; p99 for YOLOv3/Deeplabv3 exceeds the combination of the standalone cases | Sec V-B, Fig 16 |
| Insight 6: fusion delay, queue 100 | worst case over 10000 ms; average 773 ms; minimum 242 ms; queue 1000 gives smaller variation | Sec V-B, Fig 17 |

The six insights, in the paper's words, with hardware and models:
1. Scenario affects variation via the potential number of lanes/objects; pixel distribution matters more for lane detection than object detection; variation falls with rain. Hardware: Jetson AGX Xavier. Models: YOLOv3, Faster R-CNN, PINet (scenarios, pixels), Faster R-CNN and PINet (rain, size), five detectors for Fig 5.
2. I/O variation grows with subscribers per topic; ROS2 DDS better for small messages, ROS1 IPC for large. Hardware: Jetson AGX Xavier (stated as the platform for the I/O profiling). Workload: three image messages (Table V), 1 to 8 subscribers, Fast DDS.
3. Model structure matters; one-stage detectors vary less than two-stage ones; two-stage and lane-model variation comes from the number of first-stage proposals. Hardware: Jetson AGX Xavier. Models: YOLOv3, SSD MobileNetv2 (one-stage); Faster R-CNN ResNet101, Mask R-CNN Inceptionv2 (two-stage); LaneNet, PINet.
4. Deadline-based scheduling shows more variation than other RT policies; average-time deadline has far less variation than worst-observed deadline. Hardware: Jetson AGX Xavier with RT kernel. Models: PINet, YOLOv3. (See caveat on the second claim.)
5. PINet variation is mainly CPU post-processing; a powerful CPU in the embedded system would reduce variation. Hardware: AGX Xavier, Xavier NX, Fog Node, GPU workstation (Fig 13); nvprof and perf on AGX Xavier, with perf compared against Fog Node and the workstation (Table IX). Models: PINet, YOLOv3.
6. Middleware such as ROS adds large variation; long-tail latency accumulates under concurrent tasks; a larger synchronizer queue reduces variation. Hardware: Jetson AGX with RT kernel. Models: ORB-SLAM2, YOLOv3, Deeplabv3 in one pipeline.

## What it says about our core questions
Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"
Yes as evidence of variation, not of measured misses. The paper shows ranges, percentiles and tails large enough to break a 100 ms budget (Table I, Sec II-B), and a pipeline where perception tail delay hits about 4 s and fused output over 10 s (Sec V-B). The "frameworks" at fault here are mostly the surrounding stack (ROS pub/sub, message synchronization, CPU pre/post-processing, Linux scheduling) rather than the GPU kernels. It does not report a deadline-miss rate for any model, and it never counts misses against the 100 ms figure; Sec III-E reports only latency percentiles and cv under SCHED_DEADLINE, and notes the scheduler does not terminate tasks after a missed deadline.

Q2 "Why is latency hard to predict / variable on a shared accelerator?"
Partially. The paper gives a taxonomy of causes, none isolating the accelerator: input-dependent work (number of objects or lane proposals drives post-processing time, corr 0.9 or higher for two-stage models); I/O and middleware (more subscribers, message size, DDS fragmentation); CPU scheduling and co-running DNNs ("compete test" raises cv, attributed to "contention for non-preemptive system resources"; for YOLOv3, FIFO cv rises from 0.02 to 0.14); and CPU strength. It states GPU tasks are non-preemptive (citing [14]) and therefore CPU scheduling becomes the main runtime uncertainty, but it never measures GPU-side contention, kernel interleaving or memory-bandwidth sharing; multi-tenant analysis is explicitly deferred to future work. For one-stage models the GPU inference stage is the dominant variation source (corr 0.906 and 0.963), but the paper does not explain why GPU inference time itself varies.

Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"
Nothing relevant.

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- No FPGA, no accelerator sharing study (multi-tenancy is future work), no deadline-miss rates, no proposed mitigation or evaluation. It is a characterization paper with "insights", and the conclusion is two sentences.
- Mostly CPU-side variation. nvprof showed only the inference stage runs on the GPU for PINet and YOLOv3; read, pre-processing and post-processing run on the CPU. Much of the variance is therefore Python/ROS/CPU effects (cProfile, TensorFlow 1.15, Python 3.6 era stack), not GPU nondeterminism. This weakens it as evidence for GPU-sharing unpredictability (my inference).
- Table I does not state which device or setting it was measured on, or the sample count. The "majority of DNN models above 100 ms" claim rests on 4 of 7 models.
- Insight 4's second half appears inconsistent with Table VIII cv: Deadline-2 (average) has higher cv than Deadline-1 (worst observed) in all four cases (PINet single 0.44 vs 0.30, PINet compete 0.44 vs 0.33, YOLOv3 single 0.27 vs 0.13, YOLOv3 compete 0.23 vs 0.21), yet the text says the average-time deadline is "much better". The supporting evidence may be in the percentile plot Fig 12, whose values are not recoverable from the extracted text. Verify against the PDF figure.
- Table IV reports sigma values (for example 7862.6 ms for PINet at mean 228.9 ms) that are implausible as a standard deviation in ms given cv = sigma/mu is quoted; possibly a variance or outlier-driven. Treat the rain numbers with caution. Also rain is synthetic (rendered onto KITTI), and the explanation (fewer proposals) is plausible but the "lower variation under rain" result is a property of the inputs, not of hardware.
- Fog Node has no listed GPU (Table II) yet is used in "CPU vs GPU architecture" comparisons; the Insight 5 argument ("more powerful CPU helps") rests on perf-event ratios that are mostly between 1.0 and 1.25, with a few outliers, and on three devices that also differ in OS/software setup. Causality is asserted, not tested.
- DVFS disabled, MAXN, other applications off: this removes thermal/frequency variation, so the numbers are a lower bound on real in-vehicle variability for the Jetson runs (my inference). Hardware generations are old (Xavier, Volta, 2080 Ti, TensorFlow 1.15, CUDA 10.2).
- FP32 only; no TensorRT or INT8; no batching.
- End-to-end system uses ORB-SLAM2, YOLOv3 and Deeplabv3 simultaneously on one board with ROS and a queue-based synchronizer; the 4 s and 10 s tails look like queueing and backlog effects (my inference) and depend strongly on the synchronizer queue size (100 vs 1000) and unthrottled publishing, which the paper does not control for. Whether camera or file input was used in Sec V is not stated.
- Table III is missing from the numbering (II is followed by IV). The paper is an arXiv v1 with no stated venue.
- Several figure values (Fig 12, 13, 15, 16) are not recoverable as numbers from the extracted text, so only the statements about them are noted here.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "Setting up deadlines is usually based on the worst observed execution time." (Sec I)
- "The worst case of the fusion message's delay goes to over 10000ms, which means that the control module gets the lane or objects' perception results after 10 seconds since the camera captures objects or lane." (Sec V-B)
- "Since GPU tasks are non-preemptive, CPU scheduling becomes the main uncertainty in runtime" (Sec III-E)
- "Analysis of the time variations for multi-tenant DNN inference exceeds the topic of this paper and will be our future work." (Sec III-E)

## Open questions / things to verify
- Check Fig 12 in the PDF to see whether the average-time deadline really beats the worst-time deadline on percentiles, given Table VIII cv shows the opposite.
- Confirm the Table IX column order (AGX, Fog, GPU workstation for each model) and which device Table I was measured on.
- Check the unit and meaning of sigma in Table IV.
- Find the source of the 100 ms / 1 m / 40 km/h figure ([27], Kato et al., IEEE Micro 2015) before citing it as a requirement.
- Find out what the paper cites as [14] (Basaran and Kang, preemptive GPGPU execution) for the non-preemptive GPU claim, and whether newer GPUs and drivers still hold it.
- Look for follow-up work that actually measures GPU-side contention (multi-tenant) since this paper defers it; the compete-test result (Table VIII) is the only co-run data here.
- Does the post-processing / CPU variance finding transfer to our setting (satellite or physical-AI pipelines with different pre/post-processing)? (my inference: only if post-processing is data-dependent there too)
