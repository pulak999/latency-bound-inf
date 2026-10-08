# NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation — Juan M. Delfa Victoria, Taran Cyriac John (JPL / Caltech, Victoria Univ. of Wellington), Andrew W. Herson (Loft Orbital); IEEE-style conference paper, 2026 (cs.AI, v1 5 Jun 2026; venue not stated in the text); arXiv 2606.18271; https://arxiv.org/abs/2606.18271

## One-paragraph summary
NAVI-Orbital is a LangGraph-orchestrated software stack (conductor, detector and dialogue agents) that runs Gemma 3 4B (4-bit, with a multimodal projector) on Loft Orbital's YAM-9 LEO satellite. It classifies each captured scene against an operator-supplied label list, writes a text description, and answers operator questions about the accumulated results. The authors claim the first in-orbit run of a vision-language model doing autonomous multi-modal inference onboard (April 2026). Evidence comes in four tiers: a 7,960-image ground benchmark (88.16% accuracy), engineering-model timing on the YAM-9 compute nodes, Flatsat validation, and in-orbit runs on preloaded images plus two live captures. The paper is primarily a feasibility and systems-integration demonstration. It reports almost no in-orbit timing, thermal or power data; latency numbers come from ground engineering-model tests and a desktop GPU, and the only in-flight timing statement is a single phrase in the Conclusions.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
**Platform.** YAM-9 is described as a multi-mission satellite with a "heterogeneous compute cluster" of edge-class processing elements (Sec. III-A). Table I (Sec. V-B) lists three computers: Comp0 (4x2 GHz CPU0, Ubuntu 20.04), Comp1 (12x2 GHz CPU1, 248 TOPS GPU1, Ubuntu 20.04), Comp2 (12x2 GHz CPU2, 248 TOPS GPU2, OpenEmbedded Scarthgap 5.0). Only Comp2 supported GPU acceleration with NAVI-Orbital; the other nodes had library conflicts (Sec. V-B). Flight ran on Comp2 (Sec. VII-D b: "bare-metal Python virtual environment on Comp2"; Sec. VI-D1 mentions "GPU2 health"). Engineering models were Jetson Orin development kits (Sec. V, stage 2). The paper does not say which Jetson SKU; the references cite Orin Nano/NX/AGX docs (my inference: AGX Orin class, since 248 TOPS is not otherwise explained).

**Compute split.** "Heterogeneous CPU-GPU pipeline: all transformer layers are offloaded to the GPU, which dominates the token-generation phase, while four CPU threads handle ... tokenization, sampling, image I/O marshalling, and LangGraph state management" (Sec. IV-D1). Context window 20,000 tokens; output capped at 1,000 tokens "to bound per-image latency"; temperature 0.2. Gemma 3 4B needs at least 8 GB VRAM (Sec. III-A). No FPGA or other accelerator is part of the system.

**Model/software.** Gemma 3 4B instruction-tuned, Q4_0 GGUF via llama.cpp with mmproj (Sec. II-C2); adapters also exist for HuggingFace/bitsandbytes (Sec. III-B). The ground AID run is described as matching "the flight build: Gemma 3 4B at 4-bit quantization via bitsandbytes" (Sec. IV-B1), which conflicts with the GGUF/llama.cpp description (see Open questions). Interface to the spacecraft bus is a file inbox/outbox; the service runs a continuous loop with a configurable sleep when no work is available (Sec. III-B, Fig. 1).

**Workloads.** Per-image classification into one label plus description (same prompt on ground, Flatsat and orbit); a dialogue agent answers 16 scripted questions over the classification log (17 in the live-capture figures). Datasets: curated Google AID (18 classes, 7,960 images), 7 Sentinel-2 tiles, 5 Loft cross-validation images, 2 live YAM-9 captures (Sec. IV-B).

**What was measured and how latency is obtained.** All latency figures are measured wall-clock values from real runs; nothing is injected or simulated. (a) Desktop GPU, AID benchmark, 10 parallel GPU workers: "Average per-image inference time was 5.6 s" (Sec. VI-B). (b) Engineering model of Comp0/1/2 on the five cross-validation images: total execution time (Table II) and "average VLM cycle time for the five processed images was 26.6 seconds" (Sec. V-B). (c) Comp2 tegrastats-style telemetry (frequencies, temperatures, power rails) during the five VLM cycles (Fig. 3, Sec. V-B). In orbit, the paper reports outputs (labels, descriptions, dialogue answers) and a match against Flatsat results; no per-image latency, power or temperature is given.

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Quantity | Value | Source |
|---|---|---|
| AID curated set size / valid classifications | 7,960 images / 7,956 valid | Sec. VI-B |
| Overall accuracy, macro F1 | 88.16%, 0.87 (weighted avg F1 0.88) | Sec. VI-B, Table III |
| Strongest / weakest classes (F1) | Highway Interchange 0.97, Bridge 0.96, Airport 0.95 / Forest 0.72, Commercial District 0.76, Water Reservoir 0.77 | Sec. VI-B |
| Parse failures | 4 of 7,960 (0.05%); 2 fixed by retry, 2 unresolved ("Golf Course", "Park") | Sec. VI-B, VII-A |
| Retry convergence claim | ">99.9% of cases" | Sec. VII-A |
| Avg per-image inference time, desktop GPU | 5.6 s (10 GPU workers used for the benchmark) | Sec. VI-B, IV-B1 |
| Engineering-model total time, 5 cross-val images | Comp0 38 min, Comp1 37 min, Comp2 37 min (Docker, no HW accel); Comp2 13 min (venv, GPU) | Table II |
| Average VLM cycle time, Comp2 EM, 5 images | 26.6 s | Sec. V-B |
| GPU peak figures | 248 TOPS per GPU on Comp1 and Comp2 | Table I |
| Retry budget | up to 3 attempts (ground), up to 10 (flight), 30 s sleep between attempts | Sec. IV-A, IV-D1 |
| Flatsat preloaded functional test | 25/25 correct (18 AID, 7 Sentinel-2), no retries needed | Table V, Sec. VI-C1 |
| Flatsat cross-validation | 5/5 images classified correctly (qualitative, Table VI) | Sec. VI-C2 |
| In-orbit preloaded/cross-val subset | outputs consistent with Flatsat and pre-launch benchmarks; subset size not stated | Sec. VI-D1, Table VII |
| Live in-orbit captures | 2 (Toulouse 43.76N, 1.38E; Argentina coast 47.80S, 65.91W) | Sec. IV-B3b, VI-D2, Figs. 6-7 |
| Live capture dates | Toulouse 16 Apr 2026; Argentina 24 Apr 2026 (read from timeline) | Fig. 4, Abstract |
| Inter-capture cadence claim | "within a 20-second inter-capture cadence under flight thermal and power constraints" | Sec. VIII only |
| Live image size vs result size | ~9 to 14 MB frames; 700 to 1060 byte JSON results (about four orders of magnitude) | Sec. VII-C |
| Launch / checkout | 28 Nov 2025; Comp-2 checkout complete 25 Mar 2026 | Fig. 4, Sec. VI-A |

**Reconciliation of the timing numbers (check 1).**
- 26.6 s is a per-image "VLM cycle time" averaged over five images on the Comp2 engineering model (Jetson-class, ground). It is a ground measurement, not a flight one. The 5.6 s is a different machine (an unspecified "desktop GPU"), different run (7,960 images, 10 parallel GPU workers), and apparently a different quantization stack. The ratio of about 4.75x is not explained by the paper and the two are not comparable measurements of the same thing.
- The 13 min Comp2 total for 5 images is about 156 s per image if divided evenly (my inference), roughly six times the 26.6 s cycle time. The paper does not say what the rest covers (model load, the dialogue stage, startup, retries are all possible; not stated). The 37 to 38 min CPU-only totals are about 7.4 to 7.6 min per image (my inference).
- The 20-second figure appears once, in the Conclusions: two live captures were processed "within a 20-second inter-capture cadence under flight thermal and power constraints." It is not defined or supported in Sec. VI-D2 or anywhere else. It is ambiguous whether it means (i) the spacing between capture commands, (ii) a per-image processing deadline, or (iii) something else. If it were a per-image processing time, it would be shorter than the 26.6 s measured on the ground EM, and the paper does not reconcile this. Also, the two live captures are on different dates in Fig. 4 (16 and 24 April), so a 20 s spacing between them is not obviously what is meant (my inference; could instead refer to spacing between frames within each pass).
- What WAS measured in orbit: classification labels and free-text descriptions for the preloaded/cross-validation subset and the two live captures; dialogue answers; agreement of outputs with Flatsat results; implicit confirmation that GPU2 and the satellite were healthy (Sec. VI-D1). The two live captures were also "re-validated in Loft's ground test infrastructure" (Sec. IV-B3b) and re-run after dynamic-range recovery on the Flatsat (Table VIII).
- What was NOT reported from orbit: per-image or end-to-end latency, any distribution or tail of latency, number of retries actually used in flight, GPU/CPU utilisation, temperatures, power draw, throttling events, or memory use. The thermal and power traces in Fig. 3 are from the ground Comp2 engineering model, not from flight.

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Weak, indirect evidence only. The paper states a latency concern (output capped at 1,000 tokens "to bound per-image latency"; "real-time feasibility under the spacecraft's processing budget", Sec. IV-C, IV-D1) and claims a 20 s cadence was met in flight, but it never shows an in-orbit latency measurement, a deadline definition, or a miss rate. The ground EM cycle time (26.6 s) is longer than the 20 s cadence, and the ground-to-desktop gap (26.6 s vs 5.6 s) shows that the same pipeline's latency depends strongly on the platform. Prefill/token generation on the GPU and CPU-side orchestration are named as the phases (Sec. IV-D1), but no breakdown is given. So the paper is a data point that a VLM pipeline runs on a flight GPU, not that deadlines are or are not missed.

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Partial. The authors justify up to ten retries with 30 s sleeps as "resilience against the transient hardware and scheduling anomalies inherent to the space environment" (Sec. IV-D1), which implies they expect non-trivial run-to-run variability, but they give no data on it or on its causes. Fig. 3 and Sec. V-B show the Comp2 GPU saturating the external memory controller and then CPU load, with TJ temperature "critical to GPU throttling" listed as a monitored metric, so thermal throttling and memory bandwidth are named as concerns on the ground EM. Sec. IV-D1 also notes that GPU work and CPU orchestration are interleaved. The paper says nothing about other tenants, contention, or shared-accelerator scheduling; YAM-9 is described as multi-mission, but whether NAVI shared Comp2 with other workloads is not stated.

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
Nothing relevant.

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- No in-orbit timing, power or thermal telemetry is reported, yet the Conclusions assert operation "under flight thermal and power constraints" and a 20 s cadence. The only thermal/power plot (Fig. 3) is from the ground EM. Weak support for any claim about real flight latency or variability.
- The in-orbit sample is two live captures plus an unspecified preloaded/cross-validation subset; the authors call it small (Sec. VII-B). Variability or tail latency cannot be inferred from n=2.
- Timing numbers are not on a common basis: 26.6 s (Comp2 EM, 5 images, GPU2+CPU2, bare-metal venv) vs 5.6 s (desktop GPU, 10 workers, bitsandbytes per Sec. IV-B1) vs 13 min total. Whether 5.6 s is per-worker latency or effective throughput under 10 workers is unclear (my inference: ambiguous).
- Accuracy is also not clean: 7,956 are called valid while Sec. VII-A says 2 of the 4 parse failures were resolved by retry, which would give 7,958 (my inference: the 4 failures are counted before retry, but the paper does not say). The 88.16% is over valid classifications, and it was measured on the ground, not in orbit; in-orbit agreement is qualitative (output text matched Flatsat). No in-orbit accuracy exists.
- The retry loop with 30 s sleeps means per-image latency is unbounded in principle: with 10 attempts, sleep alone could add 300 s per image (my inference, if every attempt fails), and the paper gives no timing guarantee. The authors acknowledge that retries, regex gate and temperature "reduce [hallucination] frequency but do not bound it" (Sec. VII-B).
- "Deterministic" in this paper refers to graph state transitions and constrained output labels (Sec. III-B, IV-D1), not to timing determinism. It should not be cited as latency determinism.
- The software was frozen after EM testing and deployed to user space on Comp2 only; the container route failed on the target GPU at integration time (Sec. V-B, VII-D b). Portability claims are limited to this node.
- No discussion of radiation faults, FPGAs, other tenants or contention, deadlines, or power/thermal limits in numbers (no watt budget or temperature limit is given). "Strict size, weight, and power constraints" is stated without values (Sec. I-B).
- Authors are the tool builders, with Loft providing hardware; the "first" claim is theirs ("to the authors' knowledge").

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "up to ten retries with a 30-second inter-attempt sleep provide resilience against the transient hardware and scheduling anomalies inherent to the space environment, ensuring autonomous throughput without requiring ground operator intervention during a pass." (Sec. IV-D1)
- "The average VLM cycle time for the five processed images was 26.6 seconds." (Sec. V-B)
- "within a 20-second inter-capture cadence under flight thermal and power constraints" (Sec. VIII)
- "No retry mechanism, regex output gate, or sampling-temperature setting eliminates the possibility of hallucination; these mitigations reduce its frequency but do not bound it." (Sec. VII-B)

## Open questions / things to verify
- What exactly does the "20-second inter-capture cadence" refer to, and was it a measured per-image processing time in orbit or a capture-command spacing? Contact the authors or check for a longer version or Loft telemetry.
- Why is the flight-class cycle time 26.6 s (Comp2 EM) versus a 20 s cadence? Does flight Comp2 run faster than the EM, or is the cadence only a capture interval with a queue behind it?
- What accounts for the gap between 26.6 s per image and 13 min total for five images on Comp2 (Table II)?
- Was the 5.6 s per image per-worker latency or total time divided by images given 10 parallel workers? Which GPU?
- Did any in-orbit image actually trigger a retry, and what were the in-orbit retry counts and total times?
- Is Comp2 shared with other YAM-9 tenants during NAVI runs? No statement in the text.
- Flight quantization stack: Q4_0 GGUF via llama.cpp (Sec. II-C2) or bitsandbytes 4-bit (Sec. IV-B1)? Affects whether the 5.6 s and 26.6 s refer to the same software.
- Several cross-references in the text look wrong (for example "Section V-C" for compute-cluster benchmarking, which is Sec. V-B; "Section VI-D2" cited for detection outputs; "Section III-B" cited for the retry loop that lives in IV-A/IV-D), so section citations here follow the headings actually present.
- Which Jetson/YAM-9 hardware does "248 TOPS" correspond to, and what power mode was used in flight?
