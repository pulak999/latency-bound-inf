# Fast inference of deep neural networks in FPGAs for particle physics — Duarte, Han, Harris, Jindariani, Kreinar, Kreis, Ngadiuba, Pierini, Rivera, Tran, Wu; JINST (prepared for submission, Fermilab-PUB-18-089-E), 2018; arXiv 1804.06913v3 (28 Jun 2018); https://arxiv.org/abs/1804.06913

## One-paragraph summary
The paper introduces hls4ml, a compiler that turns a trained Keras or PyTorch fully connected network into FPGA firmware through Vivado HLS, aimed at the LHC Level-1 trigger where latencies are hundreds of nanoseconds to microseconds at a 40 MHz collision rate. Using a jet-substructure classifier (5 classes, 16 inputs, 3 hidden layers, 4389 parameters) as the case study, it shows how three handles control the latency/resource trade-off: compression (iterative L1 pruning, 70% of weights removed), quantization (fixed-point arithmetic, about 16 total bits suffice), and parallelization (a multiplier "reuse factor" R). The design is fully pipelined, layers run sequentially, and each layer's latency follows a closed-form expression (Eq. 2.4) in cycles. The compressed network synthesizes to about 15 clock cycles (75 ns at 200 MHz) with initiation interval equal to the reuse factor, on a Xilinx Kintex UltraScale. Resource numbers come from HLS synthesis estimates, which are compared with a placed-and-routed "bare" implementation for a smaller one-hidden-layer model. The paper contains no GPU or CPU measurements; it asserts they cannot meet the budget.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Target hardware: Xilinx Kintex UltraScale xcku115-flvb2104-2-i, Vivado HLS 2017.2, clock fixed at 200 MHz for HLS runs (Sec. 3.2). Variation in results across the O(100 MHz) range is reported as negligible.
- Models (Fig. 4, Sec. 2.2): (a) 3-hidden-layer MLP, 16 inputs, 64-32-32 ReLU, 5 softmax outputs, 4389 weights and biases uncompressed; (b) 1-hidden-layer MLP, 10 inputs, 32 ReLU, 1 sigmoid output, used only for the place-and-route study because the "bare" design maps each input and output bit to a pin (Sec. 3.3). Only fully connected networks are supported; CNNs and RNNs are future work (Sec. 4).
- Workload: simulated jets (MadGraph5 + Pythia8, 200,000 events per class, 13 TeV, jet pT roughly 0.8 to 1.6 TeV), 16 expert substructure features, 60/20/20 train/validation/test split (Sec. 2.2). The neural network input features are assumed precomputed; their cost is explicitly not counted (Sec. 2.3).
- Measured: classification AUC versus fixed-point precision (software emulation of fixed point, Figs. 9, 10); DSP, FF, LUT, latency and initiation interval from Vivado HLS synthesis (Figs. 11 to 14, Table 2); DSP, FF, LUT and power from placed-and-routed implementation (Figs. 15 to 17). BRAM use is not probed (Sec. 3.2).
- How latency is obtained: it is neither a hardware measurement nor injected. Latency and initiation interval are cycle counts reported by the HLS synthesis step, converted to ns by multiplying by the 5 ns clock period (15 cycles = 75 ns). No on-board timing run, no input/output transport, and no distribution over many inferences is reported (my inference from the absence of any such measurement in the text).
- Implementation caveat on timing: the 5 ns HLS target clock was not met in place and route; for the 3-hidden-layer pruned model at 32-bit precision an 8 ns period was needed. The one-hidden-layer implementation results use an 8 ns clock (Sec. 3.3).

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Quantity | Value | Source |
|---|---|---|
| L1 trigger context | 40 MHz collision rate; L1 pipelined latencies "hundreds of nanoseconds totalling microseconds"; HLT latencies hundreds of milliseconds | Sec. 1 |
| Uncompressed 3-hidden-layer model | 4389 parameters, 3329 DSP48E, 263,234 LUT+FF, 75 ns, AUC/Expected AUC 99.68% | Table 2 (fixed point <16,6>, 200 MHz, fully pipelined) |
| Compressed model | 1338 parameters (3.3x), 954 DSP48E, 88,797 LUT+FF, 75 ns, AUC/Expected AUC 99.55% | Table 2 |
| Pruning | 3051 of 4389 weights and biases removed (70%), seven prune/retrain iterations, L1 lambda = 1e-4 | Sec. 2.3 |
| Latency, reuse factor 1 | about 15 clock cycles = 75 ns, same for compressed and uncompressed | Sec. 3.2, Fig. 11 |
| Latency versus reuse factor | increases by 4 clocks per unit of reuse factor (four layers of neuron values) | Sec. 3.2, Fig. 14 discussion |
| Initiation interval | equals the reuse factor by design; below it only at very low precision, where multiplies move from DSPs to FFs/LUTs | Sec. 3.2, Fig. 14 discussion |
| Layer latency model | L_m = L_mult + (R - 1) x II_mult + L_activ; total latency is the sum over layers plus inter-layer connection latency | Eq. 2.4, Sec. 2.3 |
| Summary latency claim | "approximately 75-150 ns" at 200 MHz, using "roughly 10% of the available DSPs" "with results varying with the initiation interval" | Sec. 4 |
| Precision needed | about 16 total bits (<16,6>); weights need at least 3 integer bits (2 for the largest magnitude, 1 sign); integer bits fixed at 6 in scans | Sec. 3.1, Sec. 2.3 |
| DSP granularity | one DSP48E1 multiplies 25 x 18 bit; 25 x 19 needs two | Sec. 2.3 |
| DSP scaling with R | DSP usage falls proportionally to the reuse factor (R = 1 to 6 scanned) | Sec. 3.2, Fig. 12 |
| Clock slip at implementation | 8 ns needed for a block designed for 5 ns (3-hidden-layer pruned, 32-bit, all reuse factors); gap grows with the fraction of FPGA used | Sec. 3.3 |
| HLS versus implementation | FF estimates typically 2 to 4 times high (more above 32 bits); LUT overestimated; DSP agrees for single-DSP multiplies (< 24 bits), over-estimated above | Sec. 3.3, Figs. 16, 17 |
| Power | rises with precision, falls as reuse factor rises (3-hidden-layer pruned, reuse 1 to 6) | Sec. 3.3, Fig. 15 (axis to roughly 4.5 W, read from plot) |
| HLT regime (not measured) | HLT window 100 ms; inference "can be as long as 10 ms"; second tier reads output reduced by a factor of 400; FPGA "O(100) times faster than CPUs" (cited from [96], not measured here) | Sec. 4 |

Model-size ceiling, as stated: no single maximum parameter count is given. What the text supports: the 4389-parameter (1338 after pruning) network fits using roughly 10% of DSPs at <16,6> with R = 1 (Sec. 4); the uncompressed version needs 3329 DSP48E (Table 2), and Fig. 11 plots uncompressed DSP use up to roughly 3e4 at the highest precisions, against a "Max DSP" line in Figs. 12 and 16 whose value is not stated in the text. The limiting resource is "for many applications" the DSP count (Sec. 2.3). Because weights live in programmable logic, not BRAM, size is bounded by logic, and the paper says weights in BRAM would give latencies "longer than the requirements for the first stages of LHC L1 triggers" (Sec. 3.2). The stated escape hatch is a large reuse factor for 10 ms tasks (Sec. 4).

## What it says about our core questions
**Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"**
Only an assertion, not a measurement: the paper says that for the L1 task "solutions with either CPUs or GPUs are not possible due to the severe time limitation imposed" (Sec. 1). It reports no CPU/GPU/framework latency, so it does not show where or how often deadlines are missed. It does establish what the deadline regime is (sub-microsecond, 40 MHz, fully pipelined) and states that inference-time comparisons on CPU, GPU and FPGA are future work (Sec. 4).

**Q2 "Why is latency hard to predict / variable on a shared accelerator?"**
Nothing relevant.

**Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"**
This is the paper's relevant contribution; the word "determinism" never appears, so the following is my reading of the mechanisms the text does describe.
- Fixed, closed-form latency: layer latency is an explicit function of multiplier latency, reuse factor and activation latency (Eq. 2.4); total latency is a sum of per-layer terms; the design is fully pipelined with a stated initiation interval equal to R. Latency in cycles is a compile-time property of the design (my inference: nothing in the text describes data-dependent timing or queuing).
- Spatial resource dedication: operations "consume dedicated resources onboard the FPGA and cannot be dynamically remapped while running" (Sec. 2.1). Each multiply has its own multiplier at R = 1, so there is no contention between inferences (my inference).
- Latency/area knob: R trades multipliers for cycles roughly linearly (Fig. 12, Eq. 2.4); the latency cost is +4 cycles per R step for the 4-layer model.
- Numerical precision as a latency lever: fixed point instead of floating point, about 16 bits with negligible AUC loss; multiplier latency increases with precision and DSP count steps abruptly at the 25 x 18 boundary (Sec. 2.3, 3.1).
- Compression removes work for free in hardware: zero-weight multiplications are optimized out, so pruning cuts DSPs by roughly the pruning fraction at unchanged latency (Table 2, Fig. 11).
- What belongs where: the paper puts tiny, fixed-structure MLPs with weights in logic on the FPGA for L1 (sub-microsecond), and suggests FPGAs as co-processors for the HLT tier at around 10 ms with large reuse factors (Sec. 4). It makes no statement about what a GPU could borrow; any GPU analogue (static schedule, no dynamic contention) is my inference, not in the paper.
- Cost of the guarantee: the model must fit in on-chip logic, so the model-size ceiling is set by DSP/LUT/FF availability and the latency budget; weights in BRAM break the L1 latency target (Sec. 3.2).

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- Latency is a synthesis-tool estimate in clock cycles, not an on-hardware measurement; no latency distribution, jitter or worst-case tail is reported. "Deterministic" is therefore inferred from architecture, not demonstrated empirically.
- The 75 ns figure assumes the HLS target clock (5 ns). The paper itself reports that place and route needed 8 ns for some designs (Sec. 3.3); if that applied to the 15-cycle model the wall-clock figure would be larger (my inference; the paper does not state a post-implementation latency for the pruned 3-hidden-layer model).
- Place-and-route study uses a different, smaller model (10-32-1) because of pin limits; the 3-hidden-layer result is only said to show "similar quantitative conclusions" where pins sufficed (Sec. 3.3).
- Input feature computation (the 16 expert features) and I/O transport are excluded from latency; the paper notes feature pre-computation "can be quite time consuming" (Sec. 2.3). The "bare" design uses pins, not the high-speed transceivers a real system would use.
- Only dense MLPs of at most 4389 parameters on one device family (Kintex UltraScale) and one tool version (Vivado HLS 2017.2; the text notes 2016.4 met timing better). No CNN, RNN, or transformer.
- No comparison with CPU, GPU or any framework; the CPU/GPU claim is unsupported by data here, and the "O(100) times faster than CPUs" figure is cited from another work.
- No contention or multi-tenant scenario: a single model owns the device, so this says nothing about shared accelerators.
- A 2018 paper; hls4ml and FPGA toolchains have changed since (my inference).
- Accuracy numbers are for a simulated-jet benchmark; the 99.55% AUC ratio is relative to the 32-bit float model of the same network, not an absolute physics metric.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "For this task, solutions with either CPUs or GPUs are not possible due to the severe time limitation imposed." (Sec. 1)
- "such operations consume dedicated resources onboard the FPGA and cannot be dynamically remapped while running." (Sec. 2.1)
- "FPGA multpliers are pipelined; therefore, the latency of one layer computation, L_m, is approximately L_m = L_mult + (R - 1) x II_mult + L_activ" (Sec. 2.3, Eq. 2.4; "multpliers" is the paper's typo)
- "Storing and accessing neural network weights from BRAMs, for instance, leads to latencies longer than the requirements for the first stages of LHC L1 triggers." (Sec. 3.2)

## Open questions / things to verify
- Find a later hls4ml or L1-trigger paper with on-board measured latency (and any jitter data) to confirm cycle-exact behavior; this paper only gives synthesis estimates.
- Confirm the available DSP/LUT/FF totals for xcku115 from the Xilinx datasheet (the text only draws "Max" lines in figures and says about 10% DSP use); do not use an unsourced number.
- Check whether the 75 ns versus 75-150 ns range in Sec. 4 corresponds to R = 1 versus R = 2 (my guess from the +4 cycles per R step; the text does not say).
- Verify the cited "O(100) times faster than CPUs" claim in reference [96] before using it.
- Look for an apples-to-apples GPU comparison (the paper defers it to future work) if we want a quantitative Q1 claim.
- Consider whether weights-in-BRAM latency numbers appear in later hls4ml work, since that sets the practical model-size ceiling for an FPGA path.
