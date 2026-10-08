# Myrtle.ai VOLLO on STAC-ML Markets (Inference) — press release (29 Apr 2026) and STAC report page
Press release: https://aapnews.aap.com.au/news/cision20260429AE43196
STAC report page (SUT ID MRTL260323): https://docs.stacresearch.com/MRTL260323
Type: **vendor press release plus the benchmark body's summary page, secondary sources.** The full audited report is gated behind STAC Observer registration and was not read.

## Summary
Myrtle.ai claims VOLLO, its FPGA inference stack, set a new latency record on the STAC-ML Markets (Inference) benchmark,
halving its own previous record, with "deterministic" microsecond-level inference for trading workloads.

## Setup as reported
- Stack: myrtle.ai VOLLO (Rev C) on a Silicom FBAP4@VP18-2L0S PCIe card (Silicom Artena) with an AMD Versal Premium VP1802 FPGA, in a Supermicro AS-2015CS-TNR server. "PCIe Gen5x8 and more than 3.3M programmable LUTs."
- Benchmark: STAC-ML Markets (Inference), all three benchmark models.
- Models: the STAC page names LSTM_A, LSTM_B and LSTM_C. The press release gives no model sizes.

## Numbers
| Claim | Value | Source |
|---|---|---|
| Latency, 99th percentile | "as low as 2 microseconds" | press release |
| LSTM_A | "first sub 2 us 99p latency" | STAC page |
| LSTM_B | "first sub 3 us 99p latency" | STAC page |
| LSTM_C | "first sub 8 us 99p latency" | STAC page |
| Improvement | "halving its previous record" (prior value not given) | press release |

## Core questions
- **Q1:** Nothing relevant. This is a latency record on a trading workload, not a deadline-miss study.
- **Q2:** Nothing relevant. The release asserts "deterministic" latency but gives no jitter metric.
- **Q3:** Useful as a ceiling example for FPGA inference: single-digit microseconds at p99 for LSTM-class models, audited by a third party. The p99 framing matters for us because it is a tail number, which is exactly what hls4ml's cycle counts do not give us.

## Caveats
- Vendor claim. The press release has no GPU comparison, no throughput, no model sizes and no jitter figures.
- LSTMs, not transformers. The result says nothing about transformer or VLA-scale models.
- The STAC page states no date. The SUT ID MRTL260323 may encode 23 March 2026 (my inference), which fits the 29 April 2026 press release.
- Only milestone claims were visible. Full per-percentile data, throughput and error versus reference sit behind the registration wall.

## Quotes worth keeping
- "VOLLO achieved latencies as low as 2 microseconds (99th percentile)."
- "Such low, deterministic latency enables users to make more intelligent decisions."

## Use on the slide
Keep the "vendor claim" label. Cite the STAC page for the three sub-N-microsecond p99 milestones.
