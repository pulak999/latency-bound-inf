# Earth+: on-board satellite imagery compression leveraging historical earth observations — Kuntai Du, Yihua Cheng, Peder Olsen, Shadi Noghabi, Ranveer Chandra, Junchen Jiang (UChicago, Microsoft Research); ASPLOS 2025; arXiv 2403.11434 (the extracted text is the v1, 18 Mar 2024 version); https://arxiv.org/abs/2403.11434

## One-paragraph summary
Earth observation satellites capture far more imagery than the downlink can carry. Earth+ cuts downlink use by encoding only the tiles that changed relative to a reference image of the same location, and by making that reference as fresh and cloud-free as possible. Previous reference-based schemes use a reference captured by the same satellite, which is on average 51 days old (cloud-free). Earth+ lets the ground station pick the freshest cloud-free image from any satellite in the constellation and upload it to the satellite over the thin uplink (250 kbps). The uplink budget is met by downsampling references heavily, by uploading only changed tiles of references (the satellite caches references on board), and by doing accurate cloud detection on the ground. On board, the satellite runs a cheap decision-tree cloud detector, illumination alignment, tile-level change detection and JPEG-2000 ROI encoding. Evaluated on Sentinel-2 and Planet data with Doves-like specs, it reports 1.3-3.3x downlink savings at equal PSNR versus Kodan and SatRoI, with no more on-board storage, compute or uplink than the baselines. This is a bandwidth/compression paper. It has no real-time inference, no deadlines, and no accelerator.

## Setup (hardware, models, workloads, what was measured, how latency is obtained: measured vs injected)
- Workload: image compression for downlink, not neural inference. On-board pipeline (Sec 5): decision-tree cloud removal on a 64x downsampled image (heavy clouds only, via the IR band); drop images with >50% cloud; linear-regression illumination alignment; tile (64x64) change detection against a downsampled reference with a threshold theta; JPEG-2000 (Kakadu) ROI encoding of changed tiles at a constant bits-per-tile gamma. Ground side: more accurate, compute-heavy cloud detector, reference selection, upload.
- Baselines (Sec 6.1): Kodan (drops cloud data, downloads the non-cloudy area; uses an expensive cloud detector) and SatRoI (reference-based encoding with a fixed reference).
- Data (Sec 6.1, Table 2): Sentinel-2, 11 locations in Washington State, 1600 km2 each, 1 year, 13 bands, 2 satellites, images downsampled 4x; Planet, one location in the U.S., 3 months, <5% cloud, 4 bands. Fig 19 uses Planet thumbnails of Denver (1 Jul to 1 Oct 2023), simulated for different constellation sizes.
- Satellite spec (Table 1, Doves, 2017-2018; some values inferred by the authors from other sources): 10 min ground contact, 7 contacts/day, 250 kbps uplink, 200 Mbps downlink, 360 GB on-board storage, 6600x4400 images, 150 MB raw file, GSD 3.7 m.
- Link modelling: uplink assumed constant 250 kbps; downlink bandwidth is computed as the minimum average rate needed to move a fixed amount of data within a 10 minute contact. Quality metric is PSNR. No application-level accuracy is measured.
- Compute (the check you asked for): the only compute measurement is the "Computation overhead" paragraph and Fig 16 (Sec 6.2): "We use an AMD EPYC 7452 CPU with 16 cores to test the runtime." Per-image runtimes reported in the text: encoding 0.65 s for all systems; Kodan cloud detector 0.39 s; Earth+ and SatRoI cloud detector 0.12 s. Change-detection time appears only as a bar in Fig 16 with no number in the text, and it is said to be shorter for Earth+ than SatRoI because of downsampled references. Not reported: the on-board processor model, a flight-grade or embedded device, energy, power, thermal, memory use, or any accelerator (GPU/FPGA/NPU). The word "power" and the terms "GPU", "FPGA" and "energy" do not occur in the text. Compute is therefore treated as a soft relative constraint ("not more compute than the baselines", and "make cloud detection faster compared to running JPEG-2000 encoding"), checked by timing on a server CPU, not as a fixed measured on-board budget with a stated number. The design is shaped by the compute limit qualitatively: the accurate cloud detector (tens of NN layers, image sequences) is judged unaffordable on board and moved to the ground (Sec 4.3, Sec 5).
- Latency: no end-to-end latency, deadline, or reaction time is measured. The intro says Earth+ "can reduce the reaction delays of ground applications (e.g., forest-fire alerts) by upto 3x", but I found no experiment in Sec 6 that measures reaction delay (my inference: it is derived from the downlink saving). Nothing is injected; link capacities are fixed model parameters.

## Key numbers (a table of the headline results; every number must come from the text, with the section/table/figure it came from)
| Result | Value | Source |
|---|---|---|
| Downlink saving vs strongest baseline at equal PSNR, Sentinel-2 | 1.3-2.0x | Sec 6.2, Fig 11a |
| Downlink saving, Planet (40+/48 satellites) | 2.8-3.3x | Sec 6.2, Fig 11b |
| Headline claim | 3.3x less downlink; 1.3-3.3x overall | Abstract, Sec 1, Sec 6 |
| Images where Earth+ downloads <20% of tiles | >60% of images | Sec 6.2, Fig 12 |
| Baselines download >80% of tiles | for >70% of images | Sec 6.2, Fig 12 |
| Images fully downloaded by Earth+ (guaranteed monthly download) | 20% | Sec 6.2, Fig 12 |
| Earth+ vs baselines over a year on one location | 5-10x fewer tiles most of the time | Sec 6.2, Fig 13 |
| Locations where Earth+ beats strongest baseline | 10 of 11 (no gain at H, marginal at D: snow) | Sec 6.2, Fig 14 |
| Bands improved | all 13 Sentinel-2 bands | Sec 6.2, Fig 14 |
| On-board storage | 30 GB (SatRoI), 255 GB (Kodan), 24 GB (Earth+) | Sec 6.2, Fig 15 |
| Storage saved by not storing unchanged tiles | about 80% | Sec 4.3 |
| Reference caching storage overhead | about 9% (at most 0.08a MB vs 2x0.87a MB) | Sec 4.3, App A |
| Per-image runtime, one imagery, 16-core EPYC 7452 | encode 0.65 s (all); cloud detect 0.39 s (Kodan) vs 0.12 s (Earth+, SatRoI) | Sec 6.2, Fig 16 |
| Reference image compression to fit uplink | over 10,000x | Sec 6.2, Fig 17 |
| Downsampling used in the Fig 8 example | 2601x; only 1.7% of tiles mis-detected | Sec 4.3, Fig 8 |
| Extra downlink saving from raising uplink to 4 Mbps | 22 Mbps | Sec 6.2, Fig 18 |
| Compression ratio vs constellation size (Planet thumbnails, rough estimate) | 3x (1 sat) to 10x (16 sats) | Sec 6.2, Fig 19 |
| Fraction of captured data actually downloaded (cited estimate) | 2% | Sec 1, Sec 2.2 (citing [37]) |
| Doves downlink idealized capacity vs accumulated data | 21.6 TB of 191.25 TB, "less than 12%" | Sec 2.2, footnote 3 |
| Fraction of tiles changed (cloud-free, previous 5 days) | 20% on average, up to 5x saving | Sec 1 |
| Reaction delay of ground applications | "up to 3x" reduction (claimed, not measured as far as I found) | Sec 1 |
| Reference age, satellite-local vs constellation-wide | 51 days vs 4.2 days (12x) | Sec 3, Fig 5; intro says 4.21 days, "every 50 days" |
| Toy example of age effect | 30 d to 10 d reference age; changed area 55% to 15% (3.6x) | Sec 4.1, Fig 6 |
| Changed tiles vs reference age | 3x more changed tiles going from 10 to 50 days | Sec 3, Fig 4 |

## What it says about our core questions
Q1 "Is this a problem? Do inference frameworks miss deadlines, and where?"
Nothing relevant. The paper's problem is downlink capacity, not missed inference deadlines. The only latency-flavoured statement is the unmeasured "reaction delays ... up to 3x" claim in Sec 1. It is a link-bandwidth argument, not about on-board processing latency.

Q2 "Why is latency hard to predict / variable on a shared accelerator?"
Nothing relevant. There is no accelerator, no sharing and no latency variance analysis. Only fixed per-image CPU runtimes are reported (Sec 6.2).

Q3 "What can we learn from FPGAs? Can the GPU borrow determinism, and what belongs on each?"
Nothing relevant. No FPGA or GPU is mentioned. (A loose design observation, not an answer: the paper moves the expensive step, accurate cloud detection, to the ground and keeps only a cheap decision-tree detector on board, which is a compute-placement decision driven by an unquantified on-board compute limit. I would not cite this as evidence on determinism.)

## Limitations and caveats (what the paper does not measure, assumptions, anything that weakens it as evidence for us)
- No on-board hardware is named or measured; the compute number comes from a server-class CPU. No energy, power or flight-processor data. On-board compute is a qualitative constraint, not a measured budget.
- No latency, deadline or tail-latency measurement; runtimes are single per-image means with no variance reported in the text.
- Link capacities are fixed assumptions (constant 250 kbps uplink, 10 min contact); they are not measured and nothing is injected.
- PSNR is the only quality metric; no downstream application accuracy.
- Ground-side costs are unevaluated (cloud detection, sharing images across ground stations); Sec 8 acknowledges added ground-station complexity and that Earth+ needs a software update on board. Lossy only. Dropping images with >50% cloud makes it unsuitable for cloud-sensitive uses such as weather forecasting.
- Evaluation is small: Planet data is one location, 3 months; the 1-to-16 satellite scaling is a simulation on thumbnails the authors call "a rough estimation"; theta is fixed from first-year data.
- Internal inconsistencies in the text: 48 satellites (Sec 6.1, Table 2) vs "more than 40" (Sec 6.2) vs 40 (Sec 1); Planet area 64 km2 (Sec 1) vs 36 km2 (Table 2); raw image 150 MB (Table 1) vs 300 MB (App A); GSD 3.7 m (Table 1) vs 3 m (App A); the Fig 11 caption contains a leftover draft note "(KD) Redo sentinel-2 results"; Table 2 swaps the Planet/Sentinel-2 citations. This is the v1 text; the ASPLOS version may differ.
- The "over 90%" check: see the next paragraph.

Over-90% check (requested): the paper does not state that over 90% of captured images are not downloaded. The closest statements are: "According to a recent estimate, only 2% of the total image data observed by each satellite can be downloaded to the ground [37]." (Sec 1; repeated in Sec 2.2 as "only about 2% of the images captured by satellites are actually downloaded to the ground [37]"), footnote 3 (Doves idealized download of 21.6 TB is "less than 12% of the 191.25 TB of data accumulated"), and the Conclusion's "most of the imagery observed by the satellites is not downloaded". The only "91%" in the paper is unrelated: "the Biomass mission omits about 91% of the Earth's surface" (Sec 2.3). Do not cite this paper for "over 90% not downloaded" in those words; "about 98%" (via [37]) or "more than 88% (idealized)" are what the text supports. The 2% figure is a citation of Kodan [37], not a measurement here.

## Quotes worth keeping (2-4 short verbatim quotes with section refs)
- "According to a recent estimate, only 2% of the total image data observed by each satellite can be downloaded to the ground [37]." (Sec 1)
- "We note that the age of the most recent cloud-free reference image is 51 days on average." (Sec 3, Fig 5 discussion)
- "We use an AMD EPYC 7452 CPU with 16 cores to test the runtime." (Sec 6.2, Computation overhead)
- "the satellite does not have sufficient computation on board to accurately detect cloud-free imagery." (Sec 4.3)

## Open questions / things to verify
- Reference-age mechanism (requested), for reuse: a reference is stale if it is old or cloudy; changed-tile fraction grows with age (Fig 4: 3x from 10 to 50 days). Satellite-local reference averages 51 days (cloud-free; single-satellite revisit 10-15 days plus roughly 2/3 of Earth cloud covered) vs 4.2 days constellation-wide (Fig 5, Sec 4.1). At each ground contact the ground uploads the latest cloud-free image for locations the satellite will overfly before the next contact (locations predicted from TLE data, footnote 7). The satellite caches references on board so it only needs changed tiles uploaded. If the uplink dips, "the skipped images are randomly chosen" and the satellite uses older cached references, "with a slight increase on the downlink capacity" (Sec 5); the size of this penalty is not quantified in the text. A full cloud-free image is downloaded once per month as "guaranteed downloading" (Sec 5), and the false-negative risk of downsampled references is handled by a low threshold theta (Sec 4.3). Per-tile references are judged marginal (footnote 6). No distribution of reference age in the full end-to-end system (including uplink limits and ground processing delay) is reported beyond the Fig 5 CDF (my inference: Fig 5 is the idealized selection, not the deployed system).
- Verify against the published ASPLOS 2025 version: whether a reaction-delay experiment, on-board hardware, or per-stage runtimes were added.
- Fig 16 bar values (change detection, total) are not in the text; read the PDF figure if exact per-stage numbers are needed.
- Kodan [37] (source of the 2% figure) and Serval [64] may be better sources for the satellite compute-budget discussion; check if they give flight-hardware numbers.
- Whether the "up to 3x" reaction-delay claim is derived or measured.
