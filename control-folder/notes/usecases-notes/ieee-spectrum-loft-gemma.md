# NASA Puts Google's Gemma Large Language Model in Orbit — Matthew S. Smith, IEEE Spectrum, 23 Jul 2026
Link: https://spectrum.ieee.org/nasa-ai-satellite-image-analysis
Type: **news article, secondary source.** The primary source is the NAVI-Orbital paper, see [navi-orbital.md](navi-orbital.md).
Read via page fetch (full text available, not paywalled). Quotes below are as returned by the fetch.

## Summary
Reports the April 2026 demonstration in which JPL's NAVI-Orbital framework ran Google's Gemma 3 4B on Loft Orbital's YAM-9
satellite. A scientist writes a prompt, uploads it, and the satellite answers in text instead of downlinking raw imagery.

## Setup as reported
- Satellite: Loft Orbital YAM-9. Compute: Nvidia Jetson Orin AGX ran the model during live capture tests.
- Model: Gemma 3 4B in a "compressed, 4-bit format", "requires only 8 gigabytes of memory", "no modifications required".
- Software: NAVI-Orbital, with a LangGraph-based conductor. Deliberately walled off from flight software: it "reads images and produces descriptions" with no access beyond that.
- Power: "between 150 and 500 watts, depending on the position of the satellite" (my reading: a spacecraft-level budget, not the Orin's draw; the article does not say).
- YAM-9 also carries "several radiation-hardened processors (FPGAs, CPUs, and GPUs)". No detail on what the FPGAs do.

## Numbers
| Claim | Value |
|---|---|
| Classification accuracy | 88 percent on a 7,960-image benchmark, no fine-tuning |
| Live tests | two captures, over Toulouse and the Argentina coast |
| Current wildfire detection delay (context) | up to 90 minutes, "limits in downlink bandwidth and data processing" |

## Core questions
- **Q1:** The only latency figure is the 90-minute wildfire delay of today's pipelines, given as motivation. No inference latency or deadline is reported for the in-orbit run.
- **Q2:** Nothing relevant.
- **Q3:** One signal only: FPGAs are present on the platform, role unstated. Nothing on determinism.

## Caveats
- No timing, power-draw or thermal numbers for the in-orbit run. The primary paper has more (26.6 s ground cycle time, 20 s capture cadence), see its note.
- Quotes are promotional in tone ("a major shift", "tiny, tiny computer"); do not cite them as measurements.
- Accuracy is on a ground benchmark, not on the in-orbit captures.

## Quotes worth keeping
- Lasserre (Loft): "It doesn't matter if the link is slow, because you're downlinking dozens of kilobytes instead of dozens or hundreds of megabytes."
- Article text: satellites "are currently capable of detecting wildfires, but limits in downlink bandwidth and data processing can delay results by up to 90 minutes."

## Use on the slide
Fine as a pointer to the Loft deployment. For numbers, cite the NAVI-Orbital paper (arXiv 2606.18271), not this article.
