# Design doc: VLM deadline-miss harness on Jetson Thor

Oct 28, 2026 · @Rohan

## 1. Purpose

This document briefs an engineering agent on building a measurement harness for one experiment: how often a small vision-language model running on an NVIDIA Jetson Thor misses a per-frame deadline when used as an onboard image-triage classifier for a satellite, and what makes it miss. The harness is the first of two (the second targets a robot-manipulation policy) and is part of a thesis on latency-bound inference. Read this whole document before writing code.

### The problem in one paragraph

An Earth-observation satellite captures a new frame on a fixed cadence, every T seconds. An onboard VLM looks at each frame and decides keep or drop, so only worthwhile frames are queued for downlink. The world does not wait: if the label is not ready when the next frame lands, that frame was decided late or by default. Existing papers report mean inference latency on a GPU with nothing else running. Nobody has measured the per-frame latency distribution on a Jetson, with a realistic second workload sharing the GPU, and turned it into a deadline-miss rate. That is the gap this harness fills.

### Goals

1. Produce a per-frame latency distribution for a 4-bit Gemma 3 4B classifier on the Thor, under a fixed-rate frame stream, for several cadences T.
2. Repeat with a controlled co-running GPU process (light and heavy), and at a lower power mode, so the effect of sharing and of clocks on the tail is isolated.
3. Report, per setting: p50, p99, p99/p50, the miss rate at deadline D = T, the miss rate as a function of any deadline, energy per frame, and mean GPU clock.
4. Make every number reproducible from a config file and a log, with no manual steps between "run" and "plot".

### Non-goals

- Improving the model, the prompt, or the runtime. We measure what exists.
- Multi-token generation. The classifier answers with one token, so each request is pure prefill and time-to-first-token equals time-to-last-token. Decode behaviour is a separate experiment.
- Simulating an orbit, downlink windows, or ground contact. The only piece of the world we simulate is a fixed-rate frame source.
- Running on any device other than the Thor we have.

### Success criteria

The harness is done when a single command runs one cell of the matrix end to end, writes a log, and a second command turns that log into the four plots in section 5; when the dummy-model validation in section 4 passes; and when the first three real cells (model alone, three cadences) have been run and their numbers look sane to a human. A miss rate that is 0% or 100% in every cell is a sign the cadences are wrong, not that the harness is finished.

## 2. Starting point

### What already exists

A colleague has Gemma 3 inference running on the Thor and has done some initial testing. That code is the starting point, not something to replace. The runtime it uses (llama.cpp, vLLM, Ollama, a raw transformers script, or something else) is not known to the author of this document; the agent's first task is to find out.

We have SSH access to the Thor. Assume it is shared with other people and may be rebooted or busy; the harness must not assume it has the device to itself outside of a running cell, and must record enough about the device state that a run can be discarded if something else was happening.

### First tasks, in order, before any harness code

1. Inventory the existing inference setup. Report: the runtime and its version, how the model is loaded (file, quantization format, context length, image preprocessing), how a request is made (CLI, HTTP, Python call), what it returns, and whether it exposes per-request timing (for llama.cpp: `timings.prompt_ms`; for vLLM: request metrics; for others: nothing, in which case we wrap it). Also report JetPack version, CUDA version, the output of `nvpmodel -q`, and free memory with the model loaded.
2. Run ten requests by hand with one image each and a one-word prompt. Record the latency of each on the host clock. If the ten numbers are not within a factor of two of each other after the first warm-up request, say so; it changes what the harness has to control for.
3. Confirm the model can be made to answer with exactly one token. If the runtime cannot cap output at one token, or the model does not reliably emit a single word, stop and report; the prompt and the stopping rule are a human decision.
4. Write a short note with the findings and a proposed wrapping strategy, then wait for a human to confirm before building the model server.

### Constraints the agent must respect

- Do not modify the colleague's inference code in place. Wrap it, or copy it into the harness repository and change the copy, and say which you did.
- Do not change power modes, clocks, or fan settings except through the harness's own controller, which does so only at cell boundaries and restores the original mode at exit.
- Do not install system packages on the Thor without listing them first; prefer a Python virtual environment in the harness directory.
- Do not run long cells while a human is using the device. The controller takes a `--max-minutes` flag and must honour it.
- Keep every tuning constant (cadence, deadline, warm-up count, co-runner rate, image size) in one config file. Nothing numeric that affects results lives in code.

## 3. System design

Four processes on the Thor, one monotonic clock, one controller script. Each process is a separate OS process (not a thread) so that GPU contention behaves as it would between real applications.

### Components

**Emitter.** Loads a fixed set of pre-cut image tiles into memory at start, then publishes one tile every T seconds on the wall clock, scheduled against `time.monotonic_ns()` so drift does not accumulate. Each message carries a frame id, the tile id, and `t_obs`. It never waits for anything downstream. A tick that overruns by more than 10% of T is logged as an emitter fault; the frame is still sent.

**Mailbox.** A depth-1 slot between emitter and model server. A new frame replaces the one not yet taken. The emitter records when it overwrites; the overwritten frame is a miss with label `overwritten`. Implementation: ZeroMQ PUB/SUB with `CONFLATE=1` on the subscriber, or a one-element queue with replace semantics if ZeroMQ is unavailable. The choice is a config flag and the semantics must be identical.

**Model server.** Pulls the newest frame, stamps `t_start`, sends it to the existing inference setup with the fixed prompt and a one-token cap, stamps `t_end`, and emits a record. If the runtime reports its own prefill time, that is recorded as `prompt_ms`; otherwise the field is null and `t_end − t_start` is the only timing. The server runs one request at a time. It performs a configurable number of warm-up requests before signalling ready, and the controller discards those frames.

**Co-runner.** A separate process with its own CUDA context and its own fixed-rate loop. Three settings: `none`; `light`, a per-pixel band-ratio kernel over a 4K×3 array once per second (CuPy or PyTorch); `heavy`, a DenseNet-121 forward pass over a tiled 4K frame every 0.5 s (TensorRT if available, otherwise PyTorch; ImageNet weights are fine, the output is discarded). Each co-runner logs its own per-iteration latency so interference can be read in both directions. The controller starts it before a cell and stops it after.

**Sampler.** `tegrastats --interval 100`, parsed into rows on the same clock: GPU frequency, GPU load, the available power rails, memory used and total. The exact fields depend on the JetPack version; the parser must record the raw line alongside the parsed fields so nothing is lost if a field is misparsed.

**Controller.** Reads the config, iterates the run matrix, and for each cell: sets the power mode (if different from the current one), starts the sampler, starts the co-runner, starts the model server and waits for ready, starts the emitter, waits until the frame count is reached or `--max-minutes` elapses, stops everything in reverse order, and writes a cell manifest. It tags every log row with the cell id. On exit, including on error or Ctrl-C, it restores the power mode it found at start.

### Timing semantics

- Latency of a frame is `t_end − t_obs`: measured from arrival, not from when the model picked it up. Queueing counts.
- Waiting is `t_start − t_obs`. Prefill is `prompt_ms` when available, else `t_end − t_start`.
- A frame misses deadline D when `t_end − t_obs > D`, or when it was overwritten in the mailbox. The model server does not know D. The deadline is applied to the log afterwards, so one run gives the miss rate at every D. The default reported D is the cadence T.
- All timestamps are `time.monotonic_ns()` from the same host. A wall-clock timestamp is also recorded once per cell in the manifest for correlating with external events.

### Log schema

One JSONL file per cell for frames, one for sampler rows, one for co-runner iterations, plus a manifest. Fields in the frame log:

| Field | Type | Meaning |
| --- | --- | --- |
| cell\_id | string | Cell identifier from the controller |
| frame\_id | int | Sequence number from the emitter |
| tile\_id | string | Which image |
| t\_obs | int ns | Emitter timestamp |
| t\_start | int ns or null | Model server picked it up; null if overwritten |
| t\_end | int ns or null | Response received; null if overwritten |
| prompt\_ms | float or null | Runtime-reported prefill, if any |
| label | string or null | The one-token answer, verbatim |
| outcome | enum | `ok`, `overwritten`, `error` |
| error | string or null | Runtime error text if outcome is error |

The manifest records: config hash, git commit of the harness, runtime name and version, model file hash, power mode before and during, co-runner setting, cadence, warm-up count, frames requested and delivered, wall-clock start and end, and the raw `nvpmodel -q` and `jetson_release` output.

### Data

Tiles are Sentinel-2 L2A true-colour crops, resampled to 896×896 (Gemma 3's native image size) so every request has identical prefill shape. Target 1,000 tiles from a handful of scenes mixing cloud, land, and water, with the scene-classification (SCL) band saved per tile so that a cloud-fraction ground truth exists for a later accuracy check. Acquiring and cutting the tiles is a one-off script in the repository; the harness itself only reads a directory of PNGs and a CSV index. If Sentinel-2 download is blocked from the Thor, do the cutting elsewhere and copy the directory over.

## 4. Experiment plan

### Run matrix

| Axis | Values | Notes |
| --- | --- | --- |
| Cadence T | 20 s, 5 s, 1.5 s | 20 s is the cadence a flown Gemma-on-Orin system reported; 1.5 s is the frame period assumed in the space-microdatacenter literature and will likely miss |
| Co-runner | none, light, heavy | Section 3 |
| Power mode | MAXN, one lower mode | The lower mode is the lowest `nvpmodel` profile in which the model still loads; a human picks it after task 1 in section 2 |
| Runtime | the existing one; a second if time allows | Same weights precision in both |

Frames per cell: 500 at T = 1.5 s and 5 s, 200 at T = 20 s, after 20 warm-up frames that are discarded. That is 12.5 min, 42 min and 67 min per cell. With one runtime the matrix is 18 cells and about 12 hours of unattended time; run it overnight in the order below, not interleaved.

### Build order

Each step has a gate. Do not start the next step until the gate passes and the result has been reported.

1. **Harness with a dummy model.** The model server calls a stub that sleeps for a random duration drawn from a configured distribution (default: uniform 0.2 s to 2.0 s). Gate: over a 200-frame run at T = 1 s, (a) `t_obs` spacing has mean T and standard deviation under 1 ms; (b) the number of `overwritten` frames matches what the sleep distribution predicts within 10%; (c) the computed miss rate at D = 1 s matches the fraction of sleeps longer than 1 s within 2 percentage points; (d) sampler rows and frame rows interleave correctly when sorted by timestamp.
2. **Real model, no cadence.** Replace the stub with the wrapped inference setup. Run 50 tiles back to back. Gate: every answer is exactly one of the two expected words; agreement with the SCL cloud fraction is above 80%; the latency of requests 11 to 50 is within a factor of two of each other. If any part fails, report and stop; prompt and label choices are human decisions.
3. **Alone cells.** The three cadences at MAXN with no co-runner. Gate: a human looks at the three CDFs and confirms the cadences bracket the interesting region. If the miss rate at D = T is 0% in all three or 100% in all three, propose two new cadences and wait.
4. **Co-runners.** Build `light` and `heavy`, characterise each alone for five minutes with the sampler on (report mean GPU load, mean power, memory footprint, iteration latency), then run the six shared cells at MAXN.
5. **Power mode.** The nine cells at the lower mode.
6. **Second runtime**, only if steps 1 to 5 are done and a human says to.

### Analysis

A single script reads a cell directory and produces, per cell: the latency CDF with D = T marked; a table of p50, p99, p99/p50, miss rate at D = T, mean and max waiting time, mean and max prefill, energy per frame (power integrated from `t_start` to `t_end` using the sampler rows), and mean GPU frequency during the cell. Across cells it produces the miss-rate-versus-deadline curve for every setting on one axis, and a timeline for each cell with per-frame latency plotted against GPU frequency and co-runner iterations, so a slow frame can be matched to a clock drop or to co-runner activity by eye. All figures are regenerated from logs; no numbers are typed into the report by hand.

## 5. Deliverables

1. A repository containing the harness (emitter, mailbox, model server wrapper, co-runners, sampler, controller), the tile-preparation script, the analysis script, a config file with every tunable, and a README that gets a new person from SSH login to a running dummy cell in under 30 minutes.
2. The inventory note from section 2 and the per-gate reports from section 4, as short markdown files in the repository under `reports/`.
3. Logs and manifests for every cell run, under `runs/<cell_id>/`, never deleted or overwritten; a rerun gets a new cell id.
4. The four figures per cell and the cross-cell miss-rate curve, regenerated by the analysis script.
5. A one-page results summary written after step 5, stating the headline numbers and the two or three things that surprised you.

### Working conventions

- Report at every gate, even when it passes. A report is a few sentences and the numbers, not a narrative.
- When something in this document is wrong or impossible on the actual device, say so and propose the smallest change, rather than silently doing something else. The timing semantics and the log schema in section 3 are the parts most worth preserving; everything else is negotiable.
- Prefer boring technology. Python, ZeroMQ, JSONL, matplotlib. No frameworks for the harness itself.
- Keep the model server's critical path minimal: no logging I/O between `t_start` and `t_end`, no image decoding on that path if the runtime accepts raw bytes it can decode itself (record which).
- Commit after each gate with the gate's name in the commit message.
- Nothing in `runs/` goes into git. Record its location in the README.

### Open questions for humans

These are decisions, not tasks. Ask, do not guess.

1. Which runtime the colleague's setup uses, and whether we wrap it or re-host the same weights in one we control.
2. The exact prompt and the two label words. The proposal is "keep" and "drop" with the instruction to drop frames that are mostly cloud or open water; the model must be checked to actually emit those tokens.
3. Which lower `nvpmodel` profile to use.
4. Whether the Thor can reach the internet for Sentinel-2 downloads, or tiles are prepared elsewhere.
5. Whether a 4K-input DenseNet is the right heavy co-runner on this device, or whether its iteration time needs capping so it stays periodic rather than saturating.
6. Whether to add a bandwidth-only co-runner (a device-to-device memcpy loop) as a fourth setting after the first results, to separate memory contention from compute contention.
