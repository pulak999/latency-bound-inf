# Experiment design: what the raw sources say (2026-10-19)

Read from the sources themselves, not from our notes: the VLASH code
(github.com/mit-han-lab/vlash, shallow clone of main), the Space Microdatacenters
paper (MICRO '23, passat.crhc.illinois.edu/micro23.pdf), and the Thor numbers
already cited in the VLA-Perf deck (Jetson AI Lab openpi-on-Thor page, ApxInf v0.1).
The Thor itself was not reachable over SSH today, so nothing here is measured by us.

## VLASH (robot manipulation)

- **What Table II's L is.** `benchmarks/benchmark_inference_latency.py` times
  `policy.predict_action_chunk(batch)` between two `torch.cuda.synchronize()` calls,
  back to back, on batches already moved to the GPU. 10 warm-up steps, 100 samples.
  So L is model only: no camera capture, no preprocessing, no host-to-device copy,
  no queueing. This answers the open question in `usecases-notes/vlash.md`.
- **Async is not a second thread.** `vlash/run.py` `VLASHAsyncManager` calls
  `launch_next_inference` inside the control loop. The CUDA kernels are queued and
  the loop keeps sending actions. The only wait is `self.next_chunk.cpu()` at the
  chunk switch.
- **Deadline.** `examples/inference/async.yaml`: 30 Hz loop, `n_action_steps: 32`,
  `inference_overlap_steps: 4`. Inference launches at step 28 and must finish by
  step 32: 4 / 30 Hz = **133 ms from launch**.
- **What a miss does.** The `.cpu()` blocks the control loop, so the robot gets no
  new action until inference ends. `busy_wait(1/fps - elapsed)` never skips steps,
  so every later step also runs late. A miss is a control-loop stall, not a dropped
  result.
- **Output size is already fixed.** pi0.5 config: `chunk_size: 50`,
  `num_inference_steps: 10` (fixed denoising loop), `image_resolution: (224, 224)`.
- **One input that is not fixed.** `prepare_language(batch, pad_to_max_length=False)`
  with `tokenizer_max_length: 200`, so prompt length follows the task string.
  Fix the task string, or pad to max length.
- Paper Table II uses a different config (K=25 at 50 Hz, 1 image), so its
  23 to 32 ms numbers are for desktop GPUs and not our Thor anchor.

## Thor anchor for pi0.5 (from the VLA-Perf deck refs)

- pi0.5, 2 views, BF16 on Thor: **132 to 166 ms** measured (Jetson AI Lab openpi
  tutorial; ApxInf v0.1 release notes).
- FP8 on Thor: about 41 to 54 ms (TensorRT FP8 about 54 ms, ApxInf FP8 41 ms).
- BF16 sits right on VLASH's default 133 ms deadline, so precision alone should move
  the miss rate from about half to about zero. This is a good first knob.

## Space Microdatacenters (MICRO '23, image processing in orbit)

- **Arrival rate.** "We assume a ground track frame period of 1.5 s, meaning each
  satellite in the constellation generates one image every 1.5 s" (Sec. 5). At 3 m
  a frame is one 4K RGB image (Sec. 4).
- **Many satellites feed one SµDC.** At 3 m and a 1 Gbit/s inter-satellite link, each
  link carries "over four images every 1.5 s", and with two receivers one SµDC
  serves up to nine EO satellites (Sec. 8, Table 8). That gives us an arrival-rate
  knob: 1 to 9 satellites.
- **The models are CNNs, not VLMs.** Table 5: Inception-ResNet, Inception v3,
  DenseNet, MobileNet v3 (Urban Emergency Detection), EfficientNet, VGG19,
  Mask R-CNN, a 4-layer CNN, k-means, a DSP kernel.
- **Only one app is framed as latency sensitive.** Urban Emergency Detection:
  "Processing in space enables low latency detection, an important metric for this
  application" (Sec. 4). Sec. 9 says many EO apps "do not have stringent latency
  requirements", so energy efficiency matters more for them.
- **How they measured.** Means of 100 runs at the batch size that maximizes pixels
  per second per watt; Jetson AGX Xavier with tegrastats; RTX 3090 with pynvml.
  Table 6 times are per batch, not per image. There are no percentiles and no deadlines.
- **What this means for us.** The paper gives the timing (1.5 s per frame, 1 to 9
  feeding satellites) but not the model. VLM triage comes from NAVI-Orbital
  (Gemma 3 4B, 26.6 s per image on a Jetson-class board), which is about 18x
  over a 1.5 s frame period. On Thor, we need a one-token label and a fixed image
  size to have a chance. A MobileNet v3 (UED) run gives the CNN comparison point
  Harry asked for.

## Decisions (2026-10-19)

1. Serving stacks: **openpi** for pi0.5, **vLLM** for Gemma 3.
2. Simulator location: **to be investigated** (2026-10-19, was "separate Linux machine").
   Whatever runs it, the send time (1) and the back-at-sim time (5) should be stamped
   on the same clock.
6. **Synthetic inputs for now.** Image size and output length are fixed, so image
   content can't change the GPU's work. Three conditions:
   - a new image for every request, or vLLM's prefix and image caches would serve
     repeats and we'd be timing cache hits;
   - random pixels, not blank frames, because GPU power can depend on the data and
     the power limit sets clock speed (unverified, covered by the real-vs-synthetic check);
   - send images uncompressed, or decode time would depend on content.
   We lose task outcomes (does a stall make the robot fail?). That needs LIBERO
   closed-loop later, and only for showing what a miss costs.
   With synthetic inputs the "sim" is just a request sender on a fixed schedule.
7. Timestamps are now five, not seven: 1 sent, 2 arrives, 3 GPU starts, 4 GPU done,
   5 back at sim. The gaps are travel, wait, run, travel back.
3. Late space images: **keep them, let the queue grow.** vLLM with `--max-num-seqs 1`
   as the baseline, vLLM's normal batching as a separate run.
4. **Transformers only:** Gemma 3 4B (space) and pi0.5 (robot). No MobileNet.
5. Robot deadline: **LIBERO's native 20 Hz**. Start the next inference k = 2, 3 or 5
   steps before the chunk ends (100, 150, 250 ms).

## openpi and LIBERO, read from the code (openpi and LIBERO main, shallow clones)

- **Same stack as our Thor anchor.** Jetson AI Lab "OpenPi on Thor" (JetPack 7.2,
  MAXN, `pi05_libero`, action horizon 10): PyTorch BF16 "132.00 ± 0.89 ms" total
  (about 128 ms model), TensorRT FP8 about 54 ms, TensorRT FP8+NVFP4 "48.84 ± 0.16 ms".
  Served by `scripts/serve_policy.py` over a WebSocket on port 8000. The 166 ms
  ApxInf number is a different stack, so drop it as an anchor.
- **The model alone is already steady.** ±0.89 ms is on a quiet machine with the model
  only, which fits Asaf's point. What we add is the loop, the queue and co-located work.
- **openpi's LIBERO example is synchronous.** `examples/libero/main.py` calls
  `client.infer(...)` and the sim waits. It replans every `replan_steps: 5` of the
  10-action chunk. The real-time, launch-early loop is ours to write.
- **LIBERO runs at 20 Hz.** `control_freq=20` in `libero/libero/envs/bddl_base_domain.py`.
  So a deadline of k steps is k / 20 s (2 steps = 100 ms, 4 = 200 ms, 5 = 250 ms), not
  VLASH's 4 / 30 Hz = 133 ms.
- **Images are resized on the client.** The sim renders at 256 x 256 (`LIBERO_ENV_RESOLUTION`)
  and the client calls `image_tools.resize_with_pad(..., 224, 224)` before sending.
  That resize happens on the sim host, not the Thor.
- **Prompts are padded.** `PaligemmaTokenizer.tokenize` pads to `max_len`, so prompt length is
  fixed by openpi. The TensorRT engine uses a fixed 208 language tokens.
- `sample_actions(..., num_steps=10)`: 10 denoising steps by default.

## vLLM on Thor

- NVIDIA ships a vLLM container for Thor (NVIDIA forum announcement). The Jetson AI Lab
  Gemma 3 4B page serves a W4A16 checkpoint from an Orin-tagged vLLM image. Nothing
  found confirms Gemma 3 image input works in the Thor container, so test that first.
- Queue metrics: `vllm:num_requests_waiting`, `vllm:num_requests_running`,
  `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`,
  `vllm:e2e_request_latency_seconds` (histograms, not per request).
- One-token output: `max_tokens=1` plus `allowed_token_ids` limited to two single-token labels.

## Still to decide

1. Serving stack on Thor: VLASH (PyTorch, torch.compile) or openpi (JAX, what the
   Jetson AI Lab numbers use) for pi0.5. llama.cpp, TensorRT-LLM or vLLM for Gemma 3.
2. Simulator on the Thor (its rendering becomes co-located GPU work) or on a separate
   host (then clock sync matters only for measuring transport).
3. Space miss policy: count late labels and keep the queue, or drop frames past
   the deadline.
4. Space model set: Gemma 3 4B one-token label only, or also MobileNet v3.
