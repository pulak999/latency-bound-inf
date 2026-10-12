# System setup v1: quantized VLM and VLA inference on the Jetson AGX Thor (2026-10-09)

Goal: get the two use cases from the kickoff slides running on the Thor with the runtimes the industry
actually deploys, so that slide 6 (W, NM, M, D) and slide 7 (the first experiment) have real software behind them.
Stack A is the space image-triage VLM. Stack B is the robot-manipulation VLA.
Every engine below is a point on the M axis ("same model, different runtime").

Facts were checked on the box on 2026-10-09. Anything marked **verify** was not tested.

## 1. The machine

| Item | State |
|---|---|
| Board | NVIDIA Jetson AGX Thor Developer Kit, JetPack 7 (L4T R39.2.1), CUDA 13.2.2, driver 595.78 |
| GPU | Blackwell, compute capability 11.0 (`sm_110`). `nvidia-smi` works. MIG shown as "Disabled", so Thor may support MIG partitioning (**verify**, needs root) |
| Memory | 122 GB unified. 773 GB free on the NVMe. `/data` is group-writable (`thorusers`), `~/data` symlinks to it |
| OS / Python | Ubuntu 24.04, Python 3.12.3, pip 24.0, cmake, gcc, git. No `uv`, no PyTorch in the default environment |
| Containers | Docker 28 with the NVIDIA runtime. User is in the `docker` group |
| Power modes | `nvpmodel` IDs: 0 MAXN, 1 120 W (default), 2 90 W, 3 70 W. Changing them needs sudo |
| Network | PyPI, download.pytorch.org, Hugging Face, GitHub reachable. NGC pulls need `docker login nvcr.io` |

**PyTorch works from standard wheels.** A labmate's venv has `torch 2.10.0+cu130` installed with plain pip, and its
arch list includes `sm_110`. CUDA 13.2 unified the Arm SBSA toolkit with Jetson, so the `cu130` aarch64 wheels on
download.pytorch.org are the right ones. No Jetson-specific pip index is needed (the Jetson AI Lab index has no `jp7` stage today).

**No ready-made JetPack 7 containers from dusty-nv.** Docker Hub has no `r39` or `cu130` tags for `dustynv/vllm`,
`dustynv/llama_cpp`, `dustynv/sglang` or `dustynv/pytorch`. jetson-containers says it supports JetPack 7 but you
would build the image yourself. NVIDIA publishes its own Thor containers on NGC (**verify** the current tags on the Jetson AI Lab Thor page).

### Two blockers to clear first

1. **The GPU is shared and busy.** On 2026-10-09 two labmates had jobs on it (an `openwam` deploy server holding 17 GB, and a
   LIBERO eval in two processes), with utilization at 97%. Any latency measured while they run is contaminated.
   Proposal: a file `/data/gpu-lock` containing owner, purpose and ETA, plus a Slack post in `#latency-bound-inf`.
   The harness (section 5) records `nvidia-smi --query-compute-apps` before every run so co-tenants are in the metadata either way.
   Their jobs are a real "NM" load, but only useful once we can switch it on and off.
2. **No sudo.** Power-mode sweeps (`nvpmodel -m`), clock pinning (`jetson_clocks`) and MIG all need root.
   Ask the admin for sudo on those three commands only, or for a booked window. `tegrastats` runs without root.

### Conventions

- Everything under `/data/projects/pulak/`: `models/`, `envs/`, `src/`, `runs/`.
- One environment per stack, never shared. `uv` for Python envs (standalone installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`; system pip is PEP 668 locked), Docker only where NVIDIA ships the image.
- Torch pin for every venv: `torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130` (proven on this box).
- Hugging Face token in `~/.cache/huggingface/token` (`hf auth login`). Gemma 3 and the pi checkpoints are gated or large.
- Every run writes one JSONL file of per-request records plus a `tegrastats` log (section 5).

## 2. Stack A: VLM for space image triage

Target model: Gemma 3 4B instruct, 4-bit, one label per image. Same model NAVI-Orbital flew on YAM-9.

### A1. llama.cpp with Google's QAT Q4_0 GGUF (the flight stack, do this first)

NAVI-Orbital ran Gemma 3 4B as a Q4_0 GGUF through llama.cpp with the multimodal projector. Google publishes a
quantization-aware-trained Q4_0 GGUF with its own projector file, which is the closest public match.
Single process, no batching scheduler, so the latency recorded is the engine's alone.

```bash
mkdir -p /data/projects/pulak/{models,envs,src,runs}
cd /data/projects/pulak/src
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 -DLLAMA_CURL=OFF   # no libcurl-dev on the box
cmake --build build -j"$(nproc)" --target llama-mtmd-cli llama-server   # about 10 to 20 min on Thor

uv tool install huggingface_hub && hf auth login
hf download google/gemma-3-4b-it-qat-q4_0-gguf \
   --local-dir /data/projects/pulak/models/gemma-3-4b-it-qat-q4_0
# files: gemma-3-4b-it-q4_0.gguf, mmproj-model-f16-4B.gguf

# smoke test: one image, one label
M=/data/projects/pulak/models/gemma-3-4b-it-qat-q4_0
./build/bin/llama-mtmd-cli -m $M/gemma-3-4b-it-q4_0.gguf --mmproj $M/mmproj-model-f16-4B.gguf \
   --image test.jpg -n 4 --temp 0 -ngl 99 \
   -p "Classify this satellite scene with one word from: forest, water, urban, cloud, field."

# for the harness: OpenAI-compatible server, single slot
./build/bin/llama-server -m $M/gemma-3-4b-it-q4_0.gguf --mmproj $M/mmproj-model-f16-4B.gguf \
   -ngl 99 --port 8081 --parallel 1 --no-cont-batching
```

Done when: a label comes back, and `tegrastats` shows the GPU rail moving during the request.
Also build the `Q4_K_M` and `Q8_0` files from `ggml-org/gemma-3-4b-it-GGUF` later for a quantization axis.

### A2. vLLM (the serving engine the industry uses, second)

vLLM runs Gemma 3 multimodal, supports AWQ and GPTQ 4-bit plus FP8 and NVFP4 (the native Blackwell formats NVIDIA pushes
on Thor through ModelOpt), and its `/metrics` endpoint already reports time to first token and inter-token time.
Its continuous-batching scheduler adds variance of its own, which is itself a Q2 finding.

Two routes, try in this order:
1. **NVIDIA's Thor vLLM container from NGC.** Log in to `nvcr.io` and pull the tag named on the Jetson AI Lab Thor page (**verify**).
   Run with `docker run --runtime nvidia --network host -v /data/projects/pulak/models:/models ...`.
2. **Build from source** in a venv against `torch 2.10.0+cu130` with `TORCH_CUDA_ARCH_LIST="11.0"`. The PyPI aarch64 wheel
   (`vllm 0.31.0`) exists but is compiled for datacenter arches, and whether it JITs onto `sm_110` is untested (**verify** before relying on it). Expect one to two hours of build.

Models: `google/gemma-3-4b-it` in bf16 as the reference, then an AWQ 4-bit export, then NVFP4 via `nvidia-modelopt`.
Serve with `--max-num-seqs 1` for the single-request regime so the comparison with A1 is fair.

### A3. TensorRT-LLM (optional, the most static engine)

Builds a fixed engine with CUDA graphs. Closest thing NVIDIA has to "determinism by construction" and therefore the right
GPU-side comparison for Q3. Heaviest setup on Jetson; jetson-containers has no `tensorrt_llm` package. Only after A1 and A2 give a baseline.
Thor-specific TRT-LLM support should be confirmed on NGC first (**verify**).

### A0. Naive baseline (cheap, optional)

Hugging Face `transformers` in eager mode, bf16, no quantization. This is the "naive PyTorch" column Jetson-PI compares against.
Skip bitsandbytes on aarch64 unless a wheel installs cleanly; it has historically needed a source build on Jetson.

## 3. Stack B: VLA for robot manipulation

Target models: pi0.5 (3B, bf16 weights about 7 GB) and SmolVLA (450M). Both fit on Thor without quantization,
so on this board quantization is about bandwidth (273 GB/s), not memory. That is a different argument from the Orin and goes in the notes.
Simulator: LIBERO, headless with EGL. Both openpi and LeRobot ship a LIBERO eval, which is what VLASH, Jetson-PI and the ETH comparison all use.

### B1. LeRobot (Hugging Face, do this first on the VLA side)

LeRobot 0.6.2 needs Python 3.12 (matches the box), torch 2.7 to 2.11, and has `pi0`, `pi05`, `smolvla` and `rtc` policies,
a LIBERO env (`hf-libero`, pure Python; `mujoco 3.15` has aarch64 wheels), and the async inference pair
`policy_server` / `robot_client` over gRPC that VLASH builds on.

```bash
cd /data/projects/pulak/envs
uv venv lerobot --python 3.12 && source lerobot/bin/activate
uv pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu130
uv pip install "lerobot[pi,smolvla,libero]" grpcio protobuf
# torchcodec on aarch64 needs torch>=2.11; if the resolver complains, pin torchcodec out or move to torch 2.11.

# sanity: LIBERO eval, SmolVLA, one task, headless
MUJOCO_GL=egl lerobot-eval --policy.path=lerobot/smolvla_base \
   --env.type=libero --env.task=libero_spatial --eval.n_episodes=2 --eval.batch_size=1 --policy.device=cuda

# async pair, the deployment pattern we want to instrument
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
# the client handshake names the policy: policy_type=pi05 or smolvla, actions_per_chunk, chunk_size_threshold
```

Done when: a LIBERO episode runs end to end on the GPU, and the policy server answers a client with an action chunk.
Checkpoints to pull: `lerobot/smolvla_base`, the LeRobot pi0.5 LIBERO fine-tune (name **verify** on the Hub), and later the VLASH LIBERO checkpoints.

### B2. openpi (Physical Intelligence)

openpi serves pi0 and pi0.5 through a websocket `serve_policy.py` and has a LIBERO example (`examples/libero/main.py`,
plus a Docker compose that runs server and eval together). Checkpoints come from `gs://openpi-assets` (`pi05_libero`, `pi05_droid`).
It pins Python 3.11, `torch==2.7.1` and `jax[cuda12]==0.5.3`. Neither pin knows `sm_110`, so on Thor:

- Use the **PyTorch path** (added Sept 2025, validated on LIBERO) and override torch to 2.10.0+cu130.
  openpi already uses `[tool.uv] override-dependencies`; add `torch==2.10.0` and `torchvision==0.25.0` there with the cu130 index as a `[tool.uv.index]`.
- Treat the **JAX path** as untested on Thor (**verify** if needed; a `cuda12` plugin on a CUDA 13 driver with a Blackwell part is two unknowns at once).
- The PyTorch models need `transformers_replace` copied into the venv's transformers (README step), and the JAX checkpoints converted once with `examples/convert_jax_model_to_pytorch.py`.

```bash
cd /data/projects/pulak/src
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi
cd openpi
# edit pyproject.toml: add torch/torchvision to override-dependencies, add the cu130 index
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir gs://openpi-assets/checkpoints/pi05_libero \
   --output_path /data/projects/pulak/models/pi05_libero_pt
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=/data/projects/pulak/models/pi05_libero_pt
MUJOCO_GL=egl uv run examples/libero/main.py      # separate terminal, talks to port 8000
```

openpi's LIBERO Docker compose expects an x86 image; run the server and client natively instead.
Done when: `pi05_libero` serves an action chunk and the LIBERO client completes an episode.

### B3. EagleVLA-Edge, formerly Jetson-PI-Edge (quantized pi0.5 on device)

The Jetson-PI engine is a llama.cpp fork (`PKU-SEC-Lab/EagleVLA-Edge`, branch `master`; same tree as llama.cpp plus
`vit_config_pi05.json`, `latency_optimize_docs/`, `.pi/`). It is the one open implementation of a quantized VLA with
CUDA-graph reuse, preallocated buffers and unrolled denoising on Jetson, which is exactly the Q3 "can the GPU borrow determinism" experiment.
Build it the same way as A1 (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110`); the README at `master` has the pi0.5 conversion and run steps (**verify**, the file was empty when fetched from `main`).
Its headline is a mean latency cut from 1420.8 to 476.1 ms on Orin with no tail reported, so the first thing to do with it is log every inference and plot the tail.

### B4. Method baselines (separate envs, later)

- **VLASH** (`mit-han-lab/vlash`): Python 3.10 conda recipe, built on an older LeRobot. Gives the sync, naive-async, RTC and VLASH policies on LIBERO. Its released loop is single-threaded and blocking, so it is a baseline for the runtime axis, not a runtime.
- **ETH async-vla-inference** (`TheAyos/async-vla-inference`): unified LIBERO (PyTorch/LeRobot 0.4.2) and Kinetix (JAX) code for IT-RTC, TT-RTC, VLASH, A2C2 with Wilson CIs. Use for the ML-method axis once the runtime measurements exist.

## 4. Mapping to the slide-6 stack

| Axis | Stack A (space triage) | Stack B (manipulation) |
|---|---|---|
| W, trace | Sentinel-2 tiles sequenced along a ground track, one image every T seconds, T in {20, 30} s to start | LIBERO observations at 30 Hz from the sim; later a recorded LeRobot dataset for replay mode |
| M, model | Gemma 3 4B: bf16 (A0/A2), QAT Q4_0 (A1), Q4_K_M and Q8_0 (A1), AWQ and NVFP4 (A2) | pi0.5 bf16 (B1, B2), SmolVLA (B1), pi0.5 quantized (B3) |
| M, engine | llama.cpp (A1), vLLM (A2), TensorRT-LLM (A3), eager transformers (A0) | LeRobot policy server (B1), openpi websocket server (B2), EagleVLA-Edge (B3), VLASH loop (B4) |
| NM, non-model load | Labmates' jobs (uncontrolled), then our own: a YOLO-class detector, a JPEG-2000 encoder, CPU memory scrubber | VIO/SLAM stand-in, image resize and normalize on GPU, a second policy instance (maritime UAV paper pattern) |
| D, deployment | On-device Thor at 120 W, 90 W, 70 W, MAXN | Same, plus a "connected" cell with the server on a Titan over `tc netem` later |
| Simulator | Own orbit loop (fixed-rate image feeder is enough for v1) | LIBERO, headless EGL, wall-clock wrapper so the sim does not wait for the policy |
| Output shape | One label, `-n 4`, temperature 0, so output length is fixed (Asaf's point) | One action chunk, fixed K, so the request is fixed-shape prefill plus denoising |

## 5. Harness, shared by both stacks

One Python script per stack, same record format. Keep it boring.

- **Feeder.** Fires a request every T seconds on a monotonic clock whether or not the previous one returned. Never waits for the model; that is the whole point.
- **Per-request record (JSONL):** `req_id, t_send, t_first_token, t_last_token, n_prompt_tokens, n_output_tokens, deadline, missed (bool), engine, model, quant, power_mode, co_tenants`.
  `co_tenants` is the `nvidia-smi --query-compute-apps=pid,process_name,used_memory` output at send time.
- **Device log.** `tegrastats --interval 100 --logfile runs/<run>/tegrastats.log` for GPU load, clocks, rails and temperature. Parse offline.
- **Engine-side timing.** llama-server returns `timings` in its response (prompt and generation ms); vLLM exposes `/metrics`; LeRobot's policy server logs inference time per chunk. Record both client-side and engine-side times so the gap (transport, queueing, Python) is visible. That gap is the Q2 breakdown the driving study says matters.
- **Reporting.** p50, p99, p99/p50, miss rate at the deadline, Wilson CI on the miss rate, and the CDF plot from slide 8. Energy per request from the rail trace.

## 6. Order of work

| Step | What | Done when |
|---|---|---|
| 0 | Clear the two blockers: GPU booking convention, sudo for `nvpmodel`/`jetson_clocks` | A one-hour window with the GPU idle and a power-mode change that takes effect |
| 1 | A1: llama.cpp + QAT Q4_0 Gemma 3 4B | 100 single-image requests logged at T = 30 s with the GPU otherwise idle, tail plotted |
| 2 | Harness v1 on A1: feeder, JSONL, tegrastats, miss-rate script | Same run reproduced from the log alone |
| 3 | A1 under NM: labmates' job or our own detector co-located, then at 70 W | Two more CDFs on the same axes as step 1. This is slide 7 |
| 4 | B1: LeRobot SmolVLA and pi0.5 on LIBERO, async policy server | One episode end to end, per-chunk latency logged |
| 5 | A2: vLLM on Thor, bf16 then AWQ then NVFP4 | Same 100-request run as step 1, engine axis comparison |
| 6 | B2: openpi pi05_libero on the PyTorch path | Same as step 4 through the websocket server |
| 7 | B3: EagleVLA-Edge quantized pi0.5 | Per-inference tail with and without graph reuse. This is the first Q3 experiment |
| 8 | A3, B4, MIG partitioning, Titan "connected" cell | As time allows |

Steps 1 to 3 are the first result for the group. Steps 4 and 6 can run in parallel with 5 if Rohan takes the VLA side.

## 7. Open items

- Which NGC tag is the current Thor vLLM container, and does NVIDIA ship TensorRT-LLM for Thor yet.
- Does the PyPI vLLM aarch64 wheel run on `sm_110` or does it need a source build.
- The Hub name of a LeRobot-format pi0.5 LIBERO checkpoint (openpi's is in its own format and needs conversion).
- Whether the EagleVLA-Edge README on `master` has the pi0.5 GGUF conversion steps, and which pi0.5 checkpoint it expects.
- Does MIG work on Thor and can it be enabled without a reboot. If yes, it is a GPU-side spatial-isolation knob for Q3.
- `torchcodec` on aarch64 wants torch 2.11; decide between torch 2.10 (proven here) and 2.11 (within LeRobot's range, untested here).
- Dataset and licence for the Sentinel-2 feeder; the satellite notes already carry this TODO.
