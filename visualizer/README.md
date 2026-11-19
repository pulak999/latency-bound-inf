# visualizer

Kernel-level profiles of a served frame, saved so we can come back to them.

Status: **starter code.** It works end to end on the Thor (capture, bundle, serve), but the viewer
is a first version. What's still to do is in the visualizer roadmap issue, which comes after
the harness roadmap (#16).

## The idea

A frame that misses its deadline is a few thousand kernels on two CUDA streams. To say *why* it
missed, we need to see where the time went: which kernels, how long, where the GPU sat idle
("bubbles", after Hazy Research's [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
and [TP-Llama](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) posts), and later what
the warps were stalled on. Every capture is saved as a **bundle**, with notes on what we concluded,
so we can compare alone vs. partitioned vs. time-sliced, or before vs. after a change.

## Use

```bash
# capture 5 frames of one fixed tile, alone (model load and warm-up are not traced)
uv run python -m visualizer.capture configs/space_multitenancy.toml --name alone --notes "why"

# same, with neighbours and GPU sharing taken from a cell of that config
uv run python -m visualizer.capture configs/space_multitenancy.toml --name part-3 \
    --cell now-n3-saturating-partitioned

# extra llama-server flags
uv run python -m visualizer.capture configs/space_multitenancy.toml --name fa-off \
    --server-arg=--flash-attn --server-arg=off

# browse every saved bundle (then open http://127.0.0.1:8765)
uv run python -m visualizer.serve

# rebuild a bundle after changing bundle.py or adding ncu results
uv run python -m visualizer.bundle /data/projects/rohan/profiles/<bundle id>

# warp-level detail for the top kernels (needs root; starter, not yet run)
sudo -E visualizer/scripts/ncu_top_kernels.sh /data/projects/rohan/profiles/<bundle id>
```

Bundles live in `/data/projects/rohan/profiles/<YYYYmmdd-HHMMSS>-<name>/` (heavy, not in git):
`profile.nsys-rep` and `profile.sqlite` (raw, never changed), `frames.jsonl` (client side),
`meta.json`, `server.log`, `bundle.json` (what the viewer reads), `notes.md` (ours), and
`ncu/` once Nsight Compute has run.

## Files

| file | does |
|---|---|
| `capture.py` | runs llama-server under `nsys launch`, warms up, traces N frames, exports, bundles |
| `bundle.py` | nsys SQLite (+ ncu CSVs) → `bundle.json`: frames, kernels (columnar), families, streams |
| `serve.py` | local server: lists bundles, serves them, saves notes |
| `viewer/` | the page: library, frame breakdown, kernel timeline with idle bubbles, top kernels, notes |
| `scripts/ncu_top_kernels.sh` | Nsight Compute on a bundle's top kernels (root) |

## What to trust

- **Measured** (nsys/CUPTI): kernel start/end, grid, block, registers, shared memory, stream, memcpys.
- **Modeled**: blocks per SM and waves come from an occupancy model (warps, registers, shared
  memory, block limit), not a counter. The viewer says so.
- **Labelled by rule**: the stream that runs the patch embedding (`im2col`) is the vision encoder;
  kernel families come from name patterns in `bundle.py`.
- Frames are matched to the client's requests by order, not by timestamp.
