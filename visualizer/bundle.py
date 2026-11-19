"""Turn one capture into `bundle.json`: everything the viewer needs, nothing it doesn't.

    uv run python -m visualizer.bundle <bundle dir>

Reads `profile.sqlite` (the nsys export), `frames.jsonl` (the client's view of
each frame) and `meta.json`, plus `ncu/*.csv` if Nsight Compute has been run on
the bundle (visualizer/scripts/ncu_top_kernels.sh). Rebuilding is safe: the raw
files are never changed.

What goes in:

- gpu: name, SMs, limits per SM (for the wave model), memory bandwidth.
- frames: client latency, engine time, GPU span, kernel time, idle gaps,
  time per kernel family and per stream.
- kernels: one row per kernel launch, columnar, times in µs from the frame's
  first kernel. Blocks per SM come from a simple occupancy model (registers,
  shared memory, warps, block limit), so waves = blocks / (SMs x blocks per SM)
  is a model, not a measurement. It is labelled as such in the viewer.
- names: short name, full name and family for each kernel name.
- streams: what runs on each stream. The stream that runs the patch embedding
  (`im2col`) is the vision encoder; the others are the language model.
- memcpy rows, and GPU metrics if the capture had them (needs root).
- ncu: per kernel name, the raw metrics and stall reasons, if present.
"""

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

FRAME_GAP_NS = 300_000_000  # kernels further apart than this belong to different frames

# Kernel families, first match wins. "copy / convert" is pure data movement on the GPU.
FAMILIES = [
    ("attention", r"flash_attn|attn|soft_?max"),
    ("matmul", r"mul_mat|gemm|cutlass|cublas|xmma|^Kernel2$"),
    ("normalization", r"norm"),
    ("copy / convert", r"cpy|convert|set_rows|get_rows|im2col|transpose|concat|dup|cont"),
    ("quantize", r"quantize"),
    ("elementwise", r"bin_bcast|unary|scale|add|mul|gelu|silu|rope|clamp|pad"),
]


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name):
            return label
    return "other"


def blocks_per_sm(gpu: dict, threads: int, regs: int, smem: int) -> int:
    """Resident blocks per SM from warps, registers, shared memory and the block limit."""
    warps = math.ceil(threads / gpu["threads_per_warp"])
    limits = [gpu["max_blocks_per_sm"], gpu["max_warps_per_sm"] // max(1, warps)]
    if regs:
        regs_per_warp = math.ceil(regs * gpu["threads_per_warp"] / 256) * 256
        limits.append(gpu["max_registers_per_sm"] // (regs_per_warp * warps))
    if smem:
        limits.append(gpu["max_shmem_per_sm"] // (smem + 1024))  # 1 KB reserved per block
    return max(1, min(limits))


def read_gpu(db: sqlite3.Connection) -> dict:
    row = db.execute(
        """select name, smCount, clockRate, memoryBandwidth, l2CacheSize, maxWarpsPerSm,
           maxRegistersPerSm, maxShmemPerSm, maxBlocksPerSm, threadsPerWarp, computeMajor,
           computeMinor from TARGET_INFO_GPU limit 1"""
    ).fetchone()
    keys = [
        "name", "sms", "clock_hz", "memory_bandwidth_bps", "l2_bytes", "max_warps_per_sm",
        "max_registers_per_sm", "max_shmem_per_sm", "max_blocks_per_sm", "threads_per_warp",
        "cc_major", "cc_minor",
    ]  # fmt: skip
    return dict(zip(keys, row, strict=True))


def read_kernels(db: sqlite3.Connection) -> list[tuple]:
    return db.execute(
        """select k.start, k.end, s.value, d.value, k.streamId, k.graphId,
                  k.gridX * k.gridY * k.gridZ, k.blockX * k.blockY * k.blockZ,
                  k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory
           from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = k.shortName
           join StringIds d on d.id = k.demangledName
           order by k.start"""
    ).fetchall()


def read_memcpy(db: sqlite3.Connection) -> list[tuple]:
    return db.execute(
        "select start, end, bytes, copyKind, streamId "
        "from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
    ).fetchall()


def read_gpu_metrics(db: sqlite3.Connection) -> dict | None:
    """Sampled GPU metrics (SM active, DRAM bandwidth, ...) if the capture recorded them."""
    tables = {r[0] for r in db.execute("select name from sqlite_master where type = 'table'")}
    if "GPU_METRICS" not in tables:
        return None
    names = dict(db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
    series: dict[str, list] = defaultdict(list)
    for ts, metric, value in db.execute("select timestamp, metricId, value from GPU_METRICS"):
        series[names.get(metric, str(metric))].append((ts, value))
    return dict(series)


def split_frames(kernels: list[tuple]) -> list[list[tuple]]:
    frames, current = [], [kernels[0]]
    for previous, kernel in zip(kernels, kernels[1:], strict=False):
        if kernel[0] - previous[1] > FRAME_GAP_NS:
            frames.append(current)
            current = []
        current.append(kernel)
    frames.append(current)
    return frames


def label_streams(kernels: list[tuple]) -> dict[int, str]:
    streams = {k[4] for k in kernels}
    vision = {k[4] for k in kernels if k[2].startswith("im2col")}
    return {s: ("vision encoder" if s in vision else "language model") for s in streams}


def idle_ns(frame: list[tuple]) -> int:
    """Time inside the frame's span when no kernel was running on any stream."""
    busy_until, idle = frame[0][0], 0
    for kernel in frame:
        if kernel[0] > busy_until:
            idle += kernel[0] - busy_until
        busy_until = max(busy_until, kernel[1])
    return idle


def read_ncu(bundle_dir: Path) -> dict:
    """ncu results per kernel name, from the raw-page CSVs written by ncu_top_kernels.sh."""
    results = {}
    for path in sorted((bundle_dir / "ncu").glob("*.raw.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            continue
        units, data = rows[0], rows[1:]  # ncu puts the units in the first data row
        name = data[0].get("Kernel Name", path.stem)
        metrics = {}
        for key, value in data[0].items():
            try:
                metrics[key] = {"value": float(value.replace(",", "")), "unit": units.get(key, "")}
            except (ValueError, AttributeError):
                continue
        stalls = {
            key.removeprefix("smsp__pcsamp_warps_issue_stalled_").removesuffix(".sum"): m["value"]
            for key, m in metrics.items()
            if key.startswith("smsp__pcsamp_warps_issue_stalled_") and key.endswith(".sum")
            and not key.endswith("_not_issued.sum")
        }  # fmt: skip
        results[name] = {"instances": len(data), "metrics": metrics, "stalls": stalls}
    return results


def build(bundle_dir: Path) -> dict:
    db = sqlite3.connect(bundle_dir / "profile.sqlite")
    gpu = read_gpu(db)
    kernels = read_kernels(db)
    memcpy = read_memcpy(db)
    client = [json.loads(line) for line in (bundle_dir / "frames.jsonl").read_text().splitlines()]
    meta = json.loads((bundle_dir / "meta.json").read_text())
    streams = label_streams(kernels)

    names: dict[str, int] = {}
    name_rows = []
    frames_out = []
    columns: dict[str, list] = defaultdict(list)
    for index, frame in enumerate(split_frames(kernels)):
        t0, t_end = frame[0][0], max(k[1] for k in frame)
        by_family: dict[str, float] = defaultdict(float)
        by_stream: dict[str, float] = defaultdict(float)
        for start, end, short, full, stream, graph, blocks, threads, regs, smem in frame:
            if short not in names:
                names[short] = len(name_rows)
                name_rows.append({"short": short, "full": full[:300], "family": family(short)})
            bps = blocks_per_sm(gpu, threads, regs, smem)
            columns["frame"].append(index)
            columns["start_us"].append(round((start - t0) / 1e3, 2))
            columns["dur_us"].append(round((end - start) / 1e3, 2))
            columns["name"].append(names[short])
            columns["stream"].append(stream)
            columns["graph"].append(graph)
            columns["blocks"].append(blocks)
            columns["threads"].append(threads)
            columns["regs"].append(regs)
            columns["smem"].append(smem)
            columns["blocks_per_sm"].append(bps)
            by_family[family(short)] += (end - start) / 1e6
            by_stream[streams[stream]] += (end - start) / 1e6
        side = client[index] if index < len(client) else {}
        frames_out.append(
            {
                "index": index,
                "client_latency_ms": side.get("latency_ms"),
                "engine_prompt_ms": side.get("prompt_ms"),
                "gpu_span_ms": round((t_end - t0) / 1e6, 2),
                "kernel_ms": round(sum(k[1] - k[0] for k in frame) / 1e6, 2),
                "idle_ms": round(idle_ns(frame) / 1e6, 2),
                "kernels": len(frame),
                "by_family_ms": {k: round(v, 2) for k, v in sorted(by_family.items())},
                "by_stream_ms": {k: round(v, 2) for k, v in sorted(by_stream.items())},
                "t0_ns": t0,
            }
        )

    frame_t0 = [f["t0_ns"] for f in frames_out]
    memcpy_rows = []
    for start, end, size, kind, stream in memcpy:
        index = max((i for i, t in enumerate(frame_t0) if t <= start), default=None)
        if index is not None:
            memcpy_rows.append([index, round((start - frame_t0[index]) / 1e3, 2),
                                round((end - start) / 1e3, 2), size, kind, stream])  # fmt: skip

    warnings = []
    if len(frames_out) != len(client):
        warnings.append(f"{len(frames_out)} frames in the trace, {len(client)} sent by the client")
    bundle = {
        "format": 1,
        "meta": meta,
        "gpu": gpu,
        "streams": {str(s): label for s, label in streams.items()},
        "names": name_rows,
        "frames": frames_out,
        "kernels": columns,
        "memcpy": memcpy_rows,
        "gpu_metrics": read_gpu_metrics(db),
        "ncu": read_ncu(bundle_dir),
        "warnings": warnings,
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    bundle = build(args.bundle_dir)
    for frame in bundle["frames"]:
        print(
            f"frame {frame['index']}: {frame['kernels']} kernels, "
            f"GPU span {frame['gpu_span_ms']} ms, idle {frame['idle_ms']} ms, "
            f"client {frame['client_latency_ms']:.0f} ms"
        )
    for warning in bundle["warnings"]:
        print("warning:", warning)


if __name__ == "__main__":
    main()
