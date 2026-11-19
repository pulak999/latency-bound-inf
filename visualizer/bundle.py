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


TIME_TO_US = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3}
HZ_TO_MHZ = {"hz": 1e-6, "Hz": 1e-6, "khz": 1e-3, "Mhz": 1.0, "MHz": 1.0, "Ghz": 1e3, "GHz": 1e3}
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
PIPE_METRIC = re.compile(r"^sm__inst_executed_pipe_([a-z_]+)\.avg\.pct_of_peak_sustained_active$")


def summarize_ncu(header: list[str], units: list[str], data: list[list[str]]) -> dict:
    """The few ncu numbers the viewer shows, averaged over the profiled launches.

    Stall reasons are shares of all warp samples (each sample is one warp at one
    moment, and the reason it could not issue, or `selected` if it issued).
    """
    column = {name: i for i, name in enumerate(header)}

    def values(name: str) -> list[float]:
        if name not in column:
            return []
        out = []
        for row in data:
            try:
                out.append(float(row[column[name]].replace(",", "")))
            except ValueError:
                pass
        return out

    def mean(name: str, scale: float = 1.0) -> float | None:
        xs = values(name)
        return round(sum(xs) / len(xs) * scale, 3) if xs else None

    def unit(name: str) -> str:
        return units[column[name]] if name in column else ""

    stalls = {
        name.removeprefix(STALL_PREFIX): sum(values(name))
        for name in header
        if name.startswith(STALL_PREFIX) and not name.endswith("_not_issued")
    }
    samples = sum(stalls.values()) or 1.0
    pipes = {m.group(1): mean(name) for name in header if (m := PIPE_METRIC.match(name))}
    l2_sectors = mean("lts__t_sectors.sum")
    return {
        "launches": len(data),
        "grid_blocks": values("launch__grid_size"),
        "time_us": mean(
            "gpu__time_duration.sum", TIME_TO_US.get(unit("gpu__time_duration.sum"), 1)
        ),
        "clock_mhz": mean(
            "gpc__cycles_elapsed.avg.per_second",
            HZ_TO_MHZ.get(unit("gpc__cycles_elapsed.avg.per_second"), 1),
        ),
        "sm_pct": mean("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "memory_pct": mean("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        "occupancy_pct": mean("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "theoretical_occupancy_pct": mean("sm__maximum_warps_per_active_cycle_pct"),
        "waves": mean("launch__waves_per_multiprocessor"),
        "occupancy_limits": {
            limit: mean(f"launch__occupancy_limit_{limit}")
            for limit in ("blocks", "registers", "shared_mem", "warps")
        },
        "issue_active_pct": mean("smsp__issue_active.avg.pct_of_peak_sustained_active"),
        "warp_cycles_per_instruction": mean("smsp__average_warp_latency_per_inst_issued.ratio"),
        "l1_hit_pct": mean("l1tex__t_sector_hit_rate.pct"),
        "l2_hit_pct": mean("lts__t_sector_hit_rate.pct"),
        "l2_mb": round(l2_sectors * 32 / 1e6, 1) if l2_sectors is not None else None,
        "tensor_hmma_pct": mean(
            "sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_active"
        ),
        "pipes_pct": dict(sorted(((p, v) for p, v in pipes.items() if v), key=lambda x: -x[1])),
        "stalls_pct": {
            reason: round(100 * count / samples, 1)
            for reason, count in sorted(stalls.items(), key=lambda x: -x[1])
            if count
        },
    }


def read_ncu(bundle_dir: Path) -> dict:
    """ncu summaries per kernel short name, from the raw-page CSVs of ncu_top_kernels.sh.

    The raw CSVs and .ncu-rep files stay next to the bundle; only the summary goes in.
    """
    results = {}
    for path in sorted((bundle_dir / "ncu").glob("*.raw.csv")):
        with path.open() as f:
            rows = list(csv.reader(f))
        if len(rows) < 3:  # header, units, at least one launch
            continue
        header, units, data = rows[0], rows[1], rows[2:]
        summary = summarize_ncu(header, units, data)
        summary["kernel"] = data[0][header.index("Kernel Name")][:300]
        results[path.name.removesuffix(".raw.csv")] = summary
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
