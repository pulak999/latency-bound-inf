"""DESIGN.md section 2, tasks 2 and 3: a handful of requests by hand, before any cell.

    uv run python -m harness.probe configs/stack_a.toml [--n 10] [--tiles DIR]

Starts the engine from the config, sends each tile once with the config's
prompt, first with max_tokens = 1 (what the cells will do) and then with 4
tokens (to see what the model wanted to say), and prints per request: the
one-token answer verbatim, whether it is one of the configured labels, the
client latency, and the engine's prompt_ms. Ends with the spread of
latencies after the first request and a verdict on the one-token rule.
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from harness import config as config_module
from harness.config import TilesConfig
from harness.controller import make_engine
from harness.tiles import load_tiles


def expected_labels(tiles_dir: Path) -> dict[str, str]:
    """keep/drop per tile from the SCL fractions in index.csv, if present."""
    index = tiles_dir / "index.csv"
    if not index.exists():
        return {}
    with index.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or "cloud_fraction" not in rows[0]:
        return {}
    return {
        r["tile_id"]: (
            "drop"
            if float(r["cloud_fraction"]) > 0.5 or float(r["water_fraction"]) > 0.5
            else "keep"
        )
        for r in rows
    }


async def probe(args: argparse.Namespace) -> None:
    config = config_module.load(args.config)
    tiles_config = TilesConfig(args.tiles, config.tiles.synthetic_count, config.tiles.image_size)
    tiles, source = load_tiles(tiles_config)
    tiles = tiles[:: args.every][: args.n]
    labels = {label.lower() for label in config.model.labels}
    expected = expected_labels(args.tiles) if args.tiles else {}
    print(f"tiles: {source}, {len(tiles)} used; prompt: {config.model.prompt!r}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with make_engine(args.engine, config, log_dir) as engine:
        bodies_1 = [engine.prepare(t.png, config.model.prompt, 1) for t in tiles]
        bodies_4 = [engine.prepare(t.png, config.model.prompt, 4) for t in tiles]
        rows = []
        async with engine.session() as send:
            for tile, body_1, body_4 in zip(tiles, bodies_1, bodies_4, strict=True):
                t0 = time.monotonic()
                one = await send(body_1)
                latency_ms = (one.t_last_token - t0) * 1000
                four = await send(body_4)
                row = {
                    "tile": tile.tile_id,
                    "one_token": one.text,
                    "is_label": one.text.strip().lower() in labels,
                    "expected": expected.get(tile.tile_id),
                    "four_tokens": four.text,
                    "latency_ms": round(latency_ms, 1),
                    "prompt_ms": one.engine_prompt_ms,
                    "n_prompt_tokens": one.n_prompt_tokens,
                }
                rows.append(row)
                print(json.dumps(row))

    (log_dir / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    after_warmup = [r["latency_ms"] for r in rows[1:]]
    if after_warmup:
        spread = max(after_warmup) / min(after_warmup)
        print(
            f"latency after the first request: min {min(after_warmup):.0f} ms, "
            f"max {max(after_warmup):.0f} ms, max/min {spread:.2f} "
            f"({'within' if spread < 2 else 'NOT within'} a factor of two)"
        )
    n_label = sum(r["is_label"] for r in rows)
    print(f"one-token answers that are a configured label: {n_label}/{len(rows)}")
    with_truth = [r for r in rows if r["expected"] is not None]
    if with_truth and labels == {"keep", "drop"}:
        agree = sum(r["one_token"].strip().lower() == r["expected"] for r in with_truth)
        n_drop = sum(r["expected"] == "drop" for r in with_truth)
        print(
            f"agreement with SCL (drop if cloud or water > 50%): {agree}/{len(with_truth)} "
            f"= {agree / len(with_truth):.0%}; SCL says drop for {n_drop}"
        )
    print(f"log: {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--every", type=int, default=1, help="take every k-th tile, to mix scenes")
    parser.add_argument("--tiles", type=Path, help="tile directory (default: the config's)")
    parser.add_argument("--engine", choices=["llama", "dummy"], default="llama")
    parser.add_argument("--log-dir", default="runs/probe-" + time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.tiles is None:
        args.tiles = config_module.load(args.config).tiles.dir
    asyncio.run(probe(args))


if __name__ == "__main__":
    main()
