"""The images a cell sends: real Sentinel-2 tiles from a directory, or synthetic noise.

A tile directory holds PNGs and an `index.csv` with at least `tile_id` and
`file` columns (setup/tiles_sentinel2.py writes one). Every tile is read into
memory before a cell starts, so no file I/O happens on the timed path.
"""

import csv
from dataclasses import dataclass

from harness.config import TilesConfig
from harness.synthetic import synthetic_png


@dataclass(frozen=True)
class Tile:
    tile_id: str
    png: bytes


def load_tiles(config: TilesConfig) -> tuple[list[Tile], str]:
    """Return the tiles and a one-word description of where they came from."""
    if config.dir is not None and (config.dir / "index.csv").exists():
        with (config.dir / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{config.dir / 'index.csv'} lists no tiles")
        tiles = [Tile(row["tile_id"], (config.dir / row["file"]).read_bytes()) for row in rows]
        return tiles, f"sentinel2:{config.dir}"

    tiles = [
        Tile(f"synthetic-{seed}", synthetic_png(seed, size=config.image_size))
        for seed in range(config.synthetic_count)
    ]
    return tiles, "synthetic"
