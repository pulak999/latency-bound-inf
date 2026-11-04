"""Cut Sentinel-2 L2A tiles for Stack A: 896x896 true-colour PNGs with their SCL band.

    uv run --group tiles python setup/tiles_sentinel2.py --out tiles --n-tiles 1000

Reads scenes from Microsoft's Planetary Computer STAC (no account needed), takes
non-overlapping 896-pixel windows of the 10 m true-colour image, and saves each
as PNG next to its scene-classification (SCL) window, resampled from 20 m to
the same grid, as a uint8 PNG. index.csv lists every tile with its cloud, water
and nodata fractions from SCL, so an accuracy check against the model's keep/drop
is possible later. Scenes are spread over a few regions so cloud, land and water
all appear; the mix is reported at the end.

SCL classes: 0 nodata, 1 saturated, 2 dark, 3 cloud shadow, 4 vegetation, 5 bare,
6 water, 7 unclassified, 8 cloud medium, 9 cloud high, 10 thin cirrus, 11 snow.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TILE_PX = 896  # Gemma 3's native image size
CLOUD = (8, 9, 10)
WATER = (6,)

# (name, bbox west south east north, date range, max scene cloud %). A mix on
# purpose: cloud-free land, coast with water, and cloudy scenes.
REGIONS = [
    ("iberia-land", (-4.5, 39.5, -3.0, 40.5), "2025-06-01/2025-09-30", 10),
    ("brittany-coast", (-4.8, 47.8, -3.0, 48.8), "2025-05-01/2025-09-30", 40),
    ("bengal-delta", (89.0, 21.8, 90.5, 22.8), "2025-10-01/2026-03-31", 40),
    ("great-lakes", (-87.5, 44.5, -86.0, 45.5), "2025-06-01/2025-09-30", 60),
    ("amazon-cloudy", (-61.0, -4.0, -59.5, -3.0), "2025-11-01/2026-03-31", 90),
    ("uk-cloudy", (-2.5, 53.0, -1.0, 54.0), "2025-09-01/2026-02-28", 90),
]


def scenes_for(client, region, per_region: int):
    name, bbox, dates, max_cloud = region
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=dates,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "desc"}],
        max_items=per_region,
    )
    return [(name, item) for item in search.items()]


def cut_scene(item, region_name: str, out: Path, n_wanted: int, next_id: int) -> list[dict]:
    """Save up to n_wanted tiles from one scene; returns their index rows."""
    signed = planetary_computer.sign(item)
    rows = []
    with (
        rasterio.open(signed.assets["visual"].href) as rgb,
        rasterio.open(signed.assets["SCL"].href) as scl,
    ):
        n_rows, n_cols = rgb.height // TILE_PX, rgb.width // TILE_PX
        scale = scl.width / rgb.width  # SCL is 20 m, RGB is 10 m: 0.5
        positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        # Spread the windows over the scene rather than taking the top-left corner.
        step = max(1, len(positions) // n_wanted)
        for r, c in positions[::step][:n_wanted]:
            window = Window(c * TILE_PX, r * TILE_PX, TILE_PX, TILE_PX)
            pixels = rgb.read([1, 2, 3], window=window)  # (3, H, W) uint8
            scl_window = Window(
                c * TILE_PX * scale, r * TILE_PX * scale, TILE_PX * scale, TILE_PX * scale
            )
            classes = scl.read(
                1, window=scl_window, out_shape=(TILE_PX, TILE_PX), resampling=Resampling.nearest
            )
            if pixels.shape[1:] != (TILE_PX, TILE_PX):
                continue
            nodata = float(np.mean(classes == 0))
            if nodata > 0.05:
                continue  # scene edge
            tile_id = f"{next_id + len(rows):05d}"
            Image.fromarray(np.moveaxis(pixels, 0, -1)).save(out / f"{tile_id}.png")
            Image.fromarray(classes.astype(np.uint8)).save(out / f"{tile_id}_scl.png")
            bounds = rgb.window_bounds(window)
            rows.append(
                {
                    "tile_id": tile_id,
                    "file": f"{tile_id}.png",
                    "scl_file": f"{tile_id}_scl.png",
                    "scene_id": item.id,
                    "region": region_name,
                    "datetime": item.datetime.isoformat(),
                    "row": r,
                    "col": c,
                    "cloud_fraction": round(float(np.mean(np.isin(classes, CLOUD))), 4),
                    "water_fraction": round(float(np.mean(np.isin(classes, WATER))), 4),
                    "nodata_fraction": round(nodata, 4),
                    "crs": str(rgb.crs),
                    "bounds": json.dumps([round(b, 1) for b in bounds]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-tiles", type=int, default=1000)
    parser.add_argument("--scenes-per-region", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    client = pystac_client.Client.open(STAC)
    scenes = [s for region in REGIONS for s in scenes_for(client, region, args.scenes_per_region)]
    if not scenes:
        raise SystemExit("no scenes found")
    per_scene = -(-args.n_tiles // len(scenes))  # ceiling division

    rows: list[dict] = []
    for region_name, item in scenes:
        if len(rows) >= args.n_tiles:
            break
        wanted = min(per_scene, args.n_tiles - len(rows))
        new_rows = cut_scene(item, region_name, args.out, wanted, next_id=len(rows))
        rows.extend(new_rows)
        print(f"{region_name:16s} {item.id}: {len(new_rows)} tiles (total {len(rows)})")

    with (args.out / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cloud = np.array([r["cloud_fraction"] for r in rows])
    water = np.array([r["water_fraction"] for r in rows])
    print(f"{len(rows)} tiles in {args.out}")
    print(f"mostly cloud (>50%): {np.mean(cloud > 0.5):.0%}")
    print(f"mostly water (>50%): {np.mean(water > 0.5):.0%}")
    print(f"neither: {np.mean((cloud <= 0.5) & (water <= 0.5)):.0%}")


if __name__ == "__main__":
    main()
