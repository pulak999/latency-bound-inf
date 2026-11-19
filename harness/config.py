"""The one config file: every number that affects a result lives here, not in code.

A config is a TOML file (see configs/). `load` parses and validates it and keeps
the file's hash so a cell's manifest can say exactly which settings produced it.
"""

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from harness.engines.dummy import SleepDistribution

# What the controller can do today. Co-runners and power-mode changes are
# later extensions; a config that asks for them is refused up front rather
# than silently run without them.
CO_RUNNERS = ("none",)
POWER_MODES = ("current",)

# Multi-tenant cells: what the background tenants do, how the GPU is shared,
# and where the measured tenant runs. See harness/background.py and harness/mps.py.
LOADS = ("none", "idle", "steady", "bursty", "saturating")  # idle: resident, no requests
SHARING = ("timeslice", "mps", "mps-partitioned")
PLACEMENTS = ("host", "container")


@dataclass(frozen=True)
class HarnessConfig:
    runs_dir: Path
    warmup_frames: int = 20
    start_delay_s: float = 1.0
    emitter_max_late_fraction: float = 0.1
    sampler_interval_s: float = 0.01


@dataclass(frozen=True)
class ModelConfig:
    name: str  # what the record's `model` field says, e.g. "gemma-3-4b-it"
    quant: str  # e.g. "QAT Q4_0"
    prompt: str
    max_tokens: int
    labels: tuple[str, ...]  # the words the model is expected to answer with


@dataclass(frozen=True)
class LlamaServerSettings:
    server_bin: Path
    model: Path
    mmproj: Path
    port: int = 8081
    request_timeout_s: float = 300.0


@dataclass(frozen=True)
class TilesConfig:
    dir: Path | None  # directory of PNG tiles with an index.csv; None or missing -> synthetic
    synthetic_count: int = 64
    image_size: int = 896


@dataclass(frozen=True)
class ContainerConfig:
    image: str  # built from setup/containers/llama-server.Dockerfile
    cuda_dir: Path  # host CUDA toolkit, mounted into every container


@dataclass(frozen=True)
class TenancyConfig:
    mps_dir: Path  # short path: MPS sockets live here
    background_period_s: float = 3.0  # steady and bursty: one request per period on average
    burst_size: int = 5  # bursty: this many back to back, every burst_size * period


@dataclass(frozen=True)
class CellConfig:
    id: str
    cadence_s: float
    frames: int
    co_runner: str = "none"
    power_mode: str = "current"
    tenants: int = 1  # VLM servers on the GPU, the measured one included
    load: str = "none"  # what each background tenant does
    sharing: str = "timeslice"
    placement: str = (
        "host"  # where the measured tenant runs; background tenants are always containers
    )


@dataclass(frozen=True)
class Config:
    path: Path
    sha256: str
    harness: HarnessConfig
    model: ModelConfig
    llama_server: LlamaServerSettings | None
    dummy: SleepDistribution
    tiles: TilesConfig
    cells: list[CellConfig] = field(default_factory=list)
    container: ContainerConfig | None = None
    tenancy: TenancyConfig | None = None

    def cell(self, cell_id: str) -> CellConfig:
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        raise KeyError(f"no cell {cell_id!r} in {self.path}; have {[c.id for c in self.cells]}")


def _cell(raw: dict) -> CellConfig:
    cell = CellConfig(
        id=raw["id"],
        cadence_s=float(raw["cadence_s"]),
        frames=int(raw["frames"]),
        co_runner=raw.get("co_runner", "none"),
        power_mode=raw.get("power_mode", "current"),
        tenants=int(raw.get("tenants", 1)),
        load=raw.get("load", "none"),
        sharing=raw.get("sharing", "timeslice"),
        placement=raw.get("placement", "host"),
    )
    if cell.cadence_s <= 0 or cell.frames <= 0:
        raise ValueError(f"cell {cell.id}: cadence_s and frames must be positive")
    for name, value, allowed in [
        ("load", cell.load, LOADS),
        ("sharing", cell.sharing, SHARING),
        ("placement", cell.placement, PLACEMENTS),
    ]:
        if value not in allowed:
            raise ValueError(f"cell {cell.id}: {name} must be one of {allowed}, not {value!r}")
    if cell.tenants < 1:
        raise ValueError(f"cell {cell.id}: tenants must be at least 1")
    if (cell.tenants > 1) != (cell.load != "none"):
        raise ValueError(f"cell {cell.id}: background tenants need a load, and a load needs them")
    if cell.co_runner not in CO_RUNNERS:
        raise NotImplementedError(
            f"cell {cell.id}: co_runner {cell.co_runner!r} is not implemented yet "
            f"(have {CO_RUNNERS})"
        )
    if cell.power_mode not in POWER_MODES:
        raise NotImplementedError(
            f"cell {cell.id}: power_mode {cell.power_mode!r} needs sudo for nvpmodel, "
            f"which the controller does not have yet (have {POWER_MODES})"
        )
    return cell


def load(path: Path) -> Config:
    data = path.read_bytes()
    raw = tomllib.loads(data.decode())

    harness = raw.get("harness", {})
    model = raw["model"]
    llama = raw.get("llama_server")
    dummy = raw.get("dummy", {})
    tiles = raw.get("tiles", {})
    container = raw.get("container")
    tenancy = raw.get("tenancy")

    cells = [_cell(c) for c in raw.get("cells", [])]
    ids = [c.id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate cell ids in {path}: {ids}")
    for cell in cells:
        needs_container = cell.tenants > 1 or cell.placement == "container"
        if needs_container and container is None:
            raise ValueError(f"cell {cell.id} runs containers but {path} has no [container]")
        if (cell.tenants > 1 or cell.sharing != "timeslice") and tenancy is None:
            raise ValueError(f"cell {cell.id} needs a [tenancy] section in {path}")

    return Config(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        harness=HarnessConfig(
            runs_dir=Path(harness["runs_dir"]).expanduser(),
            warmup_frames=int(harness.get("warmup_frames", 20)),
            start_delay_s=float(harness.get("start_delay_s", 1.0)),
            emitter_max_late_fraction=float(harness.get("emitter_max_late_fraction", 0.1)),
            sampler_interval_s=float(harness.get("sampler_interval_s", 0.01)),
        ),
        model=ModelConfig(
            name=model["name"],
            quant=model["quant"],
            prompt=model["prompt"],
            max_tokens=int(model["max_tokens"]),
            labels=tuple(model.get("labels", [])),
        ),
        llama_server=(
            LlamaServerSettings(
                server_bin=Path(llama["server_bin"]),
                model=Path(llama["model"]),
                mmproj=Path(llama["mmproj"]),
                port=int(llama.get("port", 8081)),
                request_timeout_s=float(llama.get("request_timeout_s", 300.0)),
            )
            if llama
            else None
        ),
        dummy=SleepDistribution(
            kind=dummy.get("kind", "uniform"),
            low_s=float(dummy.get("low_s", 0.2)),
            high_s=float(dummy.get("high_s", 2.0)),
            seed=int(dummy.get("seed", 0)),
        ),
        tiles=TilesConfig(
            dir=Path(tiles["dir"]).expanduser() if tiles.get("dir") else None,
            synthetic_count=int(tiles.get("synthetic_count", 64)),
            image_size=int(tiles.get("image_size", 896)),
        ),
        cells=cells,
        container=(
            ContainerConfig(image=container["image"], cuda_dir=Path(container["cuda_dir"]))
            if container
            else None
        ),
        tenancy=(
            TenancyConfig(
                mps_dir=Path(tenancy["mps_dir"]).expanduser(),
                background_period_s=float(tenancy.get("background_period_s", 3.0)),
                burst_size=int(tenancy.get("burst_size", 5)),
            )
            if tenancy
            else None
        ),
    )
