"""What the device is doing during a run: power, GPU clock and load, co-tenants, power mode.

Power and GPU clock come straight from sysfs, read on the same monotonic clock
as the requests. tegrastats is also kept running for temperatures and memory,
but on JetPack 7 it prints no GPU frequency or load, and its one-second
timestamps are too coarse to attribute energy to a single request.

On a machine without these sensors (a laptop running the dummy engine) the
sampler still runs and still timestamps its rows; the values are NaN or None.
"""

import asyncio
import csv
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
GPU_CLOCK_FILE = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")  # Hz, readable without root


@dataclass
class PowerSample:
    t: float  # monotonic seconds
    gpu_w: float  # VDD_GPU rail; NaN if no sensor
    board_w: float  # whole-board input (VIN); NaN if no sensor
    gpu_mhz: float | None = None  # GPU core clock; None if unreadable


def _find_gpu_rail() -> tuple[Path, Path] | None:
    """Return (voltage_file, current_file) for the VDD_GPU channel of the INA3221 sensor."""
    if not HWMON.exists():
        return None
    for hwmon in HWMON.iterdir():
        for label_file in hwmon.glob("in*_label"):
            if label_file.read_text().strip() == "VDD_GPU":
                channel = label_file.name.removeprefix("in").removesuffix("_label")
                return hwmon / f"in{channel}_input", hwmon / f"curr{channel}_input"
    return None


def _find_board_power() -> Path | None:
    """Return the power file of the INA238 sensor labelled VIN (whole-board input power)."""
    if not HWMON.exists():
        return None
    for hwmon in HWMON.iterdir():
        label_file = hwmon / "label"
        if label_file.exists() and label_file.read_text().strip() == "VIN":
            return hwmon / "power1_input"
    return None


class PowerSampler:
    """Background thread that samples GPU power, board power and GPU clock every `interval_s`.

    Use as a context manager around the measured part of a run:

        with PowerSampler() as power:
            ...
        power.write_csv(run_dir / "power.csv")

    `sensors` says which of the three sources were found, for the run's manifest.
    """

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.samples: list[PowerSample] = []
        self._gpu_rail = _find_gpu_rail()
        self._board_power_file = _find_board_power()
        self._gpu_clock_file = GPU_CLOCK_FILE if GPU_CLOCK_FILE.exists() else None
        self.sensors = {
            "gpu_power": self._gpu_rail is not None,
            "board_power": self._board_power_file is not None,
            "gpu_clock": self._gpu_clock_file is not None,
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _read_once(self) -> PowerSample:
        gpu_w = board_w = math.nan
        gpu_mhz = None
        if self._gpu_rail is not None:
            voltage_file, current_file = self._gpu_rail
            gpu_w = int(voltage_file.read_text()) * int(current_file.read_text()) / 1e6
        if self._board_power_file is not None:
            board_w = int(self._board_power_file.read_text()) / 1e6
        if self._gpu_clock_file is not None:
            gpu_mhz = int(self._gpu_clock_file.read_text()) / 1e6
        return PowerSample(t=time.monotonic(), gpu_w=gpu_w, board_w=board_w, gpu_mhz=gpu_mhz)

    def _sample_until_stopped(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            self.samples.append(self._read_once())
            next_sample += self.interval_s
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def __enter__(self) -> "PowerSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "gpu_w", "board_w", "gpu_mhz"])
            for s in self.samples:
                mhz = "" if s.gpu_mhz is None else f"{s.gpu_mhz:.0f}"
                writer.writerow([f"{s.t:.6f}", f"{s.gpu_w:.3f}", f"{s.board_w:.3f}", mhz])


def read_power_csv(path: Path) -> list[PowerSample]:
    """Read samples back. Files from before the gpu_mhz column load with gpu_mhz=None."""
    with path.open() as f:
        samples = []
        for row in csv.DictReader(f):
            mhz = row.get("gpu_mhz")
            samples.append(
                PowerSample(
                    t=float(row["t"]),
                    gpu_w=float(row["gpu_w"]),
                    board_w=float(row["board_w"]),
                    gpu_mhz=float(mhz) if mhz else None,
                )
            )
        return samples


class Tegrastats:
    """Runs tegrastats for the duration of a `with` block, logging to a file.

    Does nothing on a machine without tegrastats.
    """

    def __init__(self, logfile: Path, interval_ms: int = 100):
        self.command = ["tegrastats", "--interval", str(interval_ms), "--logfile", str(logfile)]
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> "Tegrastats":
        if shutil.which("tegrastats"):
            self._process = subprocess.Popen(self.command)
        return self

    def __exit__(self, *exc) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()


class GpuUtilization:
    """Logs GPU utilisation from `nvidia-smi -lms` for a `with` block.

    nvidia-smi is the only source of GPU load on the Thor; tegrastats has none.
    One long-lived nvidia-smi process streams a line per interval; each line is
    stamped on the monotonic clock as it is read. Does nothing without nvidia-smi.
    """

    def __init__(self, logfile: Path, interval_ms: int = 100):
        self.logfile = logfile
        self.interval_ms = interval_ms
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def _read_lines(self) -> None:
        with self.logfile.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "gpu_util_pct"])
            for line in self._process.stdout:
                writer.writerow([f"{time.monotonic():.6f}", line.strip()])

    def __enter__(self) -> "GpuUtilization":
        if shutil.which("nvidia-smi"):
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._thread = threading.Thread(target=self._read_lines, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._thread.join()


def read_gpu_utilization_csv(path: Path) -> list[tuple[float, float]]:
    """(t, percent) rows; lines nvidia-smi could not parse are skipped."""
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                rows.append((float(row["t"]), float(row["gpu_util_pct"])))
            except ValueError:
                continue
    return rows


async def gpu_co_tenants(exclude_pids: set[int]) -> list[dict]:
    """Other processes holding GPU memory right now, from nvidia-smi.

    Runs as an async subprocess so it never delays a request being sent.
    Empty on a machine without nvidia-smi.
    """
    if not shutil.which("nvidia-smi"):
        return []
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    tenants = []
    for line in stdout.decode().splitlines():
        pid, name, used_mib = (part.strip() for part in line.split(","))
        if int(pid) not in exclude_pids:
            tenants.append({"pid": int(pid), "name": name, "used_mib": used_mib})
    return tenants


def _command_output(command: list[str]) -> str | None:
    """stdout of a command, or None if it is not installed or fails."""
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def power_mode() -> str:
    """The current nvpmodel power mode name, e.g. '120W'. 'unknown' off a Jetson."""
    output = _command_output(["nvpmodel", "-q"])
    if not output:
        return "unknown"
    first_line = output.splitlines()[0]  # "NV Power Mode: 120W"
    return first_line.split(":", 1)[1].strip()


def device_snapshot() -> dict[str, str | None]:
    """Raw text of the device-describing commands, for a cell's manifest."""
    return {
        "nvpmodel_q": _command_output(["nvpmodel", "-q"]),
        "jetson_release": _command_output(["jetson_release", "-v"]),
        "nv_tegra_release": (
            Path("/etc/nv_tegra_release").read_text()
            if Path("/etc/nv_tegra_release").exists()
            else None
        ),
        "nvidia_smi": _command_output(["nvidia-smi"]),
    }
