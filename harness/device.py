"""What the device is doing during a run: power, co-tenants, power mode.

Power comes straight from the board's power sensors in sysfs, read on the same
monotonic clock as the requests. tegrastats is also kept running for clocks and
temperatures, but its log has one-second timestamps, too coarse to attribute
energy to a single request.
"""

import asyncio
import csv
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HWMON = Path("/sys/class/hwmon")


@dataclass
class PowerSample:
    t: float  # monotonic seconds
    gpu_w: float  # VDD_GPU rail
    board_w: float  # whole-board input (VIN)


def _find_gpu_rail() -> tuple[Path, Path]:
    """Return (voltage_file, current_file) for the VDD_GPU channel of the INA3221 sensor."""
    for hwmon in HWMON.iterdir():
        for label_file in hwmon.glob("in*_label"):
            if label_file.read_text().strip() == "VDD_GPU":
                channel = label_file.name.removeprefix("in").removesuffix("_label")
                return hwmon / f"in{channel}_input", hwmon / f"curr{channel}_input"
    raise RuntimeError("No VDD_GPU power rail found under /sys/class/hwmon")


def _find_board_power() -> Path:
    """Return the power file of the INA238 sensor labelled VIN (whole-board input power)."""
    for hwmon in HWMON.iterdir():
        label_file = hwmon / "label"
        if label_file.exists() and label_file.read_text().strip() == "VIN":
            return hwmon / "power1_input"
    raise RuntimeError("No VIN power sensor found under /sys/class/hwmon")


class PowerSampler:
    """Background thread that samples GPU and board power every `interval_s` seconds.

    Use as a context manager around the measured part of a run:

        with PowerSampler() as power:
            ...
        power.write_csv(run_dir / "power.csv")
    """

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.samples: list[PowerSample] = []
        self._gpu_voltage_file, self._gpu_current_file = _find_gpu_rail()
        self._board_power_file = _find_board_power()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _read_once(self) -> PowerSample:
        millivolts = int(self._gpu_voltage_file.read_text())
        milliamps = int(self._gpu_current_file.read_text())
        board_microwatts = int(self._board_power_file.read_text())
        return PowerSample(
            t=time.monotonic(),
            gpu_w=millivolts * milliamps / 1e6,
            board_w=board_microwatts / 1e6,
        )

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
            writer.writerow(["t", "gpu_w", "board_w"])
            for s in self.samples:
                writer.writerow([f"{s.t:.6f}", f"{s.gpu_w:.3f}", f"{s.board_w:.3f}"])


def read_power_csv(path: Path) -> list[PowerSample]:
    with path.open() as f:
        return [
            PowerSample(t=float(row["t"]), gpu_w=float(row["gpu_w"]), board_w=float(row["board_w"]))
            for row in csv.DictReader(f)
        ]


class Tegrastats:
    """Runs tegrastats for the duration of a `with` block, logging to a file."""

    def __init__(self, logfile: Path, interval_ms: int = 100):
        self.command = ["tegrastats", "--interval", str(interval_ms), "--logfile", str(logfile)]
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> "Tegrastats":
        self._process = subprocess.Popen(self.command)
        return self

    def __exit__(self, *exc) -> None:
        self._process.terminate()
        self._process.wait()


async def gpu_co_tenants(exclude_pids: set[int]) -> list[dict]:
    """Other processes holding GPU memory right now, from nvidia-smi.

    Runs as an async subprocess so it never delays a request being sent.
    """
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


def power_mode() -> str:
    """The current nvpmodel power mode name, e.g. '120W'."""
    output = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
    first_line = output.splitlines()[0]  # "NV Power Mode: 120W"
    return first_line.split(":", 1)[1].strip()
