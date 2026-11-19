"""CUDA MPS for a cell: one private daemon, optionally with equal static SM partitions.

Three ways for tenants to share the GPU (a cell's `sharing`):

- `timeslice`: no MPS. Each tenant has its own CUDA context and the driver
  time-slices between them.
- `mps`: one MPS server; all tenants share one context and may use every SM.
- `mps-partitioned`: MPS in static partitioning mode (`-S`). The SMs are split
  into equal, disjoint partitions, one per tenant. On the Thor the unit is a
  chunk of 2 SMs (10 chunks of 20 SMs), so each tenant gets
  floor(chunks / tenants) chunks and any remainder stays idle. With 2 or 5
  tenants nothing is left over; with 3 tenants 2 SMs are idle, with 4 tenants 4.

The daemon uses its own pipe directory, so labmates' CUDA processes, which use
the default one, never connect to it. Unix socket paths are limited to 108
bytes, so the directory must have a short path.
"""

import os
import re
import subprocess
import time
from pathlib import Path

SHARING = ("timeslice", "mps", "mps-partitioned")
SMS_PER_CHUNK = 2  # Thor; `lspart` reports both chunks and SMs, and we check


class Mps:
    """Starts the MPS daemon for a `with` block (unless sharing is `timeslice`)."""

    def __init__(self, sharing: str, n_tenants: int, directory: Path):
        if sharing not in SHARING:
            raise ValueError(f"sharing must be one of {SHARING}, not {sharing!r}")
        self.sharing = sharing
        self.n_tenants = n_tenants
        self.pipe_dir = directory / "pipe"
        self.log_dir = directory / "log"
        self.partitions: list[str] = []  # "<GPU UUID>/<partition id>", one per tenant
        self.sms_per_tenant: int | None = None

    @property
    def active(self) -> bool:
        return self.sharing != "timeslice"

    def _control(self, command: str) -> str:
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            capture_output=True,
            text=True,
            env=os.environ | {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir)},
        )
        return result.stdout

    def __enter__(self) -> "Mps":
        if not self.active:
            return self
        if len(str(self.pipe_dir / "control_privileged")) > 100:
            raise ValueError(f"MPS pipe directory path too long for a socket: {self.pipe_dir}")
        if (self.pipe_dir / "control").exists() and self._control("get_server_list").strip():
            raise RuntimeError(f"an MPS daemon is already running on {self.pipe_dir}")
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        daemon = ["nvidia-cuda-mps-control", "-d"]
        if self.sharing == "mps-partitioned":
            daemon.append("-S")
        env = os.environ | {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir),
        }
        subprocess.run(daemon, env=env, check=True)
        time.sleep(1)
        if self.sharing == "mps-partitioned":
            self._partition_equally()
        return self

    def _partition_equally(self) -> None:
        free_chunks, free_sms = free_chunks_and_sms(self._control("lspart"))
        if free_sms != free_chunks * SMS_PER_CHUNK:
            raise RuntimeError(f"expected {SMS_PER_CHUNK} SMs per chunk, lspart says otherwise")
        chunks = free_chunks // self.n_tenants
        if chunks == 0:
            raise ValueError(f"{self.n_tenants} tenants do not fit in {free_chunks} chunks")
        uuid = gpu_uuid()
        for _ in range(self.n_tenants):
            created = re.search(
                r"Partition (\S+) created", self._control(f"sm_partition add {uuid} {chunks}")
            )
            if created is None:
                raise RuntimeError("could not create an MPS partition; see the MPS log")
            self.partitions.append(created.group(1))
        self.sms_per_tenant = chunks * SMS_PER_CHUNK

    def env(self, tenant: int) -> dict[str, str]:
        """Environment that puts tenant `tenant` (0 = measured) on this MPS server."""
        if not self.active:
            return {}
        env = {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir)}
        if self.partitions:
            env["CUDA_MPS_SM_PARTITION"] = self.partitions[tenant]
        return env

    def describe(self) -> dict:
        return {
            "sharing": self.sharing,
            "sms_per_tenant": self.sms_per_tenant,
            "partitions": self.partitions,
        }

    def __exit__(self, *exc) -> None:
        if self.active:
            self._control("quit")
            time.sleep(1)


def free_chunks_and_sms(lspart: str) -> tuple[int, int]:
    """Free chunks and SMs from `lspart`'s GPU line (the first line after the two headers)."""
    for line in lspart.splitlines()[2:]:
        fields = line.split()
        if len(fields) == 5:  # GPU, free chunks, used chunks, free SMs, used SMs
            return int(fields[1]), int(fields[3])
    raise RuntimeError(f"could not read lspart output:\n{lspart}")


def gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()[0]
