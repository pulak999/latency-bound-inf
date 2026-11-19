"""The other tenants of a cell: N-1 background VLM servers, each in its own container.

Tenant 0 is the measured one; the controller starts it. This module starts,
for the length of a `with` block:

1. the MPS daemon, if the cell's sharing asks for one (harness.mps);
2. one llama-server container per background tenant, same model as tenant 0,
   on ports base_port + 1, base_port + 2, ...;
3. one load-generator process per background tenant (harness.background),
   after every server is healthy.

On exit it stops them in reverse order, whatever happened inside the block.
Each background tenant's requests are logged to background-<i>.jsonl in the
cell directory and its server log to llama-server-<i>.log.
"""

import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from harness.config import CellConfig, Config
from harness.engines import llama_server
from harness.mps import Mps


def server_config(
    config: Config, port: int, container_name: str | None, env: dict
) -> llama_server.LlamaServerConfig:
    """llama-server settings for one tenant: in a container unless `container_name` is None."""
    settings = config.llama_server
    container = None
    if container_name is not None:
        container = llama_server.Container(
            image=config.container.image,
            name=container_name,
            cuda_dir=config.container.cuda_dir,
            writable=(config.tenancy.mps_dir,),
        )
    return llama_server.LlamaServerConfig(
        model=settings.model,
        mmproj=settings.mmproj,
        port=port,
        server_bin=settings.server_bin,
        request_timeout_s=settings.request_timeout_s,
        container=container,
        env=env,
    )


class Tenants:
    def __init__(self, config: Config, cell: CellConfig, cell_dir: Path):
        self.config = config
        self.cell = cell
        self.cell_dir = cell_dir
        # A plain alone cell (one tenant, no MPS) needs no [tenancy] section; config.load checks.
        mps_dir = config.tenancy.mps_dir if config.tenancy else Path()
        self.mps = Mps(cell.sharing, cell.tenants, mps_dir)
        self._stack = ExitStack()
        self._generators: list[subprocess.Popen] = []

    def tenant0_env(self) -> dict[str, str]:
        return self.mps.env(0)

    def __enter__(self) -> "Tenants":
        try:
            self._stack.enter_context(self.mps)
            ports = []
            for i in range(1, self.cell.tenants):
                port = self.config.llama_server.port + i
                name = f"lbi-{self.cell.id}-tenant{i}"
                server = llama_server.LlamaServer(
                    server_config(self.config, port, name, self.mps.env(i)),
                    logfile=self.cell_dir / f"llama-server-{i}.log",
                )
                self._stack.enter_context(server)
                ports.append(port)
            if self.cell.load != "idle":  # idle tenants hold the model in memory, nothing more
                for i, port in enumerate(ports, start=1):
                    self._start_generator(i, port)
            self._stack.callback(self._stop_generators)
        except BaseException:
            self._stack.close()
            raise
        return self

    def _start_generator(self, i: int, port: int) -> None:
        tenancy = self.config.tenancy
        command = [
            sys.executable,
            "-m",
            "harness.background",
            str(self.config.path),
            "--port",
            str(port),
            "--load",
            self.cell.load,
            "--period-s",
            str(tenancy.background_period_s),
            "--burst-size",
            str(tenancy.burst_size),
            "--tile-offset",
            str(i * 7),  # different tiles from tenant 0 and each other
            "--seed",
            str(i),
            "--out",
            str(self.cell_dir / f"background-{i}.jsonl"),
        ]
        self._generators.append(subprocess.Popen(command))

    def _stop_generators(self) -> None:
        for process in self._generators:
            process.terminate()
        for process in self._generators:
            try:
                process.wait(timeout=30)  # lets the request in flight finish
            except subprocess.TimeoutExpired:
                process.kill()

    def describe(self) -> dict:
        tenancy = self.config.tenancy
        return {
            "tenants": self.cell.tenants,
            "load": self.cell.load,
            "background_period_s": tenancy.background_period_s,
            "burst_size": tenancy.burst_size,
            "placement": self.cell.placement,
            "image": self.config.container.image if self.config.container else None,
        } | self.mps.describe()

    def __exit__(self, *exc) -> None:
        self._stack.close()
