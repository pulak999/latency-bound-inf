from itertools import islice
from pathlib import Path

import pytest

from harness import config as config_module
from harness.background import send_times
from harness.engines.llama_server import Container, LlamaServerConfig, docker_command
from harness.mps import Mps, free_chunks_and_sms

LSPART = """\
GPU                  Partition                            free     used     free   used   clients
                                                          chunks   chunks   SM     SM
GPU-a7                                                    10       0        20     0
"""

BASE = """
[harness]
runs_dir = "runs"
[model]
name = "m"
quant = "q"
prompt = "p"
max_tokens = 1
[container]
image = "lbi-llama-server:v1"
cuda_dir = "/usr/local/cuda-13.2"
[tenancy]
mps_dir = "~/.lbi-mps"
"""


def load(tmp_path: Path, text: str):
    path = tmp_path / "c.toml"
    path.write_text(text)
    return config_module.load(path)


def test_steady_and_bursty_send_at_the_same_mean_rate():
    n = 20_000
    steady = list(islice(send_times("steady", 0.0, 3.0, 5, seed=1), n))
    bursty = list(islice(send_times("bursty", 0.0, 3.0, 5, seed=1), n))
    # Same mean rate: about one request per 3 s for both.
    assert abs(steady[-1] / n - 3.0) < 0.1
    starts = [t for t in bursty if t is not None]
    assert len(starts) == n // 5
    assert abs(starts[-1] / n - 3.0) < 0.15
    # Bursty: only the first request of each burst has a time; the other four go at once.
    assert bursty[1:5] == [None] * 4 and bursty[5] is not None
    assert list(islice(send_times("saturating", 0.0, 3.0, 5, seed=1), 3)) == [None] * 3
    # Seeded, so a cell's background load can be replayed.
    assert list(islice(send_times("steady", 0.0, 3.0, 5, seed=1), 5)) == steady[:5]


def test_lspart_gives_free_chunks_and_sms():
    assert free_chunks_and_sms(LSPART) == (10, 20)


def test_timeslice_needs_no_mps_environment():
    mps = Mps("timeslice", 3, Path("/x"))
    with mps:
        assert mps.env(0) == {}


def test_container_mounts_build_cuda_and_model_and_passes_mps_env():
    config = LlamaServerConfig(
        model=Path("/models/g/model.gguf"),
        mmproj=Path("/models/g/mmproj.gguf"),
        server_bin=Path("/src/llama.cpp/build/bin/llama-server"),
        container=Container("img", "t1", Path("/usr/local/cuda-13.2"), writable=(Path("/m"),)),
        env={"CUDA_MPS_PIPE_DIRECTORY": "/m/pipe"},
    )
    command = docker_command(config, ["llama-server", "--port", "1"])
    joined = " ".join(command)
    assert command[:2] == ["docker", "run"] and command[-3:] == ["llama-server", "--port", "1"]
    for mount in [
        "/src/llama.cpp/build:/src/llama.cpp/build:ro",
        "/models/g:/models/g:ro",
        "/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro",
        "/m:/m ",
    ]:
        assert mount in joined + " "
    assert "CUDA_MPS_PIPE_DIRECTORY=/m/pipe" in joined
    assert "--ipc host" in joined and "--network host" in joined


def test_cells_with_tenants_need_a_load_and_known_sharing(tmp_path):
    good = load(
        tmp_path,
        BASE + '[[cells]]\nid = "a"\ncadence_s = 3\nframes = 5\n'
        'tenants = 3\nload = "bursty"\nsharing = "mps-partitioned"\nplacement = "container"\n',
    )
    cell = good.cell("a")
    assert (cell.tenants, cell.load, cell.sharing) == (3, "bursty", "mps-partitioned")
    assert good.tenancy.mps_dir == Path("~/.lbi-mps").expanduser()

    with pytest.raises(ValueError, match="need a load"):
        load(tmp_path, BASE + '[[cells]]\nid = "a"\ncadence_s = 3\nframes = 5\ntenants = 2\n')
    with pytest.raises(ValueError, match="sharing"):
        load(tmp_path, BASE + '[[cells]]\nid = "a"\ncadence_s = 3\nframes = 5\nsharing = "mig"\n')


def test_alone_host_cells_need_no_container_or_tenancy_sections(tmp_path):
    plain = BASE.split("[container]")[0]
    config = load(tmp_path, plain + '[[cells]]\nid = "a"\ncadence_s = 3\nframes = 5\n')
    assert config.container is None and config.tenancy is None
    with pytest.raises(ValueError, match="no \\[container\\]"):
        load(
            tmp_path,
            plain + '[[cells]]\nid = "a"\ncadence_s = 3\nframes = 5\nplacement = "container"\n',
        )
