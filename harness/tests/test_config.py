from pathlib import Path

import pytest

from harness import config

CONFIGS = Path(__file__).parents[2] / "configs"

MINIMAL = """
[harness]
runs_dir = "runs"
[model]
name = "m"
quant = "q"
prompt = "p"
max_tokens = 1
[[cells]]
id = "c"
cadence_s = 2
frames = 10
"""


def test_shipped_configs_load():
    for name in ("stack_a.toml", "gate1.toml"):
        loaded = config.load(CONFIGS / name)
        assert loaded.cells and len(loaded.sha256) == 64
        assert loaded.model.max_tokens == 1


def test_minimal_config_gets_defaults(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(MINIMAL)
    loaded = config.load(path)
    assert loaded.harness.warmup_frames == 20
    assert loaded.llama_server is None
    assert loaded.cell("c").co_runner == "none"
    with pytest.raises(KeyError):
        loaded.cell("missing")


@pytest.mark.parametrize("field,value", [("co_runner", "light"), ("power_mode", "70W")])
def test_unimplemented_axes_are_refused_up_front(tmp_path, field, value):
    path = tmp_path / "c.toml"
    path.write_text(MINIMAL + f'{field} = "{value}"\n')
    with pytest.raises(NotImplementedError, match=field):
        config.load(path)


def test_duplicate_cell_ids_are_refused(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(MINIMAL + '[[cells]]\nid = "c"\ncadence_s = 1\nframes = 1\n')
    with pytest.raises(ValueError, match="duplicate"):
        config.load(path)
