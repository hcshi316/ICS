import importlib

import yaml

import ics.evaluate
import ics.train
from ics.evaluate import eval_config
from ics.train import recipe


def test_recipes_and_eval_configs_read_3e_4_as_a_float_as_set_does(tmp_path, monkeypatch):
    (tmp_path / "trm.yaml").write_text("optim:\n  lr: 3e-4\ntasks:\n  maze:\n    optim: {lr: -.5}\n")
    for module in (ics.train, ics.evaluate):
        monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    for cfg in (recipe("trm", "sudoku"), eval_config("trm", "sudoku")):
        assert cfg["optim"]["lr"] == 3e-4 and type(cfg["optim"]["lr"]) is float
    assert recipe("trm", "maze")["optim"]["lr"] == -0.5


def test_importing_the_cli_leaves_yaml_safe_load_as_yaml_1_1():
    importlib.import_module("ics.cli")
    assert yaml.safe_load("3e-4") == "3e-4" and yaml.safe_load("1.0e-4") == 1e-4
