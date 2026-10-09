import json
import re
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from ics.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class ToyConfig:
    width: int
    forward_dtype: str = "float32"


class Toy(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(config.width, 2)
        self.register_buffer("z0", torch.zeros(config.width, dtype=getattr(torch, config.forward_dtype)))


@dataclass
class OtherConfig:
    depth: int
    forward_dtype: str = "float32"


def test_round_trip_is_strict_and_in_eval_mode(tmp_path):
    torch.manual_seed(0)
    m = Toy(ToyConfig(3))
    save_checkpoint(m.state_dict(), m.config, tmp_path / "ck", {"source": "run/best.pt"})
    back = load_checkpoint(Toy, ToyConfig, tmp_path / "ck")
    assert not back.training and back.config == ToyConfig(3)
    for k, v in m.state_dict().items():
        torch.testing.assert_close(back.state_dict()[k], v, rtol=0, atol=0)
    meta = json.loads((tmp_path / "ck" / "config.json").read_text())
    assert meta == {"model": {"width": 3, "forward_dtype": "float32"}, "provenance": {"source": "run/best.pt"}}


def test_dtype_override(tmp_path):
    m = Toy(ToyConfig(3))
    save_checkpoint(m.state_dict(), m.config, tmp_path / "ck")
    back = load_checkpoint(Toy, ToyConfig, tmp_path / "ck", dtype="bfloat16")
    assert back.config.forward_dtype == "bfloat16" and back.z0.dtype == torch.bfloat16


def test_refuses_the_config_of_another_model(tmp_path):
    m = Toy(ToyConfig(3))
    save_checkpoint(m.state_dict(), m.config, tmp_path / "ck")
    error = "exactly the OtherConfig fields; missing fields ['depth'], unknown fields ['width']"
    with pytest.raises(ValueError, match=re.escape(error)):
        load_checkpoint(Toy, OtherConfig, tmp_path / "ck")


def test_refuses_weights_that_do_not_match_the_model(tmp_path):
    m = Toy(ToyConfig(3))
    save_checkpoint({k: v for k, v in m.state_dict().items() if k != "z0"}, m.config, tmp_path / "ck")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(Toy, ToyConfig, tmp_path / "ck")
