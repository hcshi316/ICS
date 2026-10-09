"""Release checkpoints: one directory per model, the same format for TRM, the verifier and the baselines.

  config.json         {"model": every field of the model's config dataclass, "provenance": where the weights come from}
  model.safetensors   the weights under the model's state_dict() names

load_checkpoint refuses a missing or unknown config field (a checkpoint of another model) and a training run's
directory (pass one of its checkpoints, RUN/last), then builds the model and loads the weights strictly."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch

CONFIG_FILE, WEIGHTS_FILE = "config.json", "model.safetensors"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(state_dict: dict, config, out_dir, provenance: dict | None = None) -> None:
    """Write config.json ({"model": the config's fields, "provenance": ...}) and model.safetensors into out_dir."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in state_dict.items()}, str(out / WEIGHTS_FILE))
    (out / CONFIG_FILE).write_text(json.dumps({"model": asdict(config), "provenance": provenance or {}}, indent=1))


def model_section(ckpt_dir, config_cls) -> dict:
    """The model section of the checkpoint's config.json, refused unless it holds exactly the config_cls fields. A
    training run's directory (ics/train.py: no config.json of its own, its last/ checkpoint inside, and the verifier's
    best/) is refused by name."""
    path = Path(ckpt_dir) / CONFIG_FILE
    if not path.exists() and any((Path(ckpt_dir) / kept / CONFIG_FILE).exists() for kept in ("last", "best")):
        raise ValueError(f"{ckpt_dir} is a training run; pass a checkpoint, e.g. {Path(ckpt_dir) / 'last'}")
    raw = json.loads(path.read_text())["model"]
    names = {f.name for f in fields(config_cls)}
    if set(raw) != names:
        raise ValueError(f"{path}: the model section must hold exactly the {config_cls.__name__} fields; "
                         f"missing fields {sorted(names - set(raw))}, unknown fields {sorted(set(raw) - names)}")
    return raw


def load_checkpoint(model_cls, config_cls, ckpt_dir, device="cpu", dtype: str | None = None):
    """model_cls(config) on `device` (optionally in another forward dtype) with the weights loaded strictly, in eval
    mode."""
    from safetensors.torch import load_file

    config = config_cls(**model_section(ckpt_dir, config_cls))
    if dtype is not None:
        config.forward_dtype = dtype
    with torch.device(device):
        model = model_cls(config)
    model.load_state_dict(load_file(str(Path(ckpt_dir) / WEIGHTS_FILE)), strict=True)
    return model.eval()
