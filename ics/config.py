"""Settings: one YAML dialect for the training recipes (ics/configs/train), the evaluation configs (ics/configs/eval)
and the command line's --set values, and the rules that merge and check overrides.

The dialect is yaml.safe_load's (YAML 1.1), plus floats for the plain scalars YAML 1.1 leaves as strings: an exponent
without a dot or a sign (3e-4, 1e5, 1.5e5) and a signed leading dot (-.5). Quoted scalars stay strings, and
yaml.safe_load itself is unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    """yaml.SafeLoader with the floats of the dialect."""


_Loader.add_implicit_resolver("tag:yaml.org,2002:float",
                              re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?[eE][-+]?[0-9]+|\.[0-9]+(?:[eE][-+]?[0-9]+)?)$"),
                              list("-+0123456789."))


def parse_yaml(text: str):
    """`text` read in the dialect."""
    return yaml.load(text, Loader=_Loader)


def load_yaml(path):
    """The settings file at `path` (a recipe or an evaluation config) read in the dialect."""
    return parse_yaml(Path(path).read_text())


def merge_config(base: dict, over: dict | None) -> dict:
    """Recursive dict merge; values in `over` win."""
    out = dict(base)
    for key, value in (over or {}).items():
        out[key] = (merge_config(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict)
                    else value)
    return out


def check_settings(cfg: dict, over: dict, prefix: str = "") -> None:
    """Every key of `over` must be a setting of `cfg`, with a dict exactly where `cfg` has a group of settings."""
    for key, value in over.items():
        name = f"{prefix}{key}"
        if key not in cfg:
            raise ValueError(f"unknown setting {name}")
        if isinstance(cfg[key], dict) != isinstance(value, dict):
            kind = "a group of settings" if isinstance(cfg[key], dict) else "a single value"
            raise ValueError(f"setting {name} is {kind}, not {value!r}")
        if isinstance(value, dict):
            check_settings(cfg[key], value, f"{name}.")
