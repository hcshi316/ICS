import os
import types
from pathlib import Path

import pytest
import torch

from ics.cli import DATA_FOLDERS

CUBLAS = os.environ.get("CUBLAS_WORKSPACE_CONFIG")          # as the session found it

# Folder names of the built datasets under $ICS_DATA_ROOT, per task (a folder named after the task also does); the
# Light-Up and PPB ones are those python -m ics download --dataset reads.
DATA_DIRS = {"sudoku": "sudoku-extreme-1k-aug-1000", "maze": "maze-30x30-hard-1k", **DATA_FOLDERS}


@pytest.fixture(autouse=True)
def default_mode():
    """train.deterministic (the verifier's recipe sets it) turns PyTorch's deterministic mode on, and may set
    CUBLAS_WORKSPACE_CONFIG, for the rest of the process: every test starts and ends in the default mode, with the
    variable as the session found it."""
    def reset():
        torch.use_deterministic_algorithms(False)
        if CUBLAS is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS

    reset()
    yield
    reset()


@pytest.fixture
def dataset():
    """dataset(task): the task's built dataset under $ICS_DATA_ROOT, in its DATA_DIRS folder or in one named after the
    task (the README builds data/<task>); the test skips without one."""
    root = os.environ.get("ICS_DATA_ROOT")
    if not root:
        pytest.skip("set ICS_DATA_ROOT to run data tests")

    def find(task: str) -> Path:
        for name in (DATA_DIRS[task], task):
            if (Path(root) / name).is_dir():
                return Path(root) / name
        pytest.skip(f"no {task} dataset under {root} ({DATA_DIRS[task]} or {task})")

    return find


@pytest.fixture
def fake_ppbench(monkeypatch):
    """Build the PPBench datasets from fakes.ppbench_source's files without ppbench and Node.js: a fake row's
    puzzle_url is its board text, which the stand-in for ppbench's Puzzle gives back; val takes 2 boards."""
    from ics.builders import lightup, ppb

    puzzle = lambda url: types.SimpleNamespace(get_string_repr=lambda: url)
    monkeypatch.setattr(ppb, "load_ppbench", lambda: types.SimpleNamespace(Puzzle=puzzle))
    monkeypatch.setattr(lightup, "VAL", 2)
    monkeypatch.setattr(ppb, "VAL", dict.fromkeys(ppb.VAL, 2))
