"""Every task, training head and evaluation method of the package, by name. The command line (ics/cli.py), the
training loop (ics/train.py), the evaluation (ics/evaluate.py) and make_task (ics/tasks/__init__.py) look them up
here, and the CLI's choices follow the order below.

A new task: its Task (ics/tasks/), its builder (ics/builders/), and a TASKS entry.
A new baseline: its package (ics_baselines/<name>/: model.py, predict.py, train.py), its recipes
(ics/configs/train/<name>.yaml, ics/configs/eval/<name>.yaml), a seeding purpose id (ics/seeding.py), and a HEADS and
a METHODS entry."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from ics.builders.lightup import build_lightup
from ics.builders.maze import build_maze
from ics.builders.ppb import AUGMENT, build_ppb
from ics.builders.sudoku import build_sudoku
from ics.methods.ptrm import run_ptrm
from ics.methods.trm import run_trm
from ics.tasks.lightup import LightUp
from ics.tasks.maze import Maze
from ics.tasks.ppb import KINDS, PPB
from ics.tasks.sudoku import Sudoku
from ics.trm.model import load_trm
from ics.trm.train import TRMHead
from ics.verifier.train import VerifierHead
from ics_baselines.attractor.model import load_attractor
from ics_baselines.attractor.predict import run_attractor
from ics_baselines.attractor.train import AttractorHead
from ics_baselines.eqr.model import load_eqr
from ics_baselines.eqr.predict import run_eqr
from ics_baselines.eqr.train import EqRHead
from ics_baselines.gram.model import load_gram
from ics_baselines.gram.predict import run_gram
from ics_baselines.gram.train import GRAMHead


@dataclass(frozen=True)
class TaskEntry:
    task: Callable                      # task(pool) -> the Task of a pool of boards
    build: Callable                     # build(out, source=None[, seed=0]): python -m ics data
    board_rows: int = 0                 # the built train split's rows per board, each row its own group (the
                                        # verifier's data takes each board's first); 0: each group of puzzles is a board
    random_restatements: bool = False   # then task(pool, n_restatements, seed, block): restatements drawn per block


@dataclass(frozen=True)
class Method:
    load: Callable                      # load(ckpt, device, dtype) -> the model, in eval mode
    run: Callable | None                # run(model, task, **its eval config but `rows`) -> answers
                                        # (ics/methods/__init__.py); None: ICS, run per regime by ics/evaluate.py
    blocked: bool = False               # draws per block of `batch` boards (ics/seeding.py): a shard starts on one


TASKS = {
    "sudoku": TaskEntry(Sudoku, build_sudoku, random_restatements=True),
    "maze": TaskEntry(Maze, build_maze),
    "lightup": TaskEntry(LightUp, build_lightup, board_rows=8),
    **{kind: TaskEntry(partial(PPB, kind=kind), partial(build_ppb, kind=kind), board_rows=AUGMENT) for kind in KINDS},
}
HEADS = {"trm": TRMHead, "eqr": EqRHead, "gram": GRAMHead, "attractor": AttractorHead, "verifier": VerifierHead}
METHODS = {
    "trm": Method(load_trm, run_trm),
    "ptrm": Method(load_trm, run_ptrm, blocked=True),
    "ics": Method(load_trm, None),
    "gram": Method(load_gram, run_gram, blocked=True),
    "eqr": Method(load_eqr, run_eqr, blocked=True),
    "attractor": Method(load_attractor, run_attractor, blocked=True),
}
