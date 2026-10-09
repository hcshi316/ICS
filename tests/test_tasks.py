"""Properties every task shares: pin is idempotent. The registry's tasks, and an import order that works."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fakes import make_pool, maze_pool, sudoku_pool
from ics.registry import TASKS
from ics.tasks import make_task
from ics.tasks.base import on_canvas
from ics.tasks.lightup import BULB, WALL, LightUp
from ics.tasks.maze import PATH, Maze
from ics.tasks.ppb import KINDS
from ics.tasks.sudoku import DIGITS, Sudoku


def ppb_task(kind, board, vocab):
    """One PPB board (a token grid) at (1, 1) of the 26x26 canvas, inside a ring of walls (token 1), and the size of
    its vocabulary: tokens 0..4, then those of `vocab`."""
    h, w = board.shape
    x = np.zeros((26, 26), np.int64)
    x[:h + 2, :w + 2] = 1
    x[1:h + 1, 1:w + 1] = board
    pool = make_pool([x.reshape(-1)], [x.reshape(-1)], dims=np.array([[h, w, 1, 1]]),
                     vocab={str(k): v for k, v in vocab.items()})
    return make_task(kind, pool), max(vocab.values()) + 1


def synthetic(name):
    """A task on synthetic boards, and the size of its vocabulary (every token a model can write)."""
    if name == "sudoku":
        return make_task(name, sudoku_pool([(0, 10, 40, 80), (1, 2, 3)])), int(DIGITS[-1]) + 1
    if name == "maze":
        return make_task(name, maze_pool([1, 2, 3], (1, 1), (1, 10))), PATH + 1
    if name == "lightup":
        x = on_canvas(np.array([[1, 4, 1, 1], [1, 1, 2, 1]]), 26, WALL)     # a wall numbered 1 and a plain wall
        return make_task(name, make_pool([x], [x], dims=np.array([(2, 4, 1, 1)]))), BULB + 1
    if name in ("nurikabe", "tapa"):                                     # one clue, token 5, amid undecided cells
        clue = ("nu", 2) if name == "nurikabe" else ("ta", "1,1")
        return ppb_task(name, np.array([[2, 2, 2], [2, 5, 2], [2, 2, 2]]), {clue: 5})
    # Heyawake: a clue-0 room of one cell, whose structure has no shaded token, and a room of three cells
    return ppb_task(name, np.array([[5, 7], [7, 7]]), {("hy", 1, 1, 0, 0): 5, ("hy", 1, 1, 0, 2): 6,
                                                       ("hy", 0, 0, None, 0): 7, ("hy", 0, 0, None, 1): 8,
                                                       ("hy", 0, 0, None, 2): 9})


@pytest.mark.parametrize("name", TASKS)
def test_pin_is_idempotent(name):
    task, vocab_size = synthetic(name)
    rng = np.random.default_rng(0)
    for i in range(len(task)):
        for rate in (0.0, 0.05, 0.3, 1.0):                              # the label, perturbed a little, a lot, random
            for _ in range(50):
                y = task.Y[i].copy()
                cells = rng.random(y.shape) < rate
                y[cells] = rng.integers(0, vocab_size, int(cells.sum()))
                pinned = task.pin(i, y)
                np.testing.assert_array_equal(task.pin(i, pinned), pinned)


@pytest.mark.parametrize("first", ["ics.registry", "ics.tasks", "ics.trm.train", "ics.train", "ics.evaluate", "ics.cli",
                                   "ics_baselines.eqr.train", "ics.verifier.data", "ics.verifier.train"])
def test_any_module_can_be_imported_first(first):
    code = f"import {first}; from ics.tasks import make_task; from ics.registry import HEADS, METHODS, TASKS"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_make_task_builds_each_registered_task_and_restates_sudoku_by_its_settings():
    pool = sudoku_pool([(0,)])
    t = make_task("sudoku", pool, restatements=3, seed=5, block=2)
    assert isinstance(t, Sudoku) and (t.n_restatements, t.seed, t.block) == (3, 5, 2)
    maze = make_pool(np.zeros((0, 900)), np.zeros((0, 900)))
    canvas = make_pool(np.zeros((0, 676)), np.zeros((0, 676)), dims=np.zeros((0, 4), np.int64), vocab={})
    assert isinstance(make_task("maze", maze), Maze) and isinstance(make_task("lightup", canvas), LightUp)
    assert [make_task(kind, canvas).kind for kind in KINDS] == list(KINDS)        # each PPB entry keeps its kind
    with pytest.raises(ValueError, match=r"unknown task 'chess'; expected one of \('sudoku', 'maze'"):
        make_task("chess", pool)
