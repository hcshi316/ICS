"""The six puzzle tasks: a Task each (base.py has the interface); ics/registry.py names them."""
from __future__ import annotations

from ics.data import Pool
from ics.tasks.base import Task


def make_task(name: str, pool: Pool, restatements: int = 32, seed: int = 7, block: int = 768) -> Task:
    """The Task `name` (ics/registry.py) of the boards of `pool`. restatements, seed and block reach only a task whose
    restatements are drawn at random (Sudoku, ics/tasks/sudoku.py); the others restate through the 8 symmetries of the
    square."""
    # here, not at the top: the registry imports the task modules, and the heads it imports import make_task
    from ics.registry import TASKS

    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {tuple(TASKS)}")
    entry = TASKS[name]
    if entry.random_restatements:
        return entry.task(pool, n_restatements=restatements, seed=seed, block=block)
    return entry.task(pool)
