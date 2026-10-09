"""What a method needs to know about a puzzle type.

A Task wraps one pool of boards (ics.data.Pool): an evaluation pool or a window of one (ICS runs a task per window,
ics/evaluate.py), or the train boards of the verifier's data (one task of them all, ics/verifier/data.py). Board i
is the flat input row X[i] exactly as the model sees it; an answer y is a flat row of the same length.
  pin(i, y)          restore the given cells from the input and clear tokens that cannot appear in an answer
  valid(i, y)        y, exactly as submitted, is a correct answer: the rules hold and no given cell changed
  consistent(i, y)   y keeps the input's given cells and holds no token pin clears, i.e. pin(i, y) == y; the PPB tasks
                     check only their clue cells and Heyawake's room structure (ICS ranks children by it; its q
                     terminal requires it)
  editable(i, row)   cells an ICS hypothesis may write into `row` (the input plus the hints written so far)
  alternatives(i, cell, current, ranked, answer)   tokens a hypothesis may write into `cell` instead of `current`,
                     ordered by `ranked` (the model's best-first token list for that cell); `answer` is the parent's
                     pinned answer
  restatements(i, row)       equivalent rewritings of `row` as (tag, row) pairs
  restate_back(i, tag, y)    map an answer to a restated row back onto board i (None if it cannot be mapped)
Every validity check exists in two scorings: "raw" (the answer as produced) and "pinned" (the answer after pin).
"""
from __future__ import annotations

import numpy as np

from ics.data import Pool

SCORINGS = ("raw", "pinned")
DIHEDRAL = tuple((k4, mir) for k4 in range(4) for mir in (False, True))      # the 8 views, in restatement order


def rank_order(tokens, ranked) -> list[int]:
    """`tokens` in the order of `ranked` (best first); tokens missing from `ranked` keep their order after it."""
    pos = {int(t): j for j, t in enumerate(ranked)}
    return sorted((int(t) for t in tokens), key=lambda t: pos.get(t, len(ranked)))


def turn(grid: np.ndarray, k4: int, mir: bool) -> np.ndarray:
    """View (k4, mir) of a grid: k4 quarter turns (np.rot90), then mirrored left-right if `mir`."""
    grid = np.rot90(grid, k4)
    return np.fliplr(grid) if mir else grid


def canvas_views(side: int, dims) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 8 views (DIHEDRAL) of the board at dims = (h, w, r0, c0) of a side x side canvas, each at (1, 1) of an empty
    canvas inside a ring of walls, as index maps (int16; int32 past 181 x 181). With n the cells numpy crops at dims:
      cells [n]              the board's cells on its canvas, row by row
      gather [8, side**2]    view k of a row is [*row, pad, wall][gather[k]] (lay_out)
      back [8, n]            view k shows board cell j at back[k, j]
    A board that does not fit at (1, 1) raises ValueError."""
    h, w, r0, c0 = dims
    P = side * side
    dtype = np.int16 if P + 2 <= 2 ** 15 else np.int32
    index = np.arange(P).reshape(side, side)
    cells = index[r0:r0 + h, c0:c0 + w]
    gather, back = np.full((8, side, side), P), []
    for k, (k4, mir) in enumerate(DIHEDRAL):
        view = turn(cells, k4, mir)
        hh, ww = view.shape
        gather[k, :hh + 2, :ww + 2] = P + 1
        gather[k, 1:hh + 1, 1:ww + 1] = view
        shown = index[1:hh + 1, 1:ww + 1]
        back.append(np.rot90(np.fliplr(shown) if mir else shown, -k4).ravel())
    return cells.ravel().astype(dtype), gather.reshape(8, P).astype(dtype), np.stack(back).astype(dtype)


def lay_out(row, gather: np.ndarray, wall: int) -> np.ndarray:
    """The views of a row through canvas_views' gather: int64 [len(gather), side**2], pad 0 and the ring `wall`."""
    tokens = np.empty(gather.shape[1] + 2, np.int64)
    tokens[:-2], tokens[-2:] = np.asarray(row).reshape(-1), (0, wall)
    return tokens[gather]


def on_canvas(board: np.ndarray, side: int, wall: int, dtype=np.int64) -> np.ndarray:
    """`board` at (1, 1) of an empty side x side canvas, inside a ring of `wall`, flattened."""
    h, w = board.shape
    canvas = np.zeros((side, side), dtype)
    canvas[:h + 2, :w + 2] = wall
    canvas[1:h + 1, 1:w + 1] = board
    return canvas.reshape(-1)


class Task:
    def __init__(self, pool: Pool):
        self.pool = pool
        self.X = pool.inputs
        self.Y = pool.labels

    def __len__(self) -> int:
        return self.X.shape[0]

    # answers
    def pin(self, i: int, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def valid(self, i: int, y: np.ndarray) -> bool:
        raise NotImplementedError

    def consistent(self, i: int, y: np.ndarray) -> bool:
        return bool(np.array_equal(self.pin(i, y), y))

    def pin_batch(self, idx, Y) -> np.ndarray:
        Y = np.asarray(Y)
        return np.stack([self.pin(int(i), y) for i, y in zip(idx, Y)]) if len(Y) else Y.copy()

    def valid_batch(self, idx, Y) -> np.ndarray:
        return np.array([self.valid(int(i), y) for i, y in zip(idx, np.asarray(Y))], dtype=bool)

    def check(self, scoring: str, idx, Y) -> np.ndarray:
        if scoring == "raw":
            return self.valid_batch(idx, Y)
        if scoring == "pinned":
            return self.valid_batch(idx, self.pin_batch(idx, Y))
        raise ValueError(f"unknown scoring {scoring!r}")

    # hypotheses
    def editable(self, i: int, row: np.ndarray) -> list[int]:
        raise NotImplementedError

    def alternatives(self, i: int, cell: int, current: int, ranked, answer) -> list[int]:
        raise NotImplementedError

    # restatements
    def restatements(self, i: int, row: np.ndarray) -> list:
        raise NotImplementedError

    def restate_back(self, i: int, tag, y: np.ndarray):
        raise NotImplementedError
