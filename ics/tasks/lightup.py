"""Light-Up (Akari) from PPBench on a 26x26 canvas. Tokens: 0 pad, 1 empty, 2 wall, 3..7 walls numbered 0..4, 8 bulb.
Board i occupies rows r0:r0+h and columns c0:c0+w of the canvas (pool.dims[i] = (h, w, r0, c0)), inside a ring of
walls. An answer is valid when, on the board, every given cell is unchanged, every other cell is empty or a bulb, and
the Akari rules hold: every empty cell is lit, no two bulbs see each other, every numbered wall has that many adjacent
bulbs. Restatements: the 8 dihedral symmetries of the board (DIHEDRAL order), each re-embedded at (1, 1) inside its own
wall ring, through index maps kept per placement (ics/tasks/base.py canvas_views)."""
from __future__ import annotations

import numpy as np

from ics.tasks.base import DIHEDRAL, Task, canvas_views, lay_out

EMPTY, WALL, W0, BULB = 1, 2, 3, 8


def akari_valid(grid: np.ndarray, lab: np.ndarray) -> bool:
    h, w = grid.shape
    walls = grid >= WALL
    bulbs = lab == BULB
    if (bulbs & walls).any():
        return False
    lit = np.zeros_like(bulbs)
    for r in range(h):
        for c in range(w):
            if not bulbs[r, c]:
                continue
            lit[r, c] = True
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w and not walls[rr, cc]:
                    if bulbs[rr, cc]:
                        return False
                    lit[rr, cc] = True
                    rr += dr
                    cc += dc
    if ((grid == EMPTY) & ~lit & ~bulbs).any():
        return False
    for r in range(h):
        for c in range(w):
            t = int(grid[r, c])
            if W0 <= t <= W0 + 4:
                n = sum(bool(0 <= r + dr < h and 0 <= c + dc < w and bulbs[r + dr, c + dc])
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                if n != t - W0:
                    return False
    return True


class LightUp(Task):
    def __init__(self, pool):
        super().__init__(pool)
        if pool.dims is None:
            raise ValueError("Light-Up needs all__dims.npy in the data directory")
        self.side = round(self.X.shape[1] ** 0.5)
        assert self.side * self.side == self.X.shape[1]
        self.dims = pool.dims
        self._maps = {}

    def _views(self, i):
        """Board i's restatement tags and index maps (cells, gather, back), made at the first call for its placement
        and kept (see the module docstring)."""
        key = tuple(self.dims[i].tolist())
        views = self._maps.get(key)
        if views is None:
            h, w = self.board(i, self.X[i]).shape
            tags = [(k4, mir, (w, h, 1, 1) if k4 % 2 else (h, w, 1, 1)) for k4, mir in DIHEDRAL]
            views = self._maps[key] = (tags, *canvas_views(self.side, key))
        return views

    def board(self, i, row):
        h, w, r0, c0 = self.dims[i]
        return np.asarray(row).reshape(self.side, self.side)[r0:r0 + h, c0:c0 + w]

    def _put(self, i, board):
        h, w, r0, c0 = self.dims[i]
        canvas = self.X[i].reshape(self.side, self.side).copy()
        canvas[r0:r0 + h, c0:c0 + w] = board
        return canvas.reshape(-1)

    def pin(self, i, y):
        g, d = self.board(i, self.X[i]), self.board(i, y).copy()
        given = g != EMPTY
        d[given] = g[given]
        d[~given & (d != EMPTY) & (d != BULB)] = EMPTY
        return self._put(i, d)

    def valid(self, i, y):
        g, d = self.board(i, self.X[i]), self.board(i, y)
        given = g != EMPTY
        if (d[given] != g[given]).any() or not np.isin(d[~given], (EMPTY, BULB)).all():
            return False
        return akari_valid(g, d)

    def editable(self, i, row):
        _h, _w, r0, c0 = self.dims[i]
        free = (self.board(i, self.X[i]) == EMPTY) & (self.board(i, row) == EMPTY)
        return [int((r0 + r) * self.side + c0 + c) for r, c in zip(*np.nonzero(free))]

    def alternatives(self, i, cell, current, ranked, answer):
        return [BULB] if int(current) != BULB else []

    def restatements(self, i, row):
        """Tagged (k4, mir, dims): the view's quarter turns, its mirror, and its placement, (h, w, 1, 1) of the view."""
        tags, _cells, gather, _back = self._views(i)
        return list(zip(tags, lay_out(row, gather, WALL)))

    def restate_back(self, i, tag, y):
        k4, mir, _dims = tag
        _tags, cells, _gather, back = self._views(i)
        out = self.X[i].copy()
        out[cells] = np.asarray(y).reshape(self.side * self.side)[back[DIHEDRAL.index((k4, mir))]]
        return out
