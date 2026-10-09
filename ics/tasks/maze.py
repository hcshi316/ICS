"""Maze-Hard: 30x30 mazes. Tokens: 0 pad, 1 wall, 2 empty, 3 start, 4 goal, 5 path.

An answer is valid when every given cell (wall, start, goal) is unchanged, start and goal are joined through path cells
by a shortest path, and every path cell lies on such a shortest path (several may be marked). Any other token on an
empty cell is not a path cell, so raw scoring ignores junk tokens on empty cells off the path. Restatements: the 8
dihedral symmetries of the grid (DIHEDRAL order), each a permutation of the cells: view k of a row is row[VIEWS[k]],
and an answer to it maps back as y[BACKS[k]].
Hypotheses are path hypotheses only: an ICS hypothesis writes PATH into an empty cell, never WALL, and a cell already
on the parent answer's path gets none.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ics.tasks.base import DIHEDRAL, Task, turn

SIDE = 30
WALL, EMPTY, START, GOAL, PATH = 1, 2, 3, 4, 5
VIEWS = np.stack([turn(np.arange(SIDE * SIDE).reshape(SIDE, SIDE), k4, mir).ravel()
                  for k4, mir in DIHEDRAL]).astype(np.int16)
BACKS = np.argsort(VIEWS, axis=1).astype(np.int16)


def _bfs(cells: set, src: int) -> dict:
    """Distances from src walking on `cells`."""
    dist, q = {src: 0}, deque([src])
    while q:
        c = q.popleft()
        r, co = divmod(c, SIDE)
        for nr, nc in ((r - 1, co), (r + 1, co), (r, co - 1), (r, co + 1)):
            if 0 <= nr < SIDE and 0 <= nc < SIDE:
                n = nr * SIDE + nc
                if n in cells and n not in dist:
                    dist[n] = dist[c] + 1
                    q.append(n)
    return dist


def rule_valid(x: np.ndarray, y: np.ndarray) -> str:
    """"OK", or the first failed check: "given" (a wall/start/goal changed), "disc" (start and goal not connected
    through path cells), "longer" (connected but not shortest), "stray" (a path cell on no shortest start-goal path
    through path cells)."""
    walls = set(np.flatnonzero(x == WALL).tolist())
    s, g = int(np.flatnonzero(x == START)[0]), int(np.flatnonzero(x == GOAL)[0])
    givens = np.flatnonzero(x != EMPTY)
    if (y[givens] != x[givens]).any():
        return "given"
    path = set(np.flatnonzero(y == PATH).tolist())
    walk = path | {s, g}
    ds = _bfs(walk, s)
    if g not in ds:
        return "disc"
    if ds[g] != _bfs({c for c in range(SIDE * SIDE) if c not in walls}, s)[g]:
        return "longer"
    dg = _bfs(walk, g)
    on_path = {c for c in ds if c in dg and ds[c] + dg[c] == ds[g]}
    return "OK" if len(on_path) == len(walk) else "stray"


class Maze(Task):
    def pin(self, i, y):
        x = self.X[i]
        out = np.array(y, copy=True)
        given = x != EMPTY
        out[given] = x[given]
        out[~given & (out != EMPTY) & (out != PATH)] = EMPTY
        return out

    def valid(self, i, y):
        return rule_valid(self.X[i], np.asarray(y)) == "OK"

    def editable(self, i, row):
        return np.flatnonzero((self.X[i] == EMPTY) & (row == EMPTY)).tolist()

    def alternatives(self, i, cell, current, ranked, answer):
        return [PATH] if int(current) != PATH else []

    def restatements(self, i, row):
        return list(zip(DIHEDRAL, np.asarray(row).reshape(SIDE * SIDE)[VIEWS]))

    def restate_back(self, i, tag, y):
        k4, mir = tag
        return np.asarray(y).reshape(SIDE * SIDE)[BACKS[DIHEDRAL.index((k4, mir))]]
