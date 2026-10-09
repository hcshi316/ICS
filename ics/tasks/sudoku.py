"""Sudoku-Extreme: 9x9 grids. Tokens: 0 pad, 1 blank, 2..10 the digits 1..9.
Restatements: the identity, then n_restatements - 1 random validity-preserving rewritings (band/stack and in-band
row/column permutations, an optional transpose and a digit relabelling), drawn per board from a sub-seed that the
generator of its block of `block` rows, seeded by block_seed(seed, SUDOKU_RESTATEMENTS, the block's first row), deals in
row order (ics/seeding.py); a board's rewritings are drawn at its first call and kept."""
from __future__ import annotations

import random

import numpy as np

from ics.seeding import SUDOKU_RESTATEMENTS, block_seed
from ics.tasks.base import Task, rank_order

BLANK = 1
DIGITS = np.arange(2, 11)


def make_aug(rng: random.Random):
    """One random restatement: cell k of the original grid moves to cell pos[k]; digit DIGITS[j] becomes
    DIGITS[perm[j]]."""
    band_r = rng.sample(range(3), 3)
    rows_in = [rng.sample(range(3), 3) for _ in range(3)]
    band_c = rng.sample(range(3), 3)
    cols_in = [rng.sample(range(3), 3) for _ in range(3)]
    transpose = rng.random() < 0.5
    rmap = np.array([band_r[i // 3] * 3 + rows_in[i // 3][i % 3] for i in range(9)], np.int64)
    cmap = np.array([band_c[i // 3] * 3 + cols_in[i // 3][i % 3] for i in range(9)], np.int64)
    pos = np.add.outer(rmap, 9 * cmap) if transpose else np.add.outer(9 * rmap, cmap)    # [r, c]: where r * 9 + c goes
    return pos.ravel(), rng.sample(range(9), 9)


class Sudoku(Task):
    def __init__(self, pool, n_restatements: int = 32, seed: int = 7, block: int = 768):
        super().__init__(pool)
        self.n_restatements, self.seed, self.block = n_restatements, seed, block
        self._block_seeds = {}
        self._tables = {}

    def _rng(self, i) -> random.Random:
        """Board i's restatement generator (see the module docstring)."""
        row = int(self.pool.index[i])
        start = row - row % self.block
        seeds = self._block_seeds.get(start)
        if seeds is None:
            rng = np.random.default_rng(block_seed(self.seed, SUDOKU_RESTATEMENTS, start))
            seeds = self._block_seeds[start] = rng.integers(0, 2 ** 62, size=self.block)
        return random.Random(int(seeds[row - start]))

    def _rewritings(self, i):
        """Board i's n = n_restatements - 1 rewritings, drawn at the first call (see the module docstring): int8 tables
        pos [n, 81], dmap [n, 12] and inv [n, 12]. Rewriting k moves cell c to pos[k, c] and token t to dmap[k, t];
        inv[k] undoes dmap[k]."""
        tables = self._tables.get(i)
        if tables is None:
            rng, n = self._rng(i), max(self.n_restatements - 1, 0)
            pos, perm = np.empty((n, 81), np.int8), np.empty((n, 9), np.int8)
            for k in range(n):
                pos[k], perm[k] = make_aug(rng)
            dmap = np.tile(np.arange(12, dtype=np.int8), (n, 1))
            inv = dmap.copy()
            dmap[:, DIGITS] = DIGITS[perm]
            inv[np.arange(n)[:, None], DIGITS[perm]] = DIGITS
            tables = self._tables[i] = pos, dmap, inv
        return tables

    def pin(self, i, y):
        x = self.X[i]
        out = np.array(y, copy=True)
        given = x != BLANK
        out[given] = x[given]
        return out

    def pin_batch(self, idx, Y):
        X = self.X[np.asarray(idx, dtype=np.int64)]
        return np.where(X != BLANK, X, np.asarray(Y))

    def valid(self, i, y):
        return bool(self.valid_batch([i], np.asarray(y)[None])[0])

    def valid_batch(self, idx, Y):
        Y = np.asarray(Y)
        if len(Y) == 0:
            return np.zeros(0, bool)
        X = self.X[np.asarray(idx, dtype=np.int64)]
        ok = ((Y == X) | (X == BLANK)).all(1) & np.isin(Y, DIGITS).all(1)
        G = Y.reshape(-1, 9, 9)
        boxes = G.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
        units = np.sort(np.concatenate([G, G.transpose(0, 2, 1), boxes], 1), axis=-1)
        return ok & (units == DIGITS).all((1, 2))

    def editable(self, i, row):
        return np.flatnonzero(row == BLANK).tolist()

    def alternatives(self, i, cell, current, ranked, answer):
        return [t for t in rank_order(DIGITS, ranked) if t != int(current)]

    def restatements(self, i, row):
        """A rewriting's tag is (pos, inv): pos as int64 and inv in numpy's default integer, in arrays of this call's
        own, so a caller may change them."""
        pos, dmap, inv = self._rewritings(i)
        row = np.asarray(row)
        restated = np.zeros((len(pos), *row.shape), row.dtype)
        restated[np.arange(len(pos))[:, None], pos] = dmap[:, row]
        tags = zip(pos.astype(np.int64), inv.astype(np.int_))
        return [(None, np.array(row, copy=True))] + list(zip(tags, restated))

    def restate_back(self, i, tag, y):
        if tag is None:
            return np.array(y, copy=True)
        pos, inv = tag
        return inv[y][pos]
