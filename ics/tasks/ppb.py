"""Nurikabe, Tapa and Heyawake from PPBench on a 26x26 canvas. Tokens: 0 pad, 1 wall (the ring around the board), 2
undecided, 3 shaded, 4 unshaded; clue tokens from 5 on are listed in <data>/vocab.json. A Heyawake cell is one token for
(right border, down border, room clue or None, state), state 0 undecided, 1 shaded, 2 unshaded. Board i sits at
pool.dims[i] = (h, w, r0, c0). An answer is valid when every board cell carries a legal answer token (clue cells
unchanged, other cells shaded or unshaded; Heyawake: the cell's own structure with a decided state) and the puzzle's
rules hold. Pinning restores the givens and keeps each cell's state; where a Heyawake cell has no token for that state,
the cell keeps the answer's own token (and restate_back gives None). Restatements: the 8 dihedral symmetries of the
board (DIHEDRAL order), none for a board too large for a turned canvas, and no Heyawake view with a cell without a
token; their index maps are kept per placement, and Heyawake's tokens per board."""
from __future__ import annotations

import ast

import numpy as np

from ics.tasks.base import DIHEDRAL, Task, canvas_views, lay_out, turn
from ics.tasks.ppb_rules import (
    EMPTY,
    SHADE,
    WHITE,
    borders_to_rooms,
    heyawake_valid,
    nurikabe_valid,
    rooms_to_borders,
    tapa_valid,
)

SIDE = 26
WALL, T_EMPTY, T_SHADE, T_WHITE = 1, 2, 3, 4
KINDS = ("nurikabe", "tapa", "heyawake")


def decided_tokens(kind: str, vocab: dict | None) -> list[int]:
    """The tokens of a decided cell, shaded or unshaded, in a `kind` dataset of vocabulary `vocab` (its vocab.json); a
    Heyawake cell's token holds its state, so Heyawake needs the vocabulary."""
    if kind != "heyawake":
        return [T_SHADE, T_WHITE]
    if vocab is None:
        raise ValueError("PPB('heyawake') needs vocab.json in the data directory")
    return [v for k, v in vocab.items() if (key := ast.literal_eval(k))[0] == "hy" and key[4] != EMPTY]


class PPB(Task):
    def __init__(self, pool, kind: str):
        super().__init__(pool)
        if kind not in KINDS or pool.dims is None or pool.vocab is None:
            raise ValueError(f"PPB({kind!r}) needs a kind in {KINDS} and all__dims.npy + vocab.json "
                             "in the data directory")
        self.kind = kind
        self.dims = pool.dims
        self.tok = {ast.literal_eval(k): v for k, v in pool.vocab.items()}
        self.rev = {v: k for k, v in self.tok.items()}
        self.shaded_tokens = ([v for v, k in self.rev.items() if k[0] == "hy" and k[4] == SHADE] if kind == "heyawake"
                              else [T_SHADE])
        self.info = [self._parse(i) for i in range(len(self))]
        self._maps, self._tables = {}, {}
        if kind == "heyawake":              # each token's state, and the token of each (rb, db, clue, state); -1: none
            hy = {v: k for v, k in self.rev.items() if k[0] == "hy"}
            self._clue_ids = {clue: j for j, clue in enumerate(dict.fromkeys(k[3] for k in hy.values()))}
            self._token_state = np.full(max(self.rev, default=0) + 2, -1, np.int8)
            self._hy_tokens = np.full((2, 2, len(self._clue_ids), 3), -1, np.int16)
            for v, (_, rb, db, clue, state) in hy.items():
                self._token_state[v] = state
                self._hy_tokens[rb, db, self._clue_ids[clue], state] = v

    # structures
    def _crop(self, row, i):
        h, w, r0, c0 = self.dims[i]
        return np.asarray(row).reshape(SIDE, SIDE)[r0:r0 + h, c0:c0 + w]

    def _parse(self, i):
        g = self._crop(self.X[i], i)
        h, w = g.shape
        clues = [[None] * w for _ in range(h)]
        if self.kind == "heyawake":
            rb, db = [[0] * w for _ in range(h)], [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    key = self.rev.get(int(g[r, c]))
                    assert key is not None and key[0] == "hy", (i, r, c, g[r, c])
                    _, rb[r][c], db[r][c], clues[r][c], _ = key
            return {"h": h, "w": w, "rooms": borders_to_rooms(rb, db), "clues": clues, "rb": rb, "db": db}
        for r in range(h):
            for c in range(w):
                t = int(g[r, c])
                if t >= 5:
                    key = self.rev[t]
                    clues[r][c] = (key[1] if self.kind == "nurikabe"
                                   else [(-1 if x == "-" else int(x)) for x in key[1].split(",")])
        return {"h": h, "w": w, "clues": clues}

    def _state(self, i, row):
        """Board state grid (lists) of an answer."""
        return self._states(self._crop(row, i)).tolist()

    def _states(self, d):
        """State grid of a grid of board tokens: SHADE where the token is shaded, WHITE for every other token."""
        return np.where(np.isin(d, self.shaded_tokens), SHADE, WHITE)

    def _cell_token(self, i, r, c, state):
        """Token of board cell (r, c) in `state` (None when Heyawake has no such token)."""
        info = self.info[i]
        if self.kind == "heyawake":
            return self.tok.get(("hy", info["rb"][r][c], info["db"][r][c], info["clues"][r][c], state))
        return T_SHADE if state == SHADE else T_WHITE

    # answers
    def pin(self, i, y):
        out = self.X[i].copy()
        board, answer = out.reshape(SIDE, SIDE), np.asarray(y).reshape(SIDE, SIDE)
        info = self.info[i]
        _h, _w, r0, c0 = self.dims[i]
        st = self._state(i, y)
        for r in range(info["h"]):
            for c in range(info["w"]):
                if self.kind != "heyawake" and info["clues"][r][c] is not None:
                    continue
                tok = self._cell_token(i, r, c, st[r][c])
                board[r0 + r, c0 + c] = answer[r0 + r, c0 + c] if tok is None else tok   # no token: the answer's stays
        return out

    def valid(self, i, y):
        if not np.array_equal(self._crop(self.pin(i, y), i), self._crop(y, i)):
            return False
        info, st = self.info[i], self._state(i, y)
        if self.kind == "heyawake":
            return heyawake_valid(info["rooms"], info["clues"], st) == "OK"
        if self.kind == "tapa":
            return tapa_valid(info["clues"], st) == "OK"
        return nurikabe_valid(info["clues"], st) == "OK"

    def consistent(self, i, y):
        """Clue cells (and Heyawake's room structure) unchanged; the states are not looked at."""
        d, g = self._crop(y, i), self._crop(self.X[i], i)
        if self.kind == "heyawake":
            for r in range(d.shape[0]):
                for c in range(d.shape[1]):
                    key, gk = self.rev.get(int(d[r, c])), self.rev[int(g[r, c])]
                    if key is None or key[0] != "hy" or key[1:4] != gk[1:4]:
                        return False
            return True
        mask = g >= 5
        return bool((d[mask] == g[mask]).all())

    # hypotheses
    def editable(self, i, row):
        info = self.info[i]
        _h, _w, r0, c0 = self.dims[i]
        out = []
        for r in range(info["h"]):
            for c in range(info["w"]):
                if self.kind != "heyawake" and info["clues"][r][c] is not None:
                    continue
                cell = (r0 + r) * SIDE + (c0 + c)
                if int(row[cell]) == int(self.X[i][cell]):
                    out.append(int(cell))
        return out

    def alternatives(self, i, cell, current, ranked, answer):
        """The cell's state in the parent answer, flipped (a single token)."""
        info = self.info[i]
        _h, _w, r0, c0 = self.dims[i]
        r, c = cell // SIDE - r0, cell % SIDE - c0
        if not (0 <= r < info["h"] and 0 <= c < info["w"]):
            return []
        state = self._state(i, answer)[r][c]
        tok = self._cell_token(i, r, c, WHITE if state == SHADE else SHADE)
        return [] if tok is None else [int(tok)]

    # restatements
    def _views(self, i):
        """Board i's index maps (cells, gather, back), made at the first call for its placement and kept (see the module
        docstring); None for a board too large for a turned canvas."""
        key = tuple(self.dims[i].tolist())
        if key not in self._maps:
            fits = max(self.info[i]["h"], self.info[i]["w"]) <= SIDE - 2
            self._maps[key] = canvas_views(SIDE, key) if fits else None
        return self._maps[key]

    def _hy_tables(self, i):
        """Board i's Heyawake tokens, made at the first call and kept (int16, -1 where the vocabulary has none):
        views [8, n, 3], the token by which restatement k shows board cell j in state s; own [n, 3], cell j's own token
        in state s."""
        tables = self._tables.get(i)
        if tables is None:
            info = self.info[i]
            rooms = np.array(info["rooms"])
            clues = np.array([[self._clue_ids[clue] for clue in line] for line in info["clues"]])
            cells = np.arange(rooms.size).reshape(rooms.shape)
            views = np.empty((8, rooms.size, 3), np.int16)
            for k, (k4, mir) in enumerate(DIHEDRAL):
                rb, db = rooms_to_borders(turn(rooms, k4, mir).tolist())
                views[k, turn(cells, k4, mir).ravel()] = self._hy_tokens[rb, db, turn(clues, k4, mir)].reshape(-1, 3)
            own = self._hy_tokens[info["rb"], info["db"], clues].reshape(-1, 3)
            tables = self._tables[i] = views, own
        return tables

    def _hy_states(self, g):
        """The state of each token of a Heyawake grid, row by row; KeyError for a token that has none."""
        tokens = np.asarray(g).ravel()
        states = np.take(self._token_state, tokens, mode="clip")
        if (states < 0).any():
            raise KeyError(int(tokens[np.argmax(states < 0)]))
        return states

    def restatements(self, i, row):
        """Tagged (k4, mir): the view's quarter turns and its mirror."""
        if self.kind == "heyawake":
            states = self._hy_states(self._crop(row, i))
        views = self._views(i)
        if views is None:
            return []
        _cells, gather, back = views
        out = lay_out(row, gather, WALL)
        if self.kind != "heyawake":
            return list(zip(DIHEDRAL, out))
        tokens = self._hy_tables(i)[0][:, np.arange(len(states)), states]
        out[np.arange(8)[:, None], back] = tokens
        return [(tag, restated) for tag, restated, ok in zip(DIHEDRAL, out, (tokens >= 0).all(1)) if ok]

    def restate_back(self, i, tag, y):
        """Map the restatement's decoded states back onto board i and re-tokenise with the board's own structures."""
        k4, mir = tag
        cells, _gather, back = self._views(i)
        shaded = np.isin(np.asarray(y).reshape(SIDE * SIDE)[back[DIHEDRAL.index((k4, mir))]], self.shaded_tokens)
        out = self.X[i].copy()
        if self.kind == "heyawake":
            tokens = self._hy_tables(i)[1][np.arange(len(cells)), np.where(shaded, SHADE, WHITE)]
            if (tokens < 0).any():
                return None
            out[cells] = tokens
        else:
            free = out[cells] < 5                       # clue cells keep their token
            out[cells[free]] = np.where(shaded[free], T_SHADE, T_WHITE)
        return out
