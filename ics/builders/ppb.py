# solution_moves is adapted from ppbench 0.1.0, dataset._decrypt_solution (MIT License: LICENSES/MIT-PPBench.txt).
"""PPBench in the TRM layout: the source shared with Light-Up (ics/builders/lightup.py), and the Nurikabe, Tapa and
Heyawake datasets. Source: Pencil Puzzle Bench, the Hugging Face dataset bluecoconut/pencil-puzzle-bench at a pinned
revision (full_dataset.jsonl, golden_300.jsonl); reading its boards needs the [ppbench] extra (pip install
ppbench==0.1.0) and Node.js 16 or newer ($NODE_BIN, else node on PATH). Boards: the type's rows of sides at most 24.
test   val = random.Random(20260808).sample of the 10x10 boards that are not golden (Nurikabe 100, Tapa 50, Heyawake
       100), then golden_300.jsonl's boards of the type; input: the board, every cell undecided; label: the solution.
train  the other boards, 10 rows each: the board, then 9 rows whose input is the state after a prefix of the solution's
       moves, each under one symmetry of the square, drawn from the board's sort_key and `seed` (a left click shades a
       cell, a right click unshades it, the last click wins). A reseeded Heyawake build may number its tokens
       differently: evaluate a model on the build it was trained on.
No test puzzle in train: a board whose puzzle (its clues; on Heyawake, its rooms and their clues) equals a test board's
under a symmetry of the square is dropped, and PPB_MANIFEST.json lists it.
Tokens (ics/tasks/ppb.py): 0 pad, 1 wall, 2 undecided, 3 shaded, 4 unshaded, then from 5 on a token per clue (Heyawake:
per cell), vocab.json mapping each key to its token. The board sits at (1, 1) inside a ring of walls. The labels are
stored encoded (ics/data.py)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ics.builders import source_file
from ics.data import encode_labels
from ics.tasks.base import on_canvas
from ics.tasks.ppb import KINDS, SIDE, T_EMPTY, T_SHADE, T_WHITE, WALL
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

REPO, REVISION = "bluecoconut/pencil-puzzle-bench", "3ac6add0754f1458bf51aab160ca039667a67b38"
VAL = {"nurikabe": 100, "tapa": 50, "heyawake": 100}
VAL_SEED, AUGMENT = 20260808, 10
STATE_TOKENS = {EMPTY: T_EMPTY, SHADE: T_SHADE, WHITE: T_WHITE}
NEEDS = ("building the PPBench datasets needs ppbench (pip install ppbench==0.1.0) and Node.js 16 or newer "
         "($NODE_BIN, else node on PATH)")


class PPBenchUnavailable(RuntimeError):
    """ppbench cannot run here: it, or a Node.js 16 or newer, is missing. The message says what to install."""


# The source, shared with Light-Up.
def read_rows(kind: str, name: str, source=None) -> list[dict]:
    """The rows of PPBench's `name` for the puzzle type `kind` whose sides are both at most SIDE - 2, in file order."""
    with open(source_file(REPO, REVISION, name, source), encoding="utf-8") as f:
        return [r for r in map(json.loads, f)
                if r.get("pid") == kind and max(int(r["width"]), int(r["height"])) <= SIDE - 2]


def solution_moves(row: dict) -> list[str]:
    """The moves of a row's solution (moves_full)."""
    key = b"ppbench"
    raw = base64.b64decode(row["solution_enc"])
    return json.loads(bytes(b ^ key[i % len(key)] for i, b in enumerate(raw)))["moves_full"]


def load_ppbench():
    """The ppbench package, loaded before any download, after a check of Node.js 16 or newer, which its import starts.
    Puzzle(url).get_string_repr() is the puzzle in pzpr.js's file format: 4 header lines, then the board."""
    name = os.environ.get("NODE_BIN") or "node"
    if (node := shutil.which(name)) is None:
        looked_up = "node on PATH" if name == "node" else f"$NODE_BIN={name}"
        raise PPBenchUnavailable(f"{NEEDS}: Node.js not found ({looked_up})")
    try:
        version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False, timeout=10,
                                 stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        raise PPBenchUnavailable(f"{NEEDS}: {node} --version failed: {e}") from e
    if not (major := re.match(r"v(\d+)\.", version)) or int(major[1]) < 16:
        raise PPBenchUnavailable(f"{NEEDS}: {node} --version gives {version!r}")
    try:
        import ppbench
    except Exception as e:                               # an ImportError, or the bridge's own failure to start
        raise PPBenchUnavailable(f"{NEEDS}: {e}") from e
    return ppbench


def clicks(moves: list[str], h: int, w: int):
    """(r, c, button) of each move on an h x w board, in order."""
    for move in moves:
        _, button, x, y = move.split(",")[:4]
        r, c = (int(y) - 1) // 2, (int(x) - 1) // 2
        if 0 <= r < h and 0 <= c < w:
            yield r, c, button


def write_split(out, split: str, inputs: np.ndarray, labels: np.ndarray, vocab_size: int, **extra) -> None:
    """One split in the layout of the released PPBench datasets: inputs as given, labels encoded (encode_labels), the
    `extra` arrays (dims, is_golden) as int32, every row its own puzzle and group (int64 indices), and dataset.json,
    which declares the encoding."""
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    arrays = {"inputs": inputs, "labels": encode_labels(labels, vocab_size),
              **{name: np.asarray(a, np.int32) for name, a in extra.items()},
              "puzzle_identifiers": np.zeros(n, np.int32), "puzzle_indices": np.arange(n + 1, dtype=np.int64),
              "group_indices": np.arange(n + 1, dtype=np.int64)}
    for name, arr in arrays.items():
        np.save(d / f"all__{name}.npy", arr)
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": vocab_size,
            "seq_len": inputs.shape[1], "num_puzzle_identifiers": 1, "total_groups": n, "mean_puzzle_examples": 1.0,
            "total_puzzles": n, "total_samples": n, "sets": ["all"], "encoded_labels": True}
    (d / "dataset.json").write_text(json.dumps(meta), newline="\n")


# Nurikabe, Tapa, Heyawake.
def parse(kind: str, text: str, h: int, w: int) -> dict:
    """A board in pzpr.js's file format, as h x w lists: "clues", and for Heyawake "rooms" (room ids).
      nurikabe  lines 4..: the cells, "." or the island size ("-": any, -1)
      tapa      lines 4..: the cells, "." or the run lengths joined by "," ("-": any, -1)
      heyawake  line 4: the number of rooms; lines 5..: the cells' room ids, then the cells' "." or room clue"""
    lines = text.strip().splitlines()
    if (int(lines[2]), int(lines[3])) != (h, w):
        raise ValueError(f"the board text is {lines[2]}x{lines[3]}, the row {h}x{w}")

    def cells(first):
        grid = [line.split() for line in lines[first:first + h]]
        if len(grid) != h or any(len(row) != w for row in grid):
            raise ValueError(f"the board text does not have {h} lines of {w} cells from line {first}")
        return grid

    number = lambda t: -1 if t == "-" else int(t)
    if kind == "heyawake":
        rooms, n = [[int(t) for t in row] for row in cells(5)], int(lines[4])
        regions = borders_to_rooms(*rooms_to_borders(rooms))
        if {i for row in rooms for i in row} != set(range(n)) or max(map(max, regions)) != n - 1:
            raise ValueError(f"the rooms are not {n} connected regions")
        return {"rooms": rooms, "clues": [[None if t == "." else int(t) for t in row] for row in cells(5 + h)]}
    clue = (lambda t: [number(x) for x in t.split(",")]) if kind == "tapa" else number
    return {"clues": [[None if t == "." else clue(t) for t in row] for row in cells(4)]}


def check(kind: str, board: dict, state) -> str:
    """The first rule `state` breaks, or "OK"."""
    if kind == "heyawake":
        return heyawake_valid(board["rooms"], board["clues"], state)
    return (tapa_valid if kind == "tapa" else nurikabe_valid)(board["clues"], state)


def replay(moves: list[str], h: int, w: int, fill: int) -> list[list[int]]:
    """The state after `moves` (a left click shades, another unshades, the last wins); unreached cells hold `fill`."""
    state = [[fill] * w for _ in range(h)]
    for r, c, button in clicks(moves, h, w):
        state[r][c] = SHADE if button == "left" else WHITE
    return state


def load(kind: str, row: dict, puzzle) -> tuple[dict, list[str], list[list[int]]]:
    """A row's board, its solution's moves and the solved state (`puzzle`: ppbench's Puzzle); an error names the row."""
    try:
        h, w = int(row["height"]), int(row["width"])
        board, moves = parse(kind, puzzle(row["puzzle_url"]).get_string_repr(), h, w), solution_moves(row)
        solved = replay(moves, h, w, WHITE)
    except (ValueError, IndexError, KeyError) as e:
        reason = f"missing field {e}" if isinstance(e, KeyError) else e
        raise ValueError(f"{kind} {row.get('sort_key')}: {reason}") from e
    if (broken := check(kind, board, solved)) != "OK":
        raise ValueError(f"{kind} {row['sort_key']}: the solution breaks the rule {broken!r}")
    return board, moves, solved


def turn(grid: list[list], k: int, mirror: bool) -> list[list]:
    """`grid` after k quarter turns (as np.rot90), then mirrored left-right if `mirror`."""
    for _ in range(k):
        grid = [list(row) for row in zip(*grid)][::-1]
    return [row[::-1] if mirror else list(row) for row in grid]


def puzzle_key(kind: str, board: dict) -> tuple:
    """What `board` gives, without any state: its clues (a Tapa clue's runs sorted), and for Heyawake its rooms in
    reading order, each cell with its room's clue. Two boards in one frame pose the same puzzle iff their keys agree."""
    if kind != "heyawake":
        clue = (lambda c: None if c is None else tuple(sorted(c))) if kind == "tapa" else (lambda c: c)
        return tuple(tuple(clue(c) for c in row) for row in board["clues"])
    number, of_room = {}, {}
    rooms = [[number.setdefault(room, len(number)) for room in row] for row in board["rooms"]]
    for row, clues in zip(rooms, board["clues"]):
        for room, c in zip(row, clues):
            if c is not None:
                of_room[room] = c
    return tuple(tuple((room, of_room.get(room)) for room in row) for row in rooms)


def twin(kind: str, board: dict, given: dict) -> dict | None:
    """The first test board whose puzzle `board` poses under one of the 8 maps (turn's k and mirror): {"test": its
    sort_key, "k": k, "mirror": mirror}, or None. `given` maps each test board's puzzle_key to its sort_key."""
    for k in range(4):
        for mirror in (False, True):
            key = puzzle_key(kind, {name: turn(grid, k, mirror) for name, grid in board.items()})
            if key in given:
                return {"test": given[key], "k": k, "mirror": mirror}
    return None


class Tokens:
    """The tokens from 5 on, in order of first use: `table` maps each clue's key (each Heyawake cell's) to its token."""

    def __init__(self, kind: str):
        self.kind, self.table = kind, {}

    def _token(self, key) -> int:
        return self.table.setdefault(key, 5 + len(self.table))

    def _clue(self, clue) -> int:
        return self._token(("nu", clue) if self.kind == "nurikabe" else ("ta", ",".join(map(str, sorted(clue)))))

    def grid(self, board: dict, state) -> np.ndarray:
        """The [h, w] tokens of `board` in `state`, numbered row by row."""
        clues = board["clues"]
        h, w = len(clues), len(clues[0])
        if self.kind == "heyawake":
            rb, db = rooms_to_borders(board["rooms"])
            return np.array([[self._token(("hy", rb[r][c], db[r][c], clues[r][c], state[r][c])) for c in range(w)]
                             for r in range(h)], np.int32)
        return np.array([[STATE_TOKENS[state[r][c]] if clues[r][c] is None else self._clue(clues[r][c])
                          for c in range(w)] for r in range(h)], np.int32)


def build_ppb(out, kind: str, seed: int = 0, source=None) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if seed < 0:
        raise ValueError(f"seed {seed}: the seed must be 0 or more")
    puzzle = load_ppbench().Puzzle
    pool = read_rows(kind, "full_dataset.jsonl", source)
    golden = read_rows(kind, "golden_300.jsonl", source)
    gkeys = {r["sort_key"] for r in golden}
    tens = [r for r in pool if int(r["width"]) == int(r["height"]) == 10 and r["sort_key"] not in gkeys]
    others = sum(r["sort_key"] not in gkeys for r in pool)
    if len(tens) < VAL[kind] or others <= VAL[kind]:
        raise ValueError(f"{kind}: too small a source: {others} boards outside golden_300.jsonl, {len(tens)} of them "
                         f"10x10; the build needs {VAL[kind]} 10x10 boards for val and 1 more board for train")
    val = random.Random(VAL_SEED).sample(tens, VAL[kind])
    vkeys = {r["sort_key"] for r in val}
    test_keys = vkeys | gkeys
    train = [r for r in pool if r["sort_key"] not in test_keys]
    tokens = Tokens(kind)
    pair = lambda board, state, solved: (on_canvas(tokens.grid(board, state), SIDE, WALL, np.int32),
                                         on_canvas(tokens.grid(board, solved), SIDE, WALL, np.int32))
    test, dims, given = [], [], {}
    for row in val + golden:
        board, _, solved = load(kind, row, puzzle)
        h, w = len(solved), len(solved[0])
        test.append(pair(board, replay([], h, w, EMPTY), solved))
        dims.append((h, w, 1, 1))
        given.setdefault(puzzle_key(kind, board), row["sort_key"])
    rows, dropped = [], []
    for row in train:
        board, moves, solved = load(kind, row, puzzle)
        if (match := twin(kind, board, given)) is not None:        # a test board's puzzle: no train board
            dropped.append({"sort_key": row["sort_key"], **match})
            print(f"{kind}: train board {row['sort_key']} dropped: its puzzle is test board {match['test']}'s "
                  f"(k={match['k']}{', mirrored' if match['mirror'] else ''})", flush=True)
            continue
        h, w = len(solved), len(solved[0])
        rows.append(pair(board, replay([], h, w, EMPTY), solved))
        rng = random.Random(int(hashlib.md5(row["sort_key"].encode()).hexdigest()[:8], 16) + (seed << 32))
        for _ in range(AUGMENT - 1):
            prefix = replay(moves[:rng.randint(1, max(1, len(moves) - 1))], h, w, EMPTY)
            k, mirror = rng.randrange(4), rng.random() < 0.5
            view = {name: turn(grid, k, mirror) for name, grid in board.items()}
            rows.append(pair(view, turn(prefix, k, mirror), turn(solved, k, mirror)))
    if not rows:
        raise ValueError(f"{kind}: every train board poses a test board's puzzle; the build would have no train split")
    vocab_size = 5 + len(tokens.table)
    X, Y = (np.stack(a) for a in zip(*rows))
    write_split(out, "train", X, Y, vocab_size)
    X, Y = (np.stack(a) for a in zip(*test))
    write_split(out, "test", X, Y, vocab_size, dims=dims, is_golden=[0] * len(val) + [1] * len(golden))
    root = Path(out)
    (root / "identifiers.json").write_text(json.dumps(["<blank>"]), newline="\n")
    (root / "vocab.json").write_text(json.dumps({str(k): v for k, v in tokens.table.items()}, indent=1), newline="\n")
    manifest = {"pid": kind, "canvas": SIDE, "pool": len(pool), "train_boards": len(train) - len(dropped),
                "aug": AUGMENT, "val": len(val), "golden": len(golden), "vocab_size": vocab_size, "val_seed": VAL_SEED,
                "val_keys": sorted(vkeys), "golden_keys": sorted(gkeys)}
    if seed:
        manifest["seed"] = seed
    if dropped:
        manifest["dropped"] = dropped
    (root / "PPB_MANIFEST.json").write_text(json.dumps(manifest, indent=1), newline="\n")
