"""Light-Up (Akari) from PPBench in the TRM layout; the source, its pin and the optional ppbench and Node.js are those
of ics/builders/ppb.py. Boards: PPBench's Light-Up rows of sides at most 24, in file order, without a board equal to an
earlier one, or to a golden_300.jsonl board, under a symmetry of the square.
test   val = 100 of the 10x10 boards (np.random.default_rng(20260808)), then golden_300.jsonl's Light-Up boards that
       fit: 113 rows.
train  every other board in its 8 symmetries (k quarter turns of np.rot90, each followed by its left-right mirror).
Tokens (ics/tasks/lightup.py): 0 pad, 1 empty, 2 wall, 3..7 walls numbered 0..4, 8 bulb; a left click puts a bulb.
The board sits at (1, 1) inside a ring of walls; dims (h, w, 1, 1) in both splits, is_golden in the test split. The
labels are stored encoded (ics/data.py).
"""
from __future__ import annotations

import numpy as np

from ics.builders import ppb
from ics.tasks.base import DIHEDRAL, on_canvas, turn
from ics.tasks.lightup import BULB, EMPTY, W0, WALL, akari_valid

VAL, VAL_SEED = 100, 20260808
CELLS = {".": EMPTY, "-": WALL, **{str(n): W0 + n for n in range(5)}}


def parse(row: dict, puzzle) -> tuple[np.ndarray, np.ndarray]:
    """A row's board and solution, [h, w] uint8 tokens; `puzzle` is ppbench's Puzzle, whose board text holds h lines of
    w cells after its header. An error names the row."""
    try:
        h, w = int(row["height"]), int(row["width"])
        lines = [line.split() for line in puzzle(row["puzzle_url"]).get_string_repr().strip().splitlines()[4:4 + h]]
        if len(lines) != h or any(len(line) != w for line in lines):
            raise ValueError(f"the board text does not have {h} lines of {w} cells")
        if unknown := [c for line in lines for c in line if c not in CELLS]:
            raise ValueError(f"unknown cell {unknown[0]!r}")
        board = np.array([[CELLS[c] for c in line] for line in lines], np.uint8)
        solution = board.copy()
        for r, c, button in ppb.clicks(ppb.solution_moves(row), h, w):
            if button == "left":
                solution[r, c] = BULB
    except (ValueError, IndexError, KeyError) as e:
        reason = f"missing field {e}" if isinstance(e, KeyError) else e
        raise ValueError(f"lightup {row.get('sort_key')}: {reason}") from e
    if not akari_valid(board, solution):
        raise ValueError(f"lightup {row['sort_key']}: the solution breaks the Akari rules")
    return board, solution


def symmetries(grid: np.ndarray) -> list[np.ndarray]:
    """The grid's 8 symmetries (DIHEDRAL): k = 0..3 quarter turns (np.rot90), each followed by its left-right mirror."""
    return [turn(grid, k4, mir) for k4, mir in DIHEDRAL]


def canonical(grid: np.ndarray) -> tuple:
    """Equal for two grids iff one is a symmetry of the other: the least (shape, bytes) of the grid's symmetries."""
    return min((v.shape, v.tobytes()) for v in symmetries(grid))


def canvas_rows(pairs) -> tuple[np.ndarray, np.ndarray, list]:
    """Inputs and labels [N, 676] uint8 and dims of (board, solution) pairs, each at (1, 1) inside a ring of walls."""
    X, Y, dims = [], [], []
    for board, solution in pairs:
        X.append(on_canvas(board, ppb.SIDE, WALL))
        Y.append(on_canvas(solution, ppb.SIDE, WALL))
        dims.append((*board.shape, 1, 1))
    return np.stack(X).astype(np.uint8), np.stack(Y).astype(np.uint8), dims


def build_lightup(out, source=None) -> None:
    puzzle = ppb.load_ppbench().Puzzle
    rows = ppb.read_rows("lightup", "full_dataset.jsonl", source)
    golden_rows = ppb.read_rows("lightup", "golden_300.jsonl", source)
    pool, forms = {}, set()                              # sort_key -> (board, solution), in file order
    for row in rows:
        board, solution = parse(row, puzzle)
        if (form := canonical(board)) not in forms:
            forms.add(form)
            pool[row["sort_key"]] = board, solution
    golden = [parse(row, puzzle) for row in golden_rows]
    gkeys, gforms = {row["sort_key"] for row in golden_rows}, {canonical(board) for board, _ in golden}
    pool = {key: p for key, p in pool.items() if key not in gkeys and canonical(p[0]) not in gforms}
    tens = [key for key, (board, _) in pool.items() if board.shape == (10, 10)]
    if len(tens) < VAL or len(pool) <= VAL:
        raise ValueError(f"lightup: too small a source: {len(pool)} boards outside golden_300.jsonl, {len(tens)} of "
                         f"them 10x10; the build needs {VAL} 10x10 boards for val and 1 more board for train")
    val = [tens[i] for i in sorted(np.random.default_rng(VAL_SEED).choice(len(tens), size=VAL, replace=False))]
    vkeys = set(val)
    train = [pair for key, (board, solution) in pool.items() if key not in vkeys
             for pair in zip(symmetries(board), symmetries(solution))]
    X, Y, dims = canvas_rows([pool[key] for key in val] + golden)
    ppb.write_split(out, "test", X, Y, BULB + 1, dims=dims, is_golden=[0] * len(val) + [1] * len(golden))
    X, Y, dims = canvas_rows(train)
    ppb.write_split(out, "train", X, Y, BULB + 1, dims=dims)
