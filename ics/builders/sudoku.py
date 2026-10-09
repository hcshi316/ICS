# Adapted from TinyRecursiveModels@c0110373 dataset/build_sudoku_dataset.py (MIT: ics/trm/layers.py), which adapts
# github.com/sapientinc/HRM's (Apache License 2.0: LICENSES/Apache-2.0.txt). Modified.
"""Sudoku-Extreme in the TRM layout, as TRM's README builds it (--subsample-size 1000 --num-aug 1000), seeded.

train  `subsample` boards drawn from train.csv, each a group: the board, then `augment` validity-preserving rewritings
       (a digit relabelling, an optional transpose, band and in-band row and column permutations).
test   test.csv as it is, one group per board.
Tokens: 0 pad, 1 blank, 2..10 the digits 1..9. All draws come from np.random.RandomState(seed); inputs and labels are
uint8.
"""
from __future__ import annotations

import numpy as np

from ics.builders import read_csv, source_file, write_split

REPO, REVISION = "sapientinc/sudoku-extreme", "58942f96baeb572ca3127e2a9e9c70f330783d6b"


def digits(rows: list[str]) -> np.ndarray:
    """[N, 81] uint8 digits of the CSV rows, 0 for a blank ('.')."""
    if lengths := sorted({len(row) for row in rows} - {81}):
        raise ValueError(f"rows of length {lengths}, not 81")
    cells = np.frombuffer("".join(rows).replace(".", "0").encode(), np.uint8) - ord("0")
    if (cells > 9).any():
        raise ValueError(f"characters outside '.0123456789': {sorted(set(''.join(rows)) - set('.0123456789'))}")
    return cells.reshape(-1, 81)


def shuffle_sudoku(rs: np.random.RandomState, board: np.ndarray, solution: np.ndarray):
    """One random rewriting of a flat board and its solution (the original shuffle_sudoku, drawing from rs)."""
    digit_map = np.pad(rs.permutation(np.arange(1, 10)), (1, 0))
    transpose = rs.rand() < 0.5
    bands = rs.permutation(3)
    row_perm = np.concatenate([b * 3 + rs.permutation(3) for b in bands])
    stacks = rs.permutation(3)
    col_perm = np.concatenate([s * 3 + rs.permutation(3) for s in stacks])
    mapping = (row_perm[:, None] * 9 + col_perm[None, :]).reshape(-1)

    def apply(x: np.ndarray) -> np.ndarray:
        grid = x.reshape(9, 9)
        return digit_map[(grid.T if transpose else grid).reshape(-1)[mapping]]

    return apply(board), apply(solution)


def build_sudoku(out, seed: int = 0, subsample: int | None = 1000, augment: int = 1000, source=None) -> None:
    rs = np.random.RandomState(seed)
    splits = {}
    for split in ("train", "test"):
        columns = read_csv(source_file(REPO, REVISION, f"{split}.csv", source))
        try:
            boards, solutions = (digits(c) for c in columns)
        except ValueError as e:
            raise ValueError(f"{split}.csv: {e}") from e
        if not solutions.all():
            raise ValueError(f"{split}.csv: answers with a blank ('.' or '0')")
        splits[split] = boards, solutions
    for split, (boards, solutions) in splits.items():
        if split == "train" and subsample is not None and subsample < len(boards):
            pick = rs.choice(len(boards), size=subsample, replace=False)
            boards, solutions = boards[pick], solutions[pick]
        per = 1 + (augment if split == "train" else 0)
        inputs = np.zeros((len(boards) * per, 81), np.uint8)
        labels = np.zeros_like(inputs)
        for i, (board, solution) in enumerate(zip(boards, solutions)):
            inputs[i * per], labels[i * per] = board, solution
            for k in range(1, per):
                inputs[i * per + k], labels[i * per + k] = shuffle_sudoku(rs, board, solution)
        write_split(out, split, inputs + 1, labels + 1, np.full(len(boards), per), vocab_size=11)
