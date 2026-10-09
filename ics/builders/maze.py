# Adapted from TinyRecursiveModels@c0110373 dataset/build_maze_dataset.py (MIT: ics/trm/layers.py), which adapts
# github.com/sapientinc/HRM's (Apache License 2.0: LICENSES/Apache-2.0.txt). Modified.
"""Maze-Hard in the TRM layout: both splits hold their CSV's 30x30 mazes in file order, one group per maze, without
augmentation. Tokens: 0 pad, then the characters of "# SGo" (wall, empty, start, goal, path) as 1..5.
"""
from __future__ import annotations

import numpy as np

from ics.builders import read_csv, source_file, write_split

REPO, REVISION = "sapientinc/maze-30x30-hard-1k", "549754de3d67eebe57b6b22c7d226b9412d55b07"
CHARSET = "# SGo"


def tokens(rows: list[str]) -> np.ndarray:
    """[N, 900] uint8 tokens of the CSV rows."""
    if lengths := sorted({len(row) for row in rows} - {900}):
        raise ValueError(f"rows of length {lengths}, not 900 (30x30)")
    table = np.zeros(256, np.uint8)
    table[[ord(c) for c in CHARSET]] = np.arange(1, len(CHARSET) + 1)
    chars = np.frombuffer("".join(rows).encode(), np.uint8)
    if not table[chars].all():
        raise ValueError(f"characters outside {CHARSET!r}: {sorted(set(map(chr, chars[table[chars] == 0])))}")
    return table[chars].reshape(len(rows), -1)


def build_maze(out, source=None) -> None:
    splits = {}
    for split in ("train", "test"):
        columns = read_csv(source_file(REPO, REVISION, f"{split}.csv", source))
        try:
            splits[split] = [tokens(c) for c in columns]
        except ValueError as e:
            raise ValueError(f"{split}.csv: {e}") from e
    for split, (mazes, paths) in splits.items():
        write_split(out, split, mazes, paths, np.ones(len(mazes), np.int64), vocab_size=len(CHARSET) + 1)
