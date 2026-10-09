# Adapted from TinyRecursiveModels@c0110373 dataset/build_{sudoku,maze}_dataset.py (MIT: ics/trm/layers.py), which
# adapt github.com/sapientinc/HRM's (Apache License 2.0: LICENSES/Apache-2.0.txt). Modified.
"""Dataset builders: python -m ics data --task TASK --out DIR writes a dataset in the TRM layout (ics/data.py).

  sudoku    1,000 boards of sapientinc/sudoku-extreme's train.csv, 1,000 augmentations each; test.csv as it is
  maze      sapientinc/maze-30x30-hard-1k as it is
  lightup, nurikabe, tapa, heyawake
            PPBench's boards of the type, of side at most 24, on a 26x26 canvas (ics/builders/lightup.py, ppb.py)
The sources are downloaded from Hugging Face at a pinned revision, or read from `source`, a directory holding them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def source_file(repo: str, revision: str, name: str, source=None) -> Path:
    """`name` from the directory `source` if given, else from the Hugging Face dataset `repo` at `revision`."""
    if source is not None:
        return Path(source) / name
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))


def read_csv(path: Path) -> tuple[list[str], list[str]]:
    """The question and answer columns (the 2nd and 3rd) of a source CSV, in file order. A CSV without rows below its
    header, or with a row of fewer than 3 columns, is refused."""
    questions, answers = [], []
    with open(path, newline="") as f:
        rows = csv.reader(f)
        next(rows, None)
        for row in rows:
            if len(row) < 3:
                raise ValueError(f"{path}, line {rows.line_num}: fewer than 3 columns")
            questions.append(row[1])
            answers.append(row[2])
    if not questions:
        raise ValueError(f"{path}: no rows below the header")
    return questions, answers


def overlap(out, data, names: tuple[str, str]) -> str | None:
    """Why a build from the dataset `data` must not write into `out`, or None: `out` is that dataset, lies inside it or
    holds it (compared as resolved paths, so other spellings and symlinks count). The reason calls the two by `names`:
    the command line's ("--out", "--data") or the Python API's ("out", "data")."""
    o, d = Path(out).resolve(), Path(data).resolve()
    where = "is" if o == d else "is inside" if d in o.parents else "holds" if o in d.parents else None
    out_name, data_name = names
    return None if where is None else (f"{out_name} {out} {where} the dataset ({data_name} {data}); choose another "
                                       f"{out_name}")


def token_dtype(vocab_size: int) -> type:
    """The dtype write_split stores tokens in: uint8, or int32 when the vocabulary needs more than a byte."""
    return np.uint8 if vocab_size <= 256 else np.int32


def write_split(out, split: str, inputs: np.ndarray, labels: np.ndarray, group_sizes: np.ndarray,
                vocab_size: int, **meta) -> None:
    """One split: inputs [N, L] and labels [N, L] (token 0 = pad, also the ignored label) or [N] (one per row, the
    verifier's targets), group g of group_sizes[g] puzzles of one example each, and dataset.json, whose fields `meta`
    adds to or replaces. Tokens are stored in token_dtype(vocab_size)."""
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    arrays = {"inputs": inputs, "labels": labels, "group_indices": np.concatenate([[0], np.cumsum(group_sizes)]),
              "puzzle_indices": np.arange(n + 1), "puzzle_identifiers": np.zeros(n)}
    tokens = token_dtype(vocab_size)
    for name, arr in arrays.items():
        np.save(d / f"all__{name}.npy", arr.astype(tokens if name in ("inputs", "labels") else np.int32))
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": vocab_size,
            "seq_len": inputs.shape[1], "num_puzzle_identifiers": 1, "total_groups": len(group_sizes),
            "mean_puzzle_examples": 1.0, "total_puzzles": n, "sets": ["all"], **meta}
    (d / "dataset.json").write_text(json.dumps(meta), newline="\n")
    (Path(out) / "identifiers.json").write_text(json.dumps(["<blank>"]), newline="\n")
