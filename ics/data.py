# Adapted from github.com/SamsungSAILMontreal/TinyRecursiveModels@c0110373 (puzzle_dataset.py; MIT License: see
# ics/trm/layers.py). Modified.
"""Datasets in the TRM layout: evaluation pools and the training stream.

A built dataset is a directory with one folder per split:
    <data>/<split>/dataset.json          metadata (pad_id, ignore_label_id, vocab_size, seq_len, total_groups, ...)
    <data>/<split>/all__inputs.npy       [N, L] input tokens
    <data>/<split>/all__labels.npy       [N, L] label tokens (ignore_label_id where unlabelled); the verifier's: [N].
                                         Stored encoded (encode_labels) where dataset.json has "encoded_labels": true,
                                         as the Light-Up and PPB builders write them; the loaders decode them
    <data>/<split>/all__puzzle_identifiers.npy, all__puzzle_indices.npy, all__group_indices.npy
                                         puzzles of one example (a row) each, and groups of puzzles (a Sudoku board
                                         and its augmentations; one puzzle in the other tasks)
    <data>/<split>/all__dims.npy         [N, 4] (h, w, r0, c0): board placement on the canvas (Light-Up, PPB)
    <data>/<split>/all__is_golden.npy    [N] 1 for PPBench golden boards (Light-Up, PPB)
    <data>/vocab.json                    token table of the PPB puzzles
Training order (train_batches): block b = 1, 2, ... draws from Philox(seed + b) epochs_per_block permutations of the
groups and one puzzle of each group, cut into global batches; only a block's incomplete last batch is dropped.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORE = -100
DIMS = ("seq_len", "vocab_size", "num_puzzle_identifiers")      # a dataset's dimensions, which a model is built with
LABEL_KEY_SEED = 1                                              # the seed of encoded labels' key (label_key)


def label_key(seq_len: int) -> np.ndarray:
    """The key of encoded labels: an offset per position, [seq_len] int64 drawn from LABEL_KEY_SEED by RandomState,
    whose stream NumPy keeps unchanged across its versions."""
    return np.random.RandomState(LABEL_KEY_SEED).randint(0, 2**16, seq_len, dtype=np.int64)


def encode_labels(labels: np.ndarray, vocab_size: int) -> np.ndarray:
    """Labels [N, L] as an encoded split stores them, in their dtype: (label + key[j]) mod vocab_size at position j, so
    that the solutions are not stored in the clear."""
    return ((labels + label_key(labels.shape[1])) % vocab_size).astype(labels.dtype)


def decode_labels(stored: np.ndarray, vocab_size: int) -> np.ndarray:
    """The labels [N, L] that encode_labels stored as `stored`, in its dtype."""
    return ((stored - label_key(stored.shape[1])) % vocab_size).astype(stored.dtype)


@dataclass
class Pool:
    inputs: np.ndarray                 # [N, L] int64
    labels: np.ndarray                 # [N, L] int64, IGNORE where unlabelled; [N] targets in the verifier's data
    index: np.ndarray                  # [N] int64: row of each board in the full split
    meta: dict
    dims: np.ndarray | None = None
    is_golden: np.ndarray | None = None
    vocab: dict | None = None

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def take(self, rows) -> Pool:
        """The boards at `rows` (a slice or an index array) of this pool."""
        pick = lambda arr: None if arr is None else arr[rows]
        return Pool(self.inputs[rows], self.labels[rows], self.index[rows], self.meta, pick(self.dims),
                    pick(self.is_golden), self.vocab)


def load_pool(data_dir: str | Path, split: str = "test", start: int = 0, limit: int | None = None,
              rows=None) -> Pool:
    """`limit` boards of a split from `start` on (by default all of them), or the boards at `rows` (an index array into
    the split; only those rows are read)."""
    if start < 0 or (limit is not None and limit < 0):
        raise ValueError(f"start and limit must be non-negative, got start={start}, limit={limit}")
    root = Path(data_dir)
    d = root / split
    meta = json.loads((d / "dataset.json").read_text())
    sl = slice(start, None if limit is None else start + limit) if rows is None else np.array(rows, np.int64)
    inputs = np.array(np.load(d / "all__inputs.npy", mmap_mode="r")[sl], dtype=np.int64)
    labels = np.array(np.load(d / "all__labels.npy", mmap_mode="r")[sl], dtype=np.int64)
    if meta.get("encoded_labels"):
        labels = decode_labels(labels, meta["vocab_size"])
    if meta.get("ignore_label_id") is not None:
        labels[labels == meta["ignore_label_id"]] = IGNORE
    extra = {name: np.load(d / f"all__{name}.npy")[sl] for name in ("dims", "is_golden")
             if (d / f"all__{name}.npy").exists()}
    vocab = json.loads((root / "vocab.json").read_text()) if (root / "vocab.json").exists() else None
    index = np.arange(start, start + inputs.shape[0], dtype=np.int64) if rows is None else sl
    return Pool(inputs, labels, index, meta, extra.get("dims"), extra.get("is_golden"), vocab)


def dataset_dims(data_dir, split: str = "test") -> dict:
    """DIMS of a built dataset, from a split's dataset.json: the dimensions a model for it is built with."""
    meta = json.loads((Path(data_dir) / split / "dataset.json").read_text())
    return {k: meta[k] for k in DIMS}


def check_dims(ckpt, config: dict, dims: dict) -> None:
    """Refuse the checkpoint `ckpt` if one of DIMS its model config (a dict) has differs from the dataset's `dims`."""
    if wrong := [f"{k} {config[k]} vs {dims[k]}" for k in DIMS if k in config and config[k] != dims[k]]:
        raise ValueError(f"{ckpt}: the checkpoint was built for another dataset (checkpoint vs data: "
                         f"{', '.join(wrong)})")


@dataclass
class TrainSplit:
    inputs: np.ndarray                 # [N, L], memory-mapped
    labels: np.ndarray                 # [N, L] as stored (rows decodes them), or [N] targets in the verifier's data;
                                       # memory-mapped
    puzzle_identifiers: np.ndarray     # [N]
    group_starts: np.ndarray           # [G + 1] int64: the split's group_indices
    meta: dict

    def rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """A batch of rows as the model takes it: int32 arrays, labels decoded and IGNORE where a row is unlabelled."""
        labels = self.labels[rows].astype(np.int32)
        if self.meta.get("encoded_labels"):
            labels = decode_labels(labels, self.meta["vocab_size"])
        if self.meta.get("ignore_label_id") is not None:
            labels[labels == self.meta["ignore_label_id"]] = IGNORE
        return {"inputs": self.inputs[rows].astype(np.int32), "labels": labels,
                "puzzle_identifiers": self.puzzle_identifiers[rows].astype(np.int32)}


def load_train(data_dir: str | Path) -> TrainSplit:
    d = Path(data_dir) / "train"
    load = lambda name, mode=None: np.load(d / f"all__{name}.npy", mmap_mode=mode)
    puzzles = load("puzzle_indices")
    if not np.array_equal(puzzles, np.arange(len(puzzles))):
        raise ValueError(f"{d}: every puzzle must hold exactly one example (puzzle_indices = 0, 1, 2, ...)")
    return TrainSplit(load("inputs", "r"), load("labels", "r"), load("puzzle_identifiers"),
                      load("group_indices").astype(np.int64), json.loads((d / "dataset.json").read_text()))


def train_batches(group_starts: np.ndarray, seed: int, epochs_per_block: int, global_batch: int,
                  start: tuple[int, int] = (1, 0)) -> Iterator[tuple[int, int, np.ndarray]]:
    """The training order (module docstring), endless, from position `start` = (block, batch within the block) on:
    yields (block, batch, rows), rows the [global_batch] rows of the batch; a block takes O(groups) memory."""
    n = len(group_starts) - 1
    if epochs_per_block * n < global_batch:
        raise ValueError(f"a block of {epochs_per_block} epochs of {n} groups holds no batch of {global_batch}")
    block, skip = start
    while True:
        orders = np.random.Generator(np.random.Philox(seed + block))
        picks = np.random.Generator(np.random.Philox(seed + block))
        for _ in range(epochs_per_block):
            picks.permutation(n)
        queue, batch = np.empty(0, np.int64), 0
        for _ in range(epochs_per_block):
            queue = np.concatenate([queue, orders.permutation(n)])
            while len(queue) >= global_batch:
                groups, queue = queue[:global_batch], queue[global_batch:]
                rows = picks.integers(group_starts[groups], group_starts[groups + 1])
                if batch >= skip:
                    yield block, batch, rows
                batch += 1
        block, skip = block + 1, 0
