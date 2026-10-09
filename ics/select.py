"""Selection among a board's candidates [K, n, L] (candidate k of board j): PTRM's rollouts, GRAM's samples, EqR's and
Attractor's restarts. Each rule returns, per board, the index of the candidate it selects:
  first_valid(task, idx, cands, scoring)   the first candidate the certificate accepts in `scoring`; -1 if none
  best(scores)                             the largest score [K, n]; the first on ties
  majority(cands)                          the most frequent answer; between answers equally frequent, the first
take(cands, k) gives the answers, candidate 0 where k is -1. blocks runs a sampling method block by block, pick writes
one block's commits under the keys of ics/methods/__init__.py, and join concatenates the blocks.
"""
from __future__ import annotations

import numpy as np

from ics.methods import ANSWER_DTYPE
from ics.tasks.base import SCORINGS, Task


def first_valid(task: Task, idx: np.ndarray, cands: np.ndarray, scoring: str) -> np.ndarray:
    K, n, L = cands.shape
    ok = task.check(scoring, np.tile(idx, K), cands.reshape(K * n, L)).reshape(K, n)
    return np.where(ok.any(0), ok.argmax(0), -1).astype(np.int64)


def best(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores).argmax(0).astype(np.int64)


def majority(cands: np.ndarray) -> np.ndarray:
    out = np.zeros(cands.shape[1], np.int64)
    for j in range(cands.shape[1]):
        _, first, counts = np.unique(cands[:, j], axis=0, return_index=True, return_counts=True)
        out[j] = first[counts == counts.max()].min()
    return out


def take(cands: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cands[np.maximum(k, 0), np.arange(cands.shape[1])]


def pick(task: Task, idx: np.ndarray, cands: np.ndarray, rows: dict[str, np.ndarray],
         extra: str) -> dict[str, np.ndarray]:
    """One block's commits. rows maps a table row to the candidate it commits without a certificate ([n] indices);
    with one, every row commits the first candidate the certificate accepts, in each scoring (candidate 0 and index -1
    if it accepts none). The index of each commit is stored under the extra's name."""
    cert = {s: first_valid(task, idx, cands, s) for s in SCORINGS}
    out = {}
    for row, k in rows.items():
        out[f"{row}/model/{extra}"] = np.asarray(k, np.int64)
        for s in SCORINGS:
            out[f"{row}/model/{s}"] = take(cands, k)
            out[f"{row}/cert/{s}"] = take(cands, cert[s])
            out[f"{row}/cert/{s}/{extra}"] = cert[s]
    return out


def join(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def blocks(task: Task, batch: int, rows: tuple[str, ...], extra: str, sample) -> dict[str, np.ndarray]:
    """A sampling method on every board of `task`, in blocks of `batch` consecutive boards. sample(inputs [n, L], row0)
    gives a block's candidates [K, n, L] and {row: [n] the candidate it commits without a certificate}; row0, the
    block's first row in the test split, seeds its draws (ics/seeding.py). An empty task gives empty arrays."""
    parts = []
    for s0 in range(0, len(task), batch):
        idx = np.arange(s0, min(len(task), s0 + batch))
        cands, chosen = sample(task.X[idx], int(task.pool.index[s0]))
        parts.append(pick(task, idx, cands, chosen, extra))
    if not parts:
        empty = np.zeros((1, 0, task.X.shape[1]), ANSWER_DTYPE)
        parts.append(pick(task, np.zeros(0, np.int64), empty, dict.fromkeys(rows, np.zeros(0, np.int64)), extra))
    return join(parts)
