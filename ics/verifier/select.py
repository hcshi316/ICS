"""Selection (run_ics_verifier). Each board is solved under each of its restatements (T segments each); every decode is
mapped back onto the board and pinned, and is one of the board's candidates (a decode that cannot be mapped back is
none). Each distinct candidate is scored once: the verifier's q_halt after verifier_T segments, averaged over the
candidate's own restatements. The highest-scoring candidate is committed (the first on ties).
"""
from __future__ import annotations

import numpy as np

from ics.methods import ANSWER_DTYPE
from ics.tasks.base import SCORINGS, Task
from ics.trm.roll import roll


def run_ics_verifier(solver, verifier, task: Task, T: int = 16, verifier_T: int = 8,
                     batch: int = 250) -> dict[str, np.ndarray]:
    """Every board's candidate selected by the verifier: answers (ANSWER_DTYPE) under "ics/model/<scoring>", which
    share one array (treat it as read-only), plus int64 extras: "ics/model/segs", the TRM segments rolled for the
    board (T per restatement decoded, verifier_T per view of each candidate scored), and "ics/model/restatement", the
    first restatement whose decode was committed (-1: no candidate, and the input, pinned, is committed)."""
    if T < 1 or verifier_T < 1 or batch < 1:
        raise ValueError(f"run_ics_verifier needs T >= 1, verifier_T >= 1 and batch >= 1, got T={T}, "
                         f"verifier_T={verifier_T}, batch={batch}")
    X, n = task.X, len(task)
    if n == 0:
        answers = np.zeros(X.shape, ANSWER_DTYPE)
        return {**{f"ics/model/{s}": answers for s in SCORINGS},
                "ics/model/segs": np.zeros(0, np.int64), "ics/model/restatement": np.zeros(0, np.int64)}
    views = [task.restatements(i, X[i]) for i in range(n)]
    K = max(len(v) for v in views)
    cands, found = np.zeros((K, n, X.shape[1]), np.int64), np.zeros((K, n), bool)
    segs = np.zeros(n, np.int64)
    for k in range(K):
        members = [i for i in range(n) if k < len(views[i])]
        dec = roll(solver, np.stack([views[i][k][1] for i in members]), T, batch, topk=1).dec
        for i, d in zip(members, dec):
            segs[i] += T
            y = task.restate_back(i, views[i][k][0], d)
            if y is not None:
                cands[k, i], found[k, i] = task.pin(i, y), True
    scored = found.copy()                                           # each distinct candidate, where it first comes
    for i in range(n):
        seen = set()
        for k in np.flatnonzero(found[:, i]):
            key = cands[k, i].tobytes()
            scored[k, i] = key not in seen
            seen.add(key)
    scores = np.full((K, n), -np.inf, np.float32)
    for k in range(K):
        members = np.flatnonzero(scored[k])
        restated = {i: task.restatements(i, cands[k, i]) for i in members}
        total, count = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for v in range(max((len(r) for r in restated.values()), default=0)):
            mem = [i for i in members if v < len(restated[i])]
            total[mem] += roll(verifier, np.stack([restated[i][v][1] for i in mem]), verifier_T, batch, topk=1).q
            count[mem] += 1
            segs[mem] += verifier_T
        mean = np.divide(total, count, out=np.full(n, -np.inf, np.float32), where=count > 0)
        scores[k, members] = mean[members]
    restatement = np.full(n, -1, np.int64)
    for i in np.flatnonzero(found.any(0)):                          # the board's best candidate, the first on ties,
        ks = np.flatnonzero(found[:, i])                            # even when each of them scores -inf
        restatement[i] = ks[scores[ks, i].argmax()]
    committed = np.stack([cands[r, i] if r >= 0 else task.pin(i, X[i]) for i, r in enumerate(restatement)])
    out = dict.fromkeys((f"ics/model/{s}" for s in SCORINGS), committed.astype(ANSWER_DTYPE))
    out["ics/model/segs"] = segs
    out["ics/model/restatement"] = restatement
    return out
