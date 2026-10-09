"""Standard TRM, + greedy depth and + depth scaling, read off one deep roll of every board.

The frozen TRM is rolled for T segments (6,400 in the paper). Without a certificate:
  standard        commits the decode at segment `standard_at` (16)
  greedy_depth    commits the decode at the first segment with q_halt > 0 (segment T's decode if q never fires)
  depth_scaling   commits the decode at the segment with the largest q_halt (the earliest on ties)
With a certificate, greedy_depth and depth_scaling both commit the first decode on the trajectory that the certificate
accepts (segment T's decode if it accepts none); standard is the same in both regimes. The certificate runs in each
scoring (raw, pinned), so each scoring has its own certificate answer. A decode is checked only when it differs from the
row's previous decode.
"""
from __future__ import annotations

import numpy as np
import torch

from ics.methods import ANSWER_DTYPE
from ics.tasks.base import SCORINGS, Task
from ics.trm.roll import device_batch, segments


@torch.no_grad()
def run_trm(model, task: Task, T: int = 6400, batch: int = 256, standard_at: int = 16) -> dict[str, np.ndarray]:
    """One deep roll of every board: answers (ANSWER_DTYPE) under "<row>/<regime>/<scoring>" plus int64 extras,
    "<row>/model/segment", the segment greedy_depth and depth_scaling commit without a certificate, and
    "<row>/cert/<scoring>/segment", the first the certificate accepts (-1: none). Answers under "/pinned" keys are
    decodes too, pinned at scoring (task.check). Keys holding the same values share one array: treat them as read-only.
    """
    if not 1 <= standard_at <= T or batch < 1:
        raise ValueError("run_trm needs 1 <= standard_at <= T and batch >= 1, "
                         f"got standard_at={standard_at}, T={T}, batch={batch}")
    N, L = task.X.shape
    standard, greedy, deep = (np.zeros((N, L), ANSWER_DTYPE) for _ in range(3))
    first = {s: np.zeros((N, L), ANSWER_DTYPE) for s in SCORINGS}
    t_first = {s: np.full(N, -1, np.int64) for s in SCORINGS}
    t_greedy, t_deep = np.zeros(N, np.int64), np.zeros(N, np.int64)
    for s0 in range(0, N, batch):
        idx = np.arange(s0, min(N, s0 + batch))
        xb, n = device_batch(task.X[idx], batch, model.device)
        B, dev = xb.shape[0], xb.device
        best_q = torch.full((B,), -float("inf"), device=dev)
        best = torch.zeros((B, L), dtype=torch.long, device=dev)
        best_t = torch.zeros(B, dtype=torch.long, device=dev)
        fired = torch.zeros(B, dtype=torch.bool, device=dev)
        fire_dec = torch.zeros((B, L), dtype=torch.long, device=dev)
        fire_t = torch.zeros(B, dtype=torch.long, device=dev)
        unresolved = {s: np.ones(n, bool) for s in SCORINGS}
        prev = None
        for t, logits, q, latent in segments(model, xb, T):
            dec = logits.float().argmax(-1)
            q = q.float()
            better = q > best_q
            best_q = torch.where(better, q, best_q)
            best_t = torch.where(better, torch.full_like(best_t, t), best_t)
            best[better] = dec[better]
            new = (q > 0) & ~fired
            fire_dec[new] = dec[new]
            fire_t[new] = t
            fired |= new
            if t == standard_at:
                standard[idx] = dec[:n].cpu().numpy()
            changed = torch.ones(B, dtype=torch.bool, device=dev) if prev is None else (dec != prev).any(-1)
            prev = dec
            todo = changed[:n].cpu().numpy() & np.logical_or.reduce([unresolved[s] for s in SCORINGS])
            if todo.any():
                rows = np.flatnonzero(todo)
                decoded = dec[:n][torch.as_tensor(rows, device=dev)].cpu().numpy()
                for s in SCORINGS:
                    open_ = unresolved[s][rows]
                    if not open_.any():
                        continue
                    ok = task.check(s, idx[rows[open_]], decoded[open_])
                    hit = rows[open_][ok]
                    first[s][idx[hit]] = decoded[open_][ok]
                    t_first[s][idx[hit]] = t
                    unresolved[s][hit] = False
        del latent                                  # the chunk's latent does not stay on the device into the next chunk
        last = dec[:n].cpu().numpy()
        fired_n = fired[:n].cpu().numpy()
        greedy[idx] = np.where(fired_n[:, None], fire_dec[:n].cpu().numpy(), last)
        t_greedy[idx] = np.where(fired_n, fire_t[:n].cpu().numpy(), T)
        deep[idx] = best[:n].cpu().numpy()
        t_deep[idx] = best_t[:n].cpu().numpy()
        for s in SCORINGS:
            first[s][idx[unresolved[s]]] = last[unresolved[s]]
    out = {}
    for s in SCORINGS:
        out[f"standard/model/{s}"] = out[f"standard/cert/{s}"] = standard
        out[f"greedy_depth/model/{s}"] = greedy
        out[f"depth_scaling/model/{s}"] = deep
        out[f"greedy_depth/cert/{s}"] = out[f"depth_scaling/cert/{s}"] = first[s]
        out[f"greedy_depth/cert/{s}/segment"] = out[f"depth_scaling/cert/{s}/segment"] = t_first[s]
    out["greedy_depth/model/segment"] = t_greedy
    out["depth_scaling/model/segment"] = t_deep
    return out
