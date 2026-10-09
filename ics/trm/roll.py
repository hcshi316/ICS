"""Rolling a frozen TRM on input rows: T >= 1 recursion segments on the same inputs, from the model's initial latent or
from a given state (e.g. one an earlier roll kept, to continue it), halting disabled. Rows are processed in chunks of
`batch`; a final short chunk is padded with copies of its first row up to a multiple of PAD_MULTIPLE rows (never
beyond `batch`). A noise hook, hook(t, z_L) -> z_L for t = 1..T, may perturb z_L before each segment, the same hook in
every chunk.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

PAD_MULTIPLE = 8
Hook = Callable[[int, torch.Tensor], torch.Tensor]
State = tuple[torch.Tensor, torch.Tensor]       # the latent (z_H, z_L), one row per input row


def chunk_rows(n_real: int, batch: int) -> int:
    """Rows actually run for a chunk holding `n_real` real rows."""
    if n_real >= batch:
        return batch
    return min(batch, -(-n_real // PAD_MULTIPLE) * PAD_MULTIPLE)


def _pad(t: torch.Tensor, size: int) -> torch.Tensor:
    """`t` with copies of its first row appended up to `size` rows."""
    if t.shape[0] < size:
        t = torch.cat([t, t[:1].expand(size - t.shape[0], *t.shape[1:])], 0)
    return t


def device_batch(rows: np.ndarray, batch: int, device) -> tuple[torch.Tensor, int]:
    xb = torch.from_numpy(np.ascontiguousarray(rows).astype(np.int32)).to(device)
    n = xb.shape[0]
    return _pad(xb, chunk_rows(n, batch)), n


@torch.no_grad()
def segments(model, xb: torch.Tensor, T: int, noise: Hook | None = None, *,
             state: State | None = None) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, State]]:
    """Yield (t, logits, q_halt, (z_H, z_L)) after each segment t = 1..T of one chunk. The chunk starts from `state`
    (a row per row of xb, on its device) or else from the model's initial state."""
    ids = torch.zeros(xb.shape[0], dtype=torch.int32, device=xb.device)
    z_H, z_L = model.initial_state(xb.shape[0]) if state is None else state
    for t in range(1, T + 1):
        if noise is not None:
            z_L = noise(t, z_L)
        z_H, z_L, logits, q = model.segment(z_H, z_L, xb, ids)
        yield t, logits, q, (z_H, z_L)


@dataclass
class Rolled:
    dec: np.ndarray         # [N, L] argmax of the last segment's logits
    q: np.ndarray           # [N] q_halt of the last segment
    top_ids: np.ndarray     # [N, L, k] the k best tokens per cell, best first
    top_vals: np.ndarray    # [N, L, k] their logits
    state: State | None = None   # (z_H, z_L) after the last segment, on the CPU in the forward dtype (keep_state only)


def _check_state(model, state: State, n: int):
    """Refuse a start state that is not a row per input row, each row shaped like a row of model.initial_state and in
    its dtype, the model's forward dtype."""
    for name, z, z0 in zip(("z_H", "z_L"), state, model.initial_state(1)):
        want = (n, *z0.shape[1:])
        if tuple(z.shape) != want:
            raise ValueError(f"roll needs a state of shape {want} for {n} rows, got {name} of shape {tuple(z.shape)}")
        if z.dtype != z0.dtype:
            raise ValueError(f"roll needs a state in the model's forward dtype {z0.dtype}, got {name} in {z.dtype}")


@torch.no_grad()
def roll(model, rows: np.ndarray, T: int, batch: int, noise: Hook | None = None, topk: int = 10, *,
         state: State | None = None, keep_state: bool = False) -> Rolled:
    """Roll `rows` for T segments. `state`: the latent (z_H, z_L) to start from, on any device and in the model's
    forward dtype, a row per row of `rows` shaped like a row of model.initial_state (by default every row starts from
    the initial state). `keep_state`: also return the real rows' final latent, as Rolled.state."""
    if T < 1 or batch < 1:
        raise ValueError(f"roll needs T >= 1 and batch >= 1, got T={T}, batch={batch}")
    rows = np.asarray(rows)
    if state is not None:
        _check_state(model, state, rows.shape[0])
    decs, qs, ids, vals = [], [], [], []
    final = None
    if keep_state:                                  # filled chunk by chunk: the CPU holds the kept state only once
        final = tuple(torch.empty((rows.shape[0], *z0.shape[1:]), dtype=z0.dtype, device="cpu")
                      for z0 in model.initial_state(1))
    for s in range(0, rows.shape[0], batch):
        xb, n = device_batch(rows[s:s + batch], batch, model.device)
        start = None if state is None else tuple(_pad(z[s:s + n].to(model.device), xb.shape[0]) for z in state)
        for _t, logits, q, latent in segments(model, xb, T, noise, state=start):
            pass
        lg = logits.float()
        top_v, top_i = lg.topk(min(topk, lg.shape[-1]), dim=-1)
        decs.append(lg.argmax(-1)[:n].cpu().numpy())
        qs.append(q.float()[:n].cpu().numpy())
        ids.append(top_i[:n].cpu().numpy())
        vals.append(top_v[:n].cpu().numpy())
        if keep_state:
            for out, z in zip(final, latent):
                out[s:s + n].copy_(z[:n])
        del start, latent                           # a chunk's latent does not stay on the device into the next chunk
    if not decs:
        L = rows.shape[1]
        return Rolled(np.zeros((0, L), np.int64), np.zeros(0, np.float32), np.zeros((0, L, topk), np.int64),
                      np.zeros((0, L, topk), np.float32), final)
    return Rolled(np.concatenate(decs).astype(np.int64), np.concatenate(qs), np.concatenate(ids).astype(np.int64),
                  np.concatenate(vals), final)
