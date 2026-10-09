"""Attractor as a test-time protocol: R restarts of every board, restart 0 from the model's z0 = (H_init, L_init) and
restart k > 0 from z0 + (sigma * eH_k, sigma * eL_k), eH_k and eL_k [hidden] standard normal vectors (drawn and scaled
in float32, then cast to the model dtype) shared by every board of the block. Each restart runs `segments` segments and
is decoded after the last. Without a certificate the restart with the largest final q_halt is committed (the first on
ties); with one, the first restart the certificate accepts (restart 0's decode if it accepts none). Boards run in
unpadded blocks of `batch`, since the solver couples a block's boards; a block's perturbations come from a generator
seeded by block_seed(seed, ATTRACTOR_RESTARTS, the block's first row in the test split) (ics/seeding.py).
"""
from __future__ import annotations

import numpy as np
import torch

from ics.seeding import ATTRACTOR_RESTARTS, block_seed, torch_generator
from ics.select import best, blocks
from ics.tasks.base import Task


@torch.no_grad()
def restarts(model, xb: torch.Tensor, R: int, sigma: float, segments: int,
             gen: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """R restarts of the boards xb [B, L] (int32, on the model's device), solved together: decodes [R, B, L] (int16)
    and q_halt [R, B] (float32) after `segments` segments."""
    ids = torch.zeros(xb.shape[0], dtype=torch.int32, device=xb.device)
    decs, qs = [], []
    for k in range(R):
        dH = dL = None
        if k > 0 and sigma != 0:
            dH = sigma * torch.randn(model.config.hidden_size, generator=gen, device=xb.device, dtype=torch.float32)
            dL = sigma * torch.randn(model.config.hidden_size, generator=gen, device=xb.device, dtype=torch.float32)
        z_H, z_L = model.initial_state(xb.shape[0], dH, dL)
        for _ in range(segments):
            z_H, z_L, logits, q = model.segment(z_H, z_L, xb, ids)
        decs.append(logits.argmax(-1).to(torch.int16).cpu())
        qs.append(q.float().cpu())
    return torch.stack(decs).numpy(), torch.stack(qs).numpy()


@torch.no_grad()
def run_attractor(model, task: Task, R: int = 128, sigma: float = 0.5, segments: int = 16, batch: int = 128,
                  seed: int = 0) -> dict[str, np.ndarray]:
    """R restarts of every board: answers (ANSWER_DTYPE) under "attractor/{model,cert}/<scoring>" plus int64 extras,
    "attractor/model/restart" (without a certificate) and "attractor/cert/<scoring>/restart" (the first accepted; -1:
    none)."""
    if R < 1 or segments < 1 or batch < 1:
        raise ValueError(f"run_attractor needs R >= 1, segments >= 1 and batch >= 1, got R={R}, segments={segments}, "
                         f"batch={batch}")

    def sample(inputs, row0):
        xb = torch.from_numpy(inputs.astype(np.int32)).to(model.device)
        decs, qs = restarts(model, xb, R, sigma, segments,
                            torch_generator(block_seed(seed, ATTRACTOR_RESTARTS, row0), model.device))
        return decs, {"attractor": best(qs)}

    return blocks(task, batch, ("attractor",), "restart", sample)
