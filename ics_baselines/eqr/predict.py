# Adapted from github.com/locuslab/EqR@aba94e9cde0f273ce644db5261cd6915ba6561f0 (Apache License 2.0: LICENSE in this
# directory). Modified.
"""EqR as a test-time protocol: R independent restarts of every board (EqR's `different_init`), each from its own
truncated-normal latent, run for `steps` ACT steps with EqR's update noise, which stays on at inference, and decoded
after the last. Without a certificate the restart with the largest final q_halt is committed (the first on ties); with
one, the first restart the certificate accepts (restart 0's decode if it accepts none), in each scoring. Boards run in
blocks of `batch` (row b * R + r is restart r of board b); all of a block's randomness comes from one generator seeded
by block_seed(seed, EQR_RESTARTS, the block's first row in the test split) (ics/seeding.py).
"""
from __future__ import annotations

import numpy as np
import torch

from ics.seeding import EQR_RESTARTS, block_seed, torch_generator
from ics.select import best, blocks
from ics.tasks.base import Task
from ics.trm.roll import device_batch


@torch.no_grad()
def restarts(model, xb: torch.Tensor, R: int, steps: int, gen: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """R restarts of every board of the block xb [B, L] (int32, on the model's device): decodes [R, B, L] (int16) and
    q_halt [R, B] (float32) after `steps` steps, everything drawn from `gen`."""
    B = xb.shape[0]
    x = xb.repeat_interleave(R, dim=0)                         # row b * R + r: restart r of board b
    z_H, z_L = model.initial_state(x.shape[0], gen)
    for _ in range(steps):
        z_H, z_L, logits, q = model.step(z_H, z_L, x, gen)
    decs = logits.argmax(-1).to(torch.int16).view(B, R, -1).cpu().numpy()
    return decs.transpose(1, 0, 2), q.float().view(B, R).cpu().numpy().T


@torch.no_grad()
def run_eqr(model, task: Task, R: int = 128, steps: int = 16, batch: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """R restarts of every board: answers (ANSWER_DTYPE) under "eqr/{model,cert}/<scoring>" plus int64 extras,
    "eqr/model/restart" (without a certificate) and "eqr/cert/<scoring>/restart" (the first accepted; -1: none)."""
    if R < 1 or steps < 1 or batch < 1:
        raise ValueError(f"run_eqr needs R >= 1, steps >= 1 and batch >= 1, got R={R}, steps={steps}, batch={batch}")

    def sample(inputs, row0):
        xb, n = device_batch(inputs, batch, model.device)
        decs, qs = restarts(model, xb, R, steps, torch_generator(block_seed(seed, EQR_RESTARTS, row0), model.device))
        return decs[:, :n], {"eqr": best(qs[:, :n])}

    return blocks(task, batch, ("eqr",), "restart", sample)
