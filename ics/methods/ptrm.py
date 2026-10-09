"""PTRM: perturbation-based width scaling of a frozen TRM (Sghaier et al., 2026), as a test-time protocol: K rollouts
of D segments per board. Rollout 0 is the clean greedy roll; in rollout k > 0, sigma * N(0, 1) is added to z_L before
every segment after the first. Without a certificate the rollout with the largest final q_halt is committed (the first
on ties); with a certificate, the first rollout whose decode the certificate accepts (rollout 0's decode if it accepts
none), in each scoring. Boards are rolled in blocks of `batch`; rollout k's noise on a block comes from a generator
seeded by block_seed(seed, PTRM_NOISE, k, the block's first row in the test split) (ics/seeding.py).
"""
from __future__ import annotations

import numpy as np
import torch

from ics.methods import ANSWER_DTYPE
from ics.seeding import PTRM_NOISE, block_seed, torch_generator
from ics.select import best, blocks
from ics.tasks.base import Task
from ics.trm.roll import roll


class BlockNoise:
    """The noise hook of one rollout on one block, drawn from the generator of `seeds` (ics/seeding.py)."""

    def __init__(self, sigma: float, seeds: np.random.SeedSequence, device):
        self.sigma = sigma
        self.gen = torch_generator(seeds, device)

    def __call__(self, t: int, z_L: torch.Tensor) -> torch.Tensor:
        if t == 1:
            return z_L
        eps = torch.randn(z_L.shape, generator=self.gen, device=z_L.device, dtype=torch.float32)
        return z_L + (self.sigma * eps).to(z_L.dtype)


@torch.no_grad()
def run_ptrm(model, task: Task, K: int = 100, D: int = 64, sigma: float = 0.3, batch: int = 256,
             seed: int = 0) -> dict[str, np.ndarray]:
    """K rollouts of every board: answers (ANSWER_DTYPE) under "ptrm/{model,cert}/<scoring>" plus int64 extras,
    "ptrm/model/rollout", the rollout committed without a certificate, and "ptrm/cert/<scoring>/rollout", the first the
    certificate accepts (-1: none). Answers under "/pinned" keys are decodes too, pinned at scoring (task.check)."""
    if K < 1 or D < 1 or batch < 1:
        raise ValueError(f"run_ptrm needs K >= 1, D >= 1 and batch >= 1, got K={K}, D={D}, batch={batch}")

    def sample(inputs, row0):
        decs, qs = np.zeros((K, *inputs.shape), ANSWER_DTYPE), np.zeros((K, len(inputs)), np.float32)
        for k in range(K):
            noise = (BlockNoise(sigma, block_seed(seed, PTRM_NOISE, k, row0), model.device)
                     if k > 0 and sigma != 0 else None)
            r = roll(model, inputs, D, batch, noise=noise, topk=1)               # the block is the roll's one chunk
            decs[k], qs[k] = r.dec, r.q
        return decs, {"ptrm": best(qs)}

    return blocks(task, batch, ("ptrm",), "rollout", sample)
