"""GRAM as a test-time protocol: N samples of D supervision steps per board (the paper's width and depth scaling), each
a stochastic trajectory from the fixed z0 under the prior, decoded after the last step. Without a certificate,
gram_lprm commits the sample with the largest LPRM value sigmoid(v) and gram_majority the most frequent decode (the
first on ties); with one, both commit the first sample the certificate accepts (sample 0's decode if it accepts none).
Boards run in blocks of `batch`; sample i's guidance noise on a block comes from a generator seeded by block_seed(seed,
GRAM_NOISE, i, the block's first row in the test split) (ics/seeding.py), so whole-block shards do not change it.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ics.seeding import GRAM_NOISE, block_seed, torch_generator
from ics.select import best, blocks, majority
from ics.tasks.base import Task
from ics.trm.roll import device_batch


@torch.no_grad()
def samples(model, xb: torch.Tensor, N: int, D: int,
            generator: Callable[[int], torch.Generator]) -> tuple[np.ndarray, np.ndarray]:
    """N samples of the block xb [B, L] (int32, on the model's device): decodes [N, B, L] (int16) and LPRM values
    sigmoid(v) [N, B] (float32) after D steps. Sample i draws its noise from generator(i)."""
    decs, values = [], []
    for i in range(N):
        gen = generator(i)
        z_H, z_L = model.initial_state(xb.shape[0])
        for _ in range(D):
            z_H, z_L, logits, _q, v = model.step(z_H, z_L, xb, gen)
        decs.append(logits.argmax(-1).to(torch.int16).cpu())
        values.append(torch.sigmoid(v.float()).cpu())
    return torch.stack(decs).numpy(), torch.stack(values).numpy()


@torch.no_grad()
def run_gram(model, task: Task, N: int = 100, D: int = 64, batch: int = 128, seed: int = 0) -> dict[str, np.ndarray]:
    """N samples of every board: answers (ANSWER_DTYPE) under "gram_{lprm,majority}/{model,cert}/<scoring>" plus int64
    extras, "<row>/model/sample" (without a certificate) and "<row>/cert/<scoring>/sample" (the first accepted; -1:
    none)."""
    if N < 1 or D < 1 or batch < 1:
        raise ValueError(f"run_gram needs N >= 1, D >= 1 and batch >= 1, got N={N}, D={D}, batch={batch}")

    def sample(inputs, row0):
        xb, n = device_batch(inputs, batch, model.device)
        generator = lambda i: torch_generator(block_seed(seed, GRAM_NOISE, i, row0), model.device)
        decs, values = samples(model, xb, N, D, generator)
        decs, values = decs[:, :n], values[:, :n]
        return decs, {"gram_lprm": best(values), "gram_majority": majority(decs)}

    return blocks(task, batch, ("gram_lprm", "gram_majority"), "sample", sample)
