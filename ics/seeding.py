"""Block seeding: every random quantity is drawn per block of `batch` consecutive boards of the test split.

`batch` is a protocol setting. A block's generator is seeded by block_seed(seed, purpose, *key), where the key holds the
block's first row in the full test split (and, for example, the PTRM rollout). A result therefore does not depend on how
the split is sharded into whole blocks, and samplers with different purpose ids never share a stream.
"""
from __future__ import annotations

import numpy as np
import torch

# Purpose ids. A later sampler takes a new id; an id is never renumbered or reused.
SUDOKU_RESTATEMENTS = 1     # ics/tasks/sudoku.py: the restatements of the boards of a block
PTRM_NOISE = 2              # ics/methods/ptrm.py: the z_L noise of one rollout on one block
GRAM_NOISE = 3              # ics_baselines/gram/predict.py: the guidance noise of one sample on one block
EQR_RESTARTS = 4            # ics_baselines/eqr/predict.py: the initial latents and update noise of a block's restarts
ATTRACTOR_RESTARTS = 5      # ics_baselines/attractor/predict.py: the initial-latent perturbations of a block's restarts


def block_seed(seed: int, purpose: int, *key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([seed, purpose, *key])


def torch_generator(seeds: np.random.SeedSequence, device) -> torch.Generator:
    """A torch generator on `device` seeded with the first 64-bit word of `seeds`, top bit cleared."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seeds.generate_state(1, np.uint64)[0]) & (2 ** 63 - 1))
    return gen
