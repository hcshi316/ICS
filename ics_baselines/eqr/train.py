# Adapted from github.com/locuslab/EqR@aba94e9cde0f273ce644db5261cd6915ba6561f0 (Apache License 2.0: LICENSE in this
# directory). Modified.
"""Training EqR: TRMHead (ics/trm/train.py) with two differences:
  - a restarting row's latent is drawn afresh (z_H, then z_L, truncated normal), for the restarting rows only;
  - a step is EqR's (ics_baselines/eqr/model.py): H_cycles cycles of damped, noisy updates, and only the last cycle
    carries gradient.
Optimizer: AdamATan2 on every weight (ics/optim.py). Evaluation: one restart of halt_max_steps steps per board
(ics_baselines/eqr/predict.py), scored raw.
"""
from __future__ import annotations

import numpy as np
import torch

from ics.optim import AdamATan2
from ics.tasks.base import Task
from ics.trm.train import TRMHead
from ics_baselines.eqr.model import EqR, EqRConfig
from ics_baselines.eqr.predict import run_eqr


class EqRHead(TRMHead):
    config_class = EqRConfig
    model_class = EqR

    @torch.compiler.disable                 # the rows depend on the data: upstream's _reset_rows runs eager too
    def restart(self, carry: dict, halted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows' latents at this step: fresh draws for the rows that restart (halted), their own for the others."""
        rows = halted.nonzero().flatten()
        if len(rows) == 0:
            return carry["z_H"], carry["z_L"]
        z_H, z_L = self.model.initial_state(len(rows), None)
        return carry["z_H"].index_put((rows,), z_H), carry["z_L"].index_put((rows,), z_L)

    def segment(self, z_H: torch.Tensor, z_L: torch.Tensor, data: dict) -> tuple:
        return self.model.step(z_H, z_L, data["inputs"], None)

    def optimizers(self, lr: float, weight_decay: float, betas) -> list[tuple[torch.optim.Optimizer, float]]:
        return [(AdamATan2(self.model.parameters(), lr, betas, weight_decay), lr)]

    @staticmethod
    def decode(model: EqR, task: Task, batch: int) -> np.ndarray:
        """Every board's decode: one restart of halt_max_steps steps (run_eqr's restart 0)."""
        return run_eqr(model, task, R=1, steps=model.config.halt_max_steps, batch=batch)["eqr/model/raw"]
