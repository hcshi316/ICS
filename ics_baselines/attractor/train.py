# Adapted from github.com/jacobfa/Attractor@fcf045f9fb9d458f9e1574bd4036cd2fb63b3697, experiments/attractor_puzzles/
# (MIT License: LICENSE in this directory). Modified.
"""Training the Attractor: TRMHead (ics/trm/train.py) with two differences:
  - the segment is the Attractor's (ics_baselines/attractor/model.py), and only its last cycle carries gradient;
  - the loss adds jacobian_reg_lambda * the regulariser of the last cycle's solve, logged as jacobian_reg.
The solver couples the rows of a micro-batch, so the recipes fix them (train.micro_batch, ics/train.py). Optimizers:
SparseSignSGD on the puzzle embedding, then AdamW on every weight. Evaluation: restart 0 of halt_max_steps segments per
board (ics_baselines/attractor/predict.py), scored raw.
"""
from __future__ import annotations

import numpy as np
import torch

from ics.tasks.base import Task
from ics.trm.train import TRMHead
from ics_baselines.attractor.model import Attractor, AttractorConfig
from ics_baselines.attractor.predict import run_attractor


class AttractorHead(TRMHead):
    config_class = AttractorConfig
    model_class = Attractor

    def segment_terms(self, z_H: torch.Tensor, z_L: torch.Tensor, data: dict) -> tuple:
        """The segment, and in training its regulariser as the term jacobian_reg of weight jacobian_reg_lambda."""
        *out, reg = self.model.regularised_segment(z_H, z_L, data["inputs"], data["puzzle_identifiers"])
        return (*out, {} if reg is None else {"jacobian_reg": (self.model.config.jacobian_reg_lambda, reg)})

    def optimizers(self, lr: float, weight_decay: float, betas, eps: float, puzzle_emb_lr: float,
                   puzzle_emb_weight_decay: float) -> list[tuple[torch.optim.Optimizer, float]]:
        return [*self.sign_sgd(puzzle_emb_lr, puzzle_emb_weight_decay),
                (torch.optim.AdamW(self.model.parameters(), lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                   fused=self.model.device.type == "cuda"), lr)]

    @staticmethod
    def decode(model: Attractor, task: Task, batch: int) -> np.ndarray:
        """Every board's decode: restart 0 of halt_max_steps segments, in blocks of `batch` boards."""
        return run_attractor(model, task, R=1, segments=model.config.halt_max_steps, batch=batch)["attractor/model/raw"]
