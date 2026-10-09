"""Training (VerifierHead; python -m ics train --model verifier --init SOLVER/last, ics/configs/train/verifier.yaml).
The verifier starts from the solver: an update takes T = halt_max_steps segments on its rows from a cold start, without
halting, each segment's loss the BCE-with-logits of q_halt against the target, divided by T and backpropagated on its
own. AdamW trains the weights; the puzzle embedding stays the solver's. The run's metric is the AUC of q_halt after T
segments on the val candidates, held out from its train split, so the run keeps best/ beside last/. Data that python -m
ics verifier-data did not write, or wrote for another task, is refused.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ics.data import Pool
from ics.trm.model import TRM, TRMConfig
from ics.trm.roll import roll


def auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """The area under the ROC curve: the chance that a valid candidate (target 1) scores above an invalid one, a tie
    counting one half (the Mann-Whitney statistic); nan without both."""
    scores, positive = np.asarray(scores, np.float64), np.asarray(targets) > 0
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]          # 1-based; tied scores share their mean rank
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class VerifierHead(nn.Module):
    """The verifier's training head (ics/train.py; module docstring)."""
    config_class = TRMConfig
    metric_name = "auc"
    metric_held_out = True              # scored on the val split, boards held out from its train split: best/ too
    needs_init = True

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.model = TRM(config)
        if config.puzzle_emb_ndim > 0:
            self.model.inner.puzzle_emb.local_weights.requires_grad_(False)      # the solver's puzzle embedding stays

    @property
    def steps(self) -> int:
        return self.model.config.halt_max_steps

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_H, z_L = self.model.initial_state(batch["inputs"].shape[0])
        return {"z_H": z_H, "z_L": z_L, "segment": torch.zeros((), dtype=torch.int32, device=z_H.device)}

    def forward(self, carry: dict, batch: dict) -> tuple[dict, torch.Tensor, dict[str, tuple]]:
        inner, cold = self.model.inner, carry["segment"] == 0          # segment 0 of an update starts cold
        z_H = torch.where(cold, inner.H_init, carry["z_H"])
        z_L = torch.where(cold, inner.L_init, carry["z_L"])
        z_H, z_L, _, q = self.model.segment(z_H, z_L, batch["inputs"], batch["puzzle_identifiers"])
        bce = F.binary_cross_entropy_with_logits(q, batch["labels"].to(q.dtype), reduction="sum")
        carry = {"z_H": z_H, "z_L": z_L, "segment": (carry["segment"] + 1) % self.steps}
        return carry, bce / self.steps, {"bce": (bce.detach(), len(q))}

    def optimizers(self, lr: float, weight_decay: float, betas) -> list[tuple[torch.optim.Optimizer, float]]:
        return [(torch.optim.AdamW(self.model.parameters(), lr, betas=tuple(betas), weight_decay=weight_decay), lr)]

    @staticmethod
    def evaluate(model: TRM, pool: Pool, task: str, batch: int) -> np.ndarray:
        if "verifier" not in pool.meta:             # the loop's empty-share probe refuses other data before training
            raise ValueError("the verifier trains on the data of python -m ics verifier-data, whose dataset.json holds "
                             "a \"verifier\" summary; this data has none (a solver's own dataset, say)")
        if (built := pool.meta["verifier"]["task"]) != task:
            raise ValueError(f"this is the verifier data of {built} (python -m ics verifier-data --task {built}): "
                             f"train it with --task {built}")
        return roll(model, pool.inputs, model.config.halt_max_steps, batch, topk=1).q

    @staticmethod
    def metric(pool: Pool, values: np.ndarray) -> float:
        valid = int((pool.labels > 0).sum())
        if valid in (0, len(pool)):
            raise ValueError(f"the eval pool holds {valid} valid and {len(pool) - valid} invalid candidates; its AUC "
                             f"needs both: raise train.eval_boards (the val split stores its valid candidates first) "
                             f"or leave it unset (null: the whole split)")
        return auc(values, pool.labels)
