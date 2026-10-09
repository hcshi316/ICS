# Adapted from github.com/SamsungSAILMontreal/TinyRecursiveModels@c0110373 (models/recursive_reasoning/trm.py and
# models/losses.py; MIT License: see ics/trm/layers.py). Modified.
"""Training a TRM: the adaptive-computation (ACT) wrapper and the loss of the official TRM code.

TRMHead runs one segment per training step on the rows it carries:
  - a row that halted at the previous step restarts from (H_init, L_init) on the example at its position in the new
    batch; every other row continues its example from its latent;
  - after the segment a row halts at halt_max_steps segments or when q_halt > 0, unless exploration (probability
    halt_exploration_prob) holds it back to a drawn minimum of 2..halt_max_steps segments;
  - loss = the sum over rows of the mean stablemax cross-entropy (float64) over the row's labelled cells
           + 0.5 * the sum over rows of BCE-with-logits(q_halt, whether the row's decode is exact)
           + weight * value for each further term of the segment (segment_terms; TRM: none).
Evaluation decodes the last of halt_max_steps segments from a cold start (ics/trm/roll.py) and scores the accuracy.
A model trained the same way overrides what differs (ics_baselines/eqr/train.py, ics_baselines/attractor/train.py):
model_class, config_class, restart, segment, segment_terms, optimizers and decode.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ics.data import IGNORE, Pool
from ics.optim import AdamATan2, SparseSignSGD
from ics.tasks import make_task
from ics.tasks.base import Task
from ics.trm.model import TRM, TRMConfig
from ics.trm.roll import roll

DATA = ("inputs", "labels", "puzzle_identifiers")


def stablemax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """-log stablemax(logits)[label] per cell, in float64; 0 where mask is False."""
    x = logits.to(torch.float64)
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    logprobs = torch.log(s / torch.sum(s, dim=-1, keepdim=True))
    index = torch.where(mask, labels, 0).to(torch.long).unsqueeze(-1)
    return -torch.where(mask, torch.gather(logprobs, index=index, dim=-1).squeeze(-1), 0)


class TRMHead(nn.Module):
    config_class = TRMConfig
    model_class = TRM
    steps = 1                           # training steps (segments) per update
    metric_name = "accuracy"
    metric_held_out = False             # scored on the test split: the run keeps its last checkpoint alone (train.py)

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.model = self.model_class(config)

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n, c = batch["inputs"].shape[0], self.model.config
        z = torch.zeros(n, c.seq_len + c.prefix_len, c.hidden_size, dtype=getattr(torch, c.forward_dtype),
                        device=self.model.device)
        device = batch["inputs"].device
        return {"z_H": z, "z_L": z.clone(), "steps": torch.zeros(n, dtype=torch.int32, device=device),
                "halted": torch.ones(n, dtype=torch.bool, device=device),
                **{k: torch.zeros_like(batch[k]) for k in DATA}}

    def restart(self, carry: dict, halted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows' latents at this step: (H_init, L_init) for a row that restarts (halted), its own for the others."""
        inner = self.model.inner
        return (torch.where(halted.view(-1, 1, 1), inner.H_init, carry["z_H"]),
                torch.where(halted.view(-1, 1, 1), inner.L_init, carry["z_L"]))

    def segment(self, z_H: torch.Tensor, z_L: torch.Tensor, data: dict) -> tuple:
        """One segment on the rows' current data: (z_H, z_L, logits, q_halt)."""
        return self.model.segment(z_H, z_L, data["inputs"], data["puzzle_identifiers"])

    def segment_terms(self, z_H: torch.Tensor, z_L: torch.Tensor, data: dict) -> tuple:
        """segment's outputs, then the segment's loss terms beyond TRM's as name -> (weight, value): the loss adds
        weight * value, and the statistics log value (TRM, EqR: none)."""
        return (*self.segment(z_H, z_L, data), {})

    def forward(self, carry: dict, batch: dict) -> tuple[dict, torch.Tensor, dict[str, tuple]]:
        config, halted = self.model.config, carry["halted"]
        z_H, z_L = self.restart(carry, halted)
        steps = torch.where(halted, 0, carry["steps"])
        data = {k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], carry[k]) for k in DATA}
        z_H, z_L, logits, q_halt, terms = self.segment_terms(z_H, z_L, data)
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= config.halt_max_steps
            if self.training and config.halt_max_steps > 1:
                halted = halted | (q_halt > 0)
                minimum = (torch.rand_like(q_halt) < config.halt_exploration_prob) * torch.randint_like(
                    steps, low=2, high=config.halt_max_steps + 1)
                halted = halted & (steps >= minimum)
            labels = data["labels"]
            mask = labels != IGNORE
            counts = mask.sum(-1)
            exact = (mask & (logits.argmax(-1) == labels)).sum(-1) == counts
            ended = halted & (counts > 0)
        lm_loss = (stablemax_cross_entropy(logits, labels, mask) / counts.clamp_min(1).unsqueeze(-1)).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, exact.to(q_halt.dtype), reduction="sum")
        n, ends = len(labels), ended.sum()
        stats = {"lm_loss": (lm_loss.detach(), n), "q_halt_loss": (q_halt_loss.detach(), n),
                 "exact": ((ended & exact).sum(), ends), "segments": (torch.where(ended, steps, 0).sum(), ends)}
        loss = lm_loss + 0.5 * q_halt_loss
        for name, (weight, value) in terms.items():
            loss = loss + weight * value
            stats[name] = (value.detach(), 1)
        return {"z_H": z_H, "z_L": z_L, "steps": steps, "halted": halted, **data}, loss, stats

    def optimizers(self, lr: float, weight_decay: float, betas, puzzle_emb_lr: float,
                   puzzle_emb_weight_decay: float) -> list[tuple[torch.optim.Optimizer, float]]:
        return [*self.sign_sgd(puzzle_emb_lr, puzzle_emb_weight_decay),
                (AdamATan2(self.model.parameters(), lr, betas, weight_decay), lr)]

    def sign_sgd(self, puzzle_emb_lr: float,
                 puzzle_emb_weight_decay: float) -> list[tuple[torch.optim.Optimizer, float]]:
        """The first of the optimizers: SparseSignSGD on the puzzle embedding at its base lr; none without one."""
        if self.model.config.puzzle_emb_ndim > 0:
            return [(SparseSignSGD(self.model.inner.puzzle_emb, puzzle_emb_lr, puzzle_emb_weight_decay), puzzle_emb_lr)]
        return []

    @classmethod
    def evaluate(cls, model, pool: Pool, task: str, batch: int) -> np.ndarray:
        if "verifier" in pool.meta:                 # the loop's empty-share probe refuses it before training
            raise ValueError("this is verifier data (python -m ics verifier-data): train it with --model verifier")
        t = make_task(task, pool)
        return t.check("raw", np.arange(len(pool)), cls.decode(model, t, batch)).astype(np.float64)

    @staticmethod
    def decode(model: TRM, task: Task, batch: int) -> np.ndarray:
        """Every board's decode: the last of halt_max_steps segments from a cold start."""
        return roll(model, task.X, model.config.halt_max_steps, batch, topk=1).dec

    @staticmethod
    def metric(pool: Pool, values: np.ndarray) -> float:
        return float(values.mean())
