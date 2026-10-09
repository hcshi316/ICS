"""Training GRAM (ics_baselines/gram/model.py) in the loop of ics/train.py.

GRAMHead runs one supervision step per training step on the rows it carries, with TRMHead's carry and restart from z0
(ics/trm/train.py), but without halting: every row runs halt_max_steps steps and all rows restart together. A step
samples its guidance noise from the posterior and returns the KL of its transitions.
  loss = the sum over rows of the mean stablemax cross-entropy (float64) over the row's labelled cells
         + kl_coef * the sum over rows of the KL
         + 0.5 * the sum over rows of BCE-with-logits(q_halt, whether the row's decode is exact).
At a cycle's last step the loss adds the deferred LPRM: 0.5 * the sum, over the cycle's steps and the rows, of
(sigmoid(v) - the row's token accuracy at this last step)^2, v read from the first register token of each step's
latent; its gradient reaches v_head alone. Optimizer: AdamW on every weight. Evaluation: one prior sample of
halt_max_steps steps per board (ics_baselines/gram/predict.py), scored raw."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ics.data import IGNORE
from ics.tasks.base import Task
from ics.trm.train import DATA, TRMHead, stablemax_cross_entropy
from ics_baselines.gram.model import GRAM, GRAMConfig
from ics_baselines.gram.predict import run_gram


class GRAMHead(TRMHead):
    config_class = GRAMConfig
    model_class = GRAM

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """TRMHead's carry, and z0: the first register token of the latent after each of the cycle's steps so far."""
        carry = super().initial_carry(batch)
        z = carry["z_H"]
        carry["z0"] = z.new_zeros(self.model.config.halt_max_steps, z.shape[0], z.shape[-1])
        return carry

    def forward(self, carry: dict, batch: dict) -> tuple[dict, torch.Tensor, dict[str, tuple]]:
        config, halted = self.model.config, carry["halted"]
        z_H, z_L = self.restart(carry, halted)
        steps = torch.where(halted, 0, carry["steps"])
        data = {k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], carry[k]) for k in DATA}
        z_H, z_L, logits, q_halt, _, kl = self.model.train_step(z_H, z_L, data)
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= config.halt_max_steps             # every row at once: no halting in training
            labels = data["labels"]
            mask = labels != IGNORE
            counts = mask.sum(-1)
            correct = mask & (logits.argmax(-1) == labels)
            accuracy = (correct.to(torch.float32) / counts.clamp_min(1).unsqueeze(-1)).sum(-1)
            exact = correct.sum(-1) == counts
            ended = halted & (counts > 0)
            z0 = torch.cat((carry["z0"][1:], z_H[:, 0].unsqueeze(0)))
        lm_loss = (stablemax_cross_entropy(logits, labels, mask) / counts.clamp_min(1).unsqueeze(-1)).sum()
        kl_loss = kl.sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, exact.to(q_halt.dtype), reduction="sum")
        loss = lm_loss + config.kl_coef * kl_loss + 0.5 * q_halt_loss
        n = len(labels)
        stats = {"lm_loss": (lm_loss.detach(), n), "kl": (kl_loss.detach(), n),
                 "q_halt_loss": (q_halt_loss.detach(), n), "exact": ((ended & exact).sum(), ended.sum())}
        if halted.all():                                        # the cycle's last step: the deferred LPRM
            v_head = self.model.inner.v_head
            v_loss = sum(F.mse_loss(torch.sigmoid(v_head(z).to(torch.float32).squeeze(-1)), accuracy, reduction="sum")
                         for z in z0)
            loss = loss + 0.5 * v_loss
            stats["v_loss"] = (v_loss.detach(), n * len(z0))
        return {"z_H": z_H, "z_L": z_L, "steps": steps, "halted": halted, "z0": z0, **data}, loss, stats

    def optimizers(self, lr: float, weight_decay: float, betas) -> list[tuple[torch.optim.Optimizer, float]]:
        return [(torch.optim.AdamW(self.model.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay), lr)]

    @staticmethod
    def decode(model: GRAM, task: Task, batch: int) -> np.ndarray:
        """Every board's decode: one prior sample of halt_max_steps steps from z0 (run_gram's N = 1)."""
        return run_gram(model, task, N=1, D=model.config.halt_max_steps, batch=batch)["gram_lprm/model/raw"]
