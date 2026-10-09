# Adapted from github.com/SamsungSAILMontreal/TinyRecursiveModels@c0110373 (models/sparse_embedding.py; MIT License: see
# ics/trm/layers.py). Modified.
"""Optimizers of the TRM recipe.

AdamATan2      Adam with the epsilon division replaced by atan2, and decoupled weight decay: the update rule of
               imoneoi/adam-atan2 0.0.3 in PyTorch. Per parameter p with gradient g, at its step t:
                   m <- lerp(m, g, 1 - beta1);  v <- lerp(v, g^2, 1 - beta2)
                   p <- p * (1 - lr * weight_decay)
                   p <- p - lr / (1 - beta1^t) * atan2(m, sqrt(v) / sqrt(1 - beta2^t))
SparseSignSGD  the puzzle embedding's optimizer: only the rows looked up in the step change, by
               w <- w * (1 - lr * weight_decay) - lr * sign(g), g the row's gradient summed over the global batch.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


class AdamATan2(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "betas": tuple(betas), "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.lerp_(p.grad, 1 - beta1)
                v.lerp_(p.grad.square(), 1 - beta2)
                denom = v.sqrt().div_(math.sqrt(1 - beta2 ** t))
                p.mul_(1 - lr * wd)
                p.add_(torch.atan2(m, denom), alpha=-(lr / (1 - beta1 ** t)))


class SparseSignSGD(torch.optim.Optimizer):
    """`embedding` is a CastedSparseEmbedding (ics/trm/layers.py) in training mode. Every forward overwrites its local
    rows and ids, so a hook keeps each backward's row gradients with their ids (`pending`); step() updates from all
    of them, the micro-batches of a gradient-accumulation step in order."""

    def __init__(self, embedding, lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__([embedding.local_weights], {"lr": lr, "weight_decay": weight_decay})
        self.embedding, self.pending = embedding, []
        embedding.local_weights.register_post_accumulate_grad_hook(self._keep)

    def _keep(self, local_weights: torch.Tensor) -> None:
        self.pending.append((local_weights.grad, self.embedding.local_ids.clone()))
        local_weights.grad = None

    @torch.no_grad()
    def step(self):
        if not self.pending:
            return
        grad, ids = (torch.cat(parts) for parts in zip(*self.pending))
        self.pending = []
        if dist.is_initialized() and dist.get_world_size() > 1:
            grad, ids = (torch.cat(_all_gather(t)) for t in (grad, ids))
        rows, inverse = ids.unique(return_inverse=True)
        summed = torch.zeros((len(rows), grad.shape[1]), dtype=grad.dtype, device=grad.device)
        summed.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, grad.shape[1]), grad)
        lr, wd = self.param_groups[0]["lr"], self.param_groups[0]["weight_decay"]
        weights = self.embedding.weights
        w = weights[rows]
        w.mul_(1.0 - lr * wd).add_(torch.sign(summed), alpha=-lr)
        weights[rows] = w


def _all_gather(t: torch.Tensor) -> list[torch.Tensor]:
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return out
