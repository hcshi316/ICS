# Adapted from github.com/jacobfa/Attractor@fcf045f9fb9d458f9e1574bd4036cd2fb63b3697, experiments/attractor_puzzles/
# (MIT License: LICENSE in this directory). Modified.
"""Attractor (Fein-Ashley and Rashidinejad, 2026): TRM whose L-cycle loop is a fixed-point solve. The embeddings,
blocks, heads and buffers are TRM's (ics/trm/model.py); the Anderson solver mixes its last deq_anderson_m iterates.

  segment(z_H, z_L, inputs, puzzle_ids)   H_cycles x (z_L <- the fixed point of z -> L_level(z, z_H + x), found by
      Anderson acceleration from the current z_L and followed by max(1, bptt_through) more applications of the map;
      z_H <- L_level(z_H, z_L)); only the last cycle carries gradient. Returns the latent, the logits and q_halt.
  regularised_segment(...)                the same, then, in training with jacobian_reg_lambda > 0, the Jacobian
      regulariser mean((J^T v)^2) of the last cycle's solve (J the Jacobian of the map at the solve's result, v a
      standard normal probe / sqrt(hidden_size)), else None.
  initial_state(B, dH, dL)                z0 = (H_init + dH, L_init + dL) broadcast over the batch (None: no
      perturbation).
The solver stops once every row's relative residual is below deq_tol (between deq_min_iter and deq_max_iter map
evaluations), so a row's result can depend on the other rows of its batch."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ics.checkpoint import load_checkpoint
from ics.trm.model import TRM, TRMConfig, TRMInner


@dataclass
class AttractorConfig(TRMConfig):
    # The inherited L_cycles is not read: the fixed-point solve replaces the L-cycle loop, whatever its value.
    deq_max_iter: int = 8                # map evaluations per solve, at most
    deq_min_iter: int = 4                # map evaluations per solve, at least
    deq_tol: float = 1e-3                # stop once every row's relative residual is below this
    deq_anderson_m: int = 5              # history length
    deq_anderson_beta: float = 1.0       # mixing
    bptt_through: int = 2                # map applications after the solve (the gradient path in training; always run)
    jacobian_reg_lambda: float = 1e-3    # training: the regulariser's weight in the loss (0: no regulariser)


def anderson(f, y0: torch.Tensor, *, max_iter: int, tol: float, min_iter: int, m: int, beta: float) -> torch.Tensor:
    """Anderson-accelerated fixed-point iteration of f from y0 in float32, reading the history from the last min(k, m)
    slots. The first evaluation of f runs in y0's dtype, the others in float32. Returns the last iterate in y0's
    dtype."""
    B = y0.size(0)
    n = y0.numel() // B
    dtype, device = torch.float32, y0.device
    Y = torch.zeros(B, m, n, dtype=dtype, device=device)
    F_ = torch.zeros(B, m, n, dtype=dtype, device=device)
    H = torch.zeros(B, m + 1, m + 1, dtype=dtype, device=device)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    rhs = torch.zeros(B, m + 1, 1, dtype=dtype, device=device)
    rhs[:, 0] = 1.0

    Y[:, -2] = y0.detach().reshape(B, n).to(dtype)
    F_[:, -2] = f(y0).detach().reshape(B, n).to(dtype)
    Y[:, -1] = F_[:, -2]
    F_[:, -1] = f(F_[:, -2].view_as(y0)).detach().reshape(B, n).to(dtype)

    for k in range(2, max_iter):
        nn_ = min(k, m)
        G = F_[:, -nn_:] - Y[:, -nn_:]
        H[:, 1:nn_ + 1, 1:nn_ + 1] = (torch.bmm(G, G.transpose(1, 2))
                                      + 1e-4 * torch.eye(nn_, dtype=dtype, device=device).unsqueeze(0))
        try:
            alpha = torch.linalg.solve(H[:, :nn_ + 1, :nn_ + 1], rhs[:, :nn_ + 1])[:, 1:nn_ + 1, 0]
        except RuntimeError:
            break
        y_new = (beta * (alpha[..., None] * F_[:, -nn_:]).sum(dim=1)
                 + (1 - beta) * (alpha[..., None] * Y[:, -nn_:]).sum(dim=1))
        F_new = f(y_new.view_as(y0)).detach().reshape(B, n).to(dtype)
        Y = torch.roll(Y, shifts=-1, dims=1)
        F_ = torch.roll(F_, shifts=-1, dims=1)
        Y[:, -1] = y_new
        F_[:, -1] = F_new
        rel = float(((F_new - y_new).norm(dim=1) / F_new.norm(dim=1).clamp_min(1e-9)).max())
        if k + 1 >= min_iter and rel < tol:
            break
    return Y[:, -1].view_as(y0).to(y0.dtype)


class AttractorInner(TRMInner):
    def regulariser(self, z, ctx, seq_info):
        """mean((J^T v)^2) at z, J the Jacobian of z -> L_level(z, ctx) and v a standard normal probe drawn like z,
        divided by sqrt(hidden_size); J^T v by a double backward through math attention, kept in the graph."""
        v = (torch.randn_like(z) / math.sqrt(z.shape[-1])).detach()
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            z_in = z.detach().requires_grad_(True)
            s = (v * self.L_level(z_in, ctx, **seq_info)).sum()
            (jtv,) = torch.autograd.grad(s, z_in, create_graph=True)
        return jtv.pow(2).mean()

    def solve(self, z_L, ctx, seq_info):
        """The fixed point of z -> L_level(z, ctx) from z_L (no gradient), then max(1, bptt_through) more applications
        of the map; and the regulariser at the result in training mode with jacobian_reg_lambda > 0, else None."""
        c = self.config

        def fmap(z):
            return self.L_level(z, ctx, **seq_info)

        with torch.no_grad():
            z = anderson(fmap, z_L, max_iter=int(c.deq_max_iter), tol=float(c.deq_tol), min_iter=int(c.deq_min_iter),
                         m=int(c.deq_anderson_m), beta=float(c.deq_anderson_beta))
        z = z.detach()
        for _ in range(max(1, int(c.bptt_through))):
            z = fmap(z)
        return z, (self.regulariser(z, ctx, seq_info) if self.training and c.jacobian_reg_lambda > 0 else None)

    def forward(self, z_H, z_L, inputs, puzzle_identifiers):
        seq_info = {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}
        x = self._input_embeddings(inputs, puzzle_identifiers)
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                z_L, _ = self.solve(z_L, z_H + x, seq_info)          # a warm-up's regulariser is discarded
                z_H = self.L_level(z_H, z_L, **seq_info)
        z_L, reg = self.solve(z_L, z_H + x, seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return z_H.detach(), z_L.detach(), logits, q[..., 0], reg


class Attractor(TRM):
    inner_class = AttractorInner

    def initial_state(self, batch_size: int, dH: torch.Tensor | None = None, dL: torch.Tensor | None = None):
        H, L = self.inner.H_init, self.inner.L_init
        if dH is not None:
            H = H + dH.to(H.dtype)
        if dL is not None:
            L = L + dL.to(L.dtype)
        shape = (batch_size, self.config.seq_len + self.inner.puzzle_emb_len, self.config.hidden_size)
        return H.expand(shape).clone(), L.expand(shape).clone()

    def segment(self, z_H, z_L, inputs, puzzle_identifiers):
        return self.inner(z_H, z_L, inputs, puzzle_identifiers)[:4]

    def regularised_segment(self, z_H, z_L, inputs, puzzle_identifiers):
        return self.inner(z_H, z_L, inputs, puzzle_identifiers)


def load_attractor(ckpt_dir, device="cpu", dtype: str | None = None) -> Attractor:
    """A release Attractor checkpoint (ics/checkpoint.py) on `device`, optionally in another forward dtype, in eval
    mode."""
    return load_checkpoint(Attractor, AttractorConfig, ckpt_dir, device, dtype)
