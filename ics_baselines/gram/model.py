"""GRAM: Generative Recursive reAsoning Models (Baek et al., 2026, arXiv:2605.19376), as reproduced by us, on TRM's
layers, block and reasoning module (ics/trm/layers.py, ics/trm/model.py).

  initial_state(B)             the fixed z0: fresh copies of H_init and L_init broadcast over the batch
  step(z_H, z_L, inputs, gen)  one supervision step at inference: H_cycles stochastic transitions under the prior. A
                               transition updates z_L L_cycles times with f_L(z_L, z_H + x), proposes u = f_H(z_H, z_L)
                               and moves to z_H = u + mu(u) + sigma(u) * e, e standard normal from `gen`. Returns the
                               latent, the token logits, q_halt and v (the logit of the LPRM value).
  train_step(z_H, z_L, data)   one supervision step of training: mu and sigma from the posterior head, which also sees
                               the embedded labels, and e from the global generator; only the last transition carries
                               gradient. Returns the detached latent, logits, q_halt, v and kl [B]: the balanced KL of
                               the last transition, or with kl_all_transitions of every transition.
The balanced KL of a transition is kl_balance * KL(sg q || p) + (1 - kl_balance) * KL(q || sg p) of the diagonal
Gaussians q (the posterior) and p (the prior), summed over channels and positions; sg stops the gradient."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ics.checkpoint import load_checkpoint
from ics.data import IGNORE
from ics.trm.layers import CastedEmbedding, CastedLinear, RotaryEmbedding, SwiGLU, trunc_normal_init_
from ics.trm.model import Block, ReasoningModule


@dataclass
class GRAMConfig:
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 1
    puzzle_emb_len: int = 16             # learned register tokens in front of the grid
    H_cycles: int = 3                    # T: transitions per supervision step
    L_cycles: int = 6                    # K: z_L updates per transition
    L_layers: int = 2
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "none"          # "rope" | "none"
    mlp_t: bool = False                  # MLP token mixer instead of attention (the Sudoku model)
    noise_expansion: float = 1.0         # expansion of the noise heads' SwiGLU
    min_std: float = 1e-3                # sigma = softplus(raw sigma) + min_std
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16             # training: N_sup, the supervision steps of every row's trajectory
    kl_coef: float = 0.1                 # training: beta, the KL's weight in the loss
    kl_balance: float = 0.8              # training: the share of the KL's gradient that trains the prior
    kl_all_transitions: bool = True      # training: charge every transition's KL, not only the last one's
    grad_checkpoint: bool = True         # training: recompute the last transition's core calls in the backward
    forward_dtype: str = "bfloat16"

    @property
    def prefix_len(self) -> int:
        """Positions the register tokens occupy in front of the grid (TRM's Block reads it)."""
        return self.puzzle_emb_len


class NoiseHead(nn.Module):
    """A SwiGLU MLP giving (mu, raw sigma) of the guidance noise; its output projection starts at zero."""

    def __init__(self, in_size: int, hidden_size: int, expansion: float):
        super().__init__()
        self.net = SwiGLU(hidden_size=in_size, expansion=expansion, out_size=2 * hidden_size)
        with torch.no_grad():
            self.net.down_proj.weight.zero_()

    def forward(self, x: torch.Tensor):
        return self.net(x).chunk(2, dim=-1)


def gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) per element, in float32."""
    mu_q, sigma_q, mu_p, sigma_p = (t.to(torch.float32) for t in (mu_q, sigma_q, mu_p, sigma_p))
    return torch.log(sigma_p / sigma_q) + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5


class GRAMInner(nn.Module):
    def __init__(self, config: GRAMConfig):
        super().__init__()
        if config.pos_encodings not in ("rope", "none"):
            raise ValueError(f"GRAM supports pos_encodings 'rope' and 'none', got {config.pos_encodings!r}")
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std,
                                            cast_to=self.forward_dtype)
        self.puzzle_emb = CastedEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_len * config.hidden_size,
                                          init_std=0, cast_to=self.forward_dtype)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_heads,
                                              max_position_embeddings=config.seq_len + config.puzzle_emb_len,
                                              base=config.rope_theta)
        self.f_L = ReasoningModule([Block(config) for _ in range(config.L_layers)])
        self.f_H = ReasoningModule([Block(config) for _ in range(config.L_layers)])
        self.prior_head = NoiseHead(config.hidden_size, config.hidden_size, config.noise_expansion)
        self.post_head = NoiseHead(2 * config.hidden_size, config.hidden_size, config.noise_expansion)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self.v_head = CastedLinear(config.hidden_size, 1, bias=True)
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype),
                                                          std=1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype),
                                                          std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
            self.v_head.weight.zero_()
            self.v_head.bias.fill_(0)

    def embed(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """The register tokens, then the grid's token embeddings; scaled by sqrt(hidden)."""
        tokens = self.embed_tokens(inputs.to(torch.int32))
        registers = self.puzzle_emb(puzzle_identifiers.to(torch.int32))
        registers = registers.view(-1, self.config.puzzle_emb_len, self.config.hidden_size)
        return self.embed_scale * torch.cat((registers, tokens), dim=-2)

    def embed_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """e_y, the posterior's view of the target: zero register rows, then the labels' token embeddings (an
        unlabelled cell as token 0); scaled by sqrt(hidden)."""
        tokens = self.embed_tokens(torch.where(labels == IGNORE, torch.zeros_like(labels), labels).to(torch.int32))
        registers = torch.zeros(tokens.shape[0], self.config.puzzle_emb_len, self.config.hidden_size,
                                dtype=tokens.dtype, device=tokens.device)
        return self.embed_scale * torch.cat((registers, tokens), dim=-2)

    def sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw.to(torch.float32)) + self.config.min_std

    def balanced_kl(self, q_mu, q_sigma, p_mu, p_sigma) -> torch.Tensor:
        """[B]: kl_balance * KL(sg q || p) + (1 - kl_balance) * KL(q || sg p), summed over channels and positions."""
        lhs = gaussian_kl(q_mu.detach(), q_sigma.detach(), p_mu, p_sigma)
        rhs = gaussian_kl(q_mu, q_sigma, p_mu.detach(), p_sigma.detach())
        return (self.config.kl_balance * lhs + (1.0 - self.config.kl_balance) * rhs).sum(-1).sum(-1)

    def core(self, module, hidden_states, injection, cos_sin):
        """f_L or f_H; with gradient and grad_checkpoint, its activations are recomputed in the backward."""
        if self.config.grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(module, hidden_states, injection, use_reentrant=False, cos_sin=cos_sin)
        return module(hidden_states, injection, cos_sin=cos_sin)

    def transition(self, z_H, z_L, x, cos_sin, gen: torch.Generator | None, e_y=None, kl: bool = False):
        """One transition: (z_H, z_L, its balanced KL [B] if kl, else None, the proposal u). The noise is the
        posterior's when e_y is given, else the prior's."""
        for _ in range(self.config.L_cycles):
            z_L = self.core(self.f_L, z_L, z_H + x, cos_sin)
        u = self.core(self.f_H, z_H, z_L, cos_sin)
        p_mu, p_raw = self.prior_head(u)
        if e_y is None:
            mu, sigma = p_mu, self.sigma(p_raw)
        else:
            mu, q_raw = self.post_head(torch.cat((u, e_y), dim=-1))
            sigma = self.sigma(q_raw)
        noise = torch.randn(sigma.shape, generator=gen, device=sigma.device, dtype=torch.float32)
        z_H = u + (mu.to(torch.float32) + sigma * noise).to(u.dtype)
        return z_H, z_L, self.balanced_kl(mu, sigma, p_mu, self.sigma(p_raw)) if kl else None, u

    def head_kl(self, u: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
        """The balanced KL of a transition run without gradient: its noise heads re-run on its (detached) proposal."""
        p_mu, p_raw = self.prior_head(u)
        q_mu, q_raw = self.post_head(torch.cat((u, e_y), dim=-1))
        return self.balanced_kl(q_mu, self.sigma(q_raw), p_mu, self.sigma(p_raw))

    def forward(self, z_H, z_L, inputs, puzzle_identifiers, gen: torch.Generator | None, labels=None):
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        x = self.embed(inputs, puzzle_identifiers)
        e_y = None if labels is None else self.embed_labels(labels)
        kl_extra = None
        for _ in range(self.config.H_cycles - 1):
            with torch.no_grad():
                z_H, z_L, _, u = self.transition(z_H, z_L, x, cos_sin, gen, e_y)
            if e_y is not None and self.config.kl_all_transitions and torch.is_grad_enabled():
                kl_t = self.head_kl(u, e_y)
                kl_extra = kl_t if kl_extra is None else kl_extra + kl_t
        z_H, z_L, kl, _ = self.transition(z_H, z_L, x, cos_sin, gen, e_y, kl=e_y is not None)
        if kl is not None and kl_extra is not None:
            kl = kl + kl_extra
        logits = self.lm_head(z_H)[:, self.config.puzzle_emb_len:]
        q = self.q_head(z_H[:, 0].detach()).to(torch.float32)
        v = self.v_head(z_H[:, 0].detach()).to(torch.float32)
        return z_H.detach(), z_L.detach(), logits, q[..., 0], v[..., 0], kl


class GRAM(nn.Module):
    def __init__(self, config: GRAMConfig):
        super().__init__()
        self.config = config
        self.inner = GRAMInner(config)

    @property
    def device(self) -> torch.device:
        return self.inner.H_init.device

    def initial_state(self, batch_size: int):
        shape = (batch_size, self.config.seq_len + self.config.puzzle_emb_len, self.config.hidden_size)
        return self.inner.H_init.expand(shape).clone(), self.inner.L_init.expand(shape).clone()

    def step(self, z_H, z_L, inputs, gen: torch.Generator):
        ids = torch.zeros(inputs.shape[0], dtype=torch.int32, device=inputs.device)
        return self.inner(z_H, z_L, inputs, ids, gen)[:5]

    def train_step(self, z_H, z_L, data: dict):
        return self.inner(z_H, z_L, data["inputs"], data["puzzle_identifiers"], None, data["labels"])


def load_gram(ckpt_dir, device="cpu", dtype: str | None = None) -> GRAM:
    """A release GRAM checkpoint (ics/checkpoint.py) on `device`, optionally in another forward dtype, in eval mode."""
    return load_checkpoint(GRAM, GRAMConfig, ckpt_dir, device, dtype)
