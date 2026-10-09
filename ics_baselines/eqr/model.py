# Adapted from github.com/locuslab/EqR@aba94e9cde0f273ce644db5261cd6915ba6561f0 (Apache License 2.0: LICENSE in this
# directory). Modified.
"""EqR: a recursive reasoner whose shared update is damped and noisy, at inference too. Its block and layers are TRM's
(ics/trm/model.py, ics/trm/layers.py), its attention PyTorch's scaled_dot_product_attention. Unlike upstream's
evaluator, which redraws (and discards) both latents at every later step, a restart draws its latent once.

  initial_state(B, gen)          a fresh latent per row and position: z_H, then z_L, truncated normal (std H_init_std,
                                 L_init_std) drawn in the model dtype from `gen` (None: the global generator)
  step(z_H, z_L, inputs, gen)    one ACT step: H_cycles x (L_cycles z_L updates, then one z_H update). An update is
                                 h <- (1 - lambda_) h + lambda_ blocks(h + injection) + noise_scale e, with e standard
                                 normal drawn in the model dtype from `gen`; z_L's injection is z_H + x, z_H's is z_L.
                                 Only the last cycle carries gradient. Returns the detached latent, the logits, q_halt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ics.checkpoint import load_checkpoint
from ics.trm.layers import CastedEmbedding, CastedLinear, RotaryEmbedding, trunc_normal_init_
from ics.trm.model import Block


@dataclass
class EqRConfig:
    seq_len: int
    vocab_size: int
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "none"          # "rope" | "none"
    mlp_t: bool = False                  # MLP token mixer instead of attention
    lambda_: float = 0.95                # damping of every update
    noise_scale: float = 0.01            # std of the update noise (on at inference too)
    H_init_std: float = 1.0              # std of a restart's initial z_H
    L_init_std: float = 1.0              # std of a restart's initial z_L
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16             # training: steps per example
    halt_exploration_prob: float = 0.1   # training: ACT exploration
    forward_dtype: str = "bfloat16"

    @property
    def prefix_len(self) -> int:
        """EqR has no puzzle embedding: TRM's Block mixes the grid's positions only."""
        return 0


class NoisyReasoningModule(nn.Module):
    """EqR's L_level, shared by the z_L and z_H updates."""

    def __init__(self, config: EqRConfig, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.lambda_, self.noise_scale = config.lambda_, config.noise_scale

    def forward(self, hidden_states, input_injection, cos_sin, gen: torch.Generator | None):
        updated = hidden_states + input_injection
        for layer in self.layers:
            updated = layer(cos_sin=cos_sin, hidden_states=updated)
        noise = torch.randn(hidden_states.shape, generator=gen, dtype=hidden_states.dtype,
                            device=hidden_states.device) * self.noise_scale
        return (1 - self.lambda_) * hidden_states + self.lambda_ * updated + noise


class EqRInner(nn.Module):
    def __init__(self, config: EqRConfig):
        super().__init__()
        if config.pos_encodings not in ("rope", "none"):
            raise ValueError(f"EqR supports pos_encodings 'rope' and 'none', got {config.pos_encodings!r}")
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=1.0 / self.embed_scale,
                                            cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.seq_len,
                                              config.rope_theta)
        self.L_level = NoisyReasoningModule(config, [Block(config) for _ in range(config.L_layers)])
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def cycle(self, z_H, z_L, x, cos_sin, gen: torch.Generator | None):
        """L_cycles z_L updates, then one z_H update (upstream's latent_recursion)."""
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, cos_sin, gen)
        return self.L_level(z_H, z_L, cos_sin, gen), z_L

    def forward(self, z_H, z_L, inputs, gen: torch.Generator | None):
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        x = self.embed_scale * self.embed_tokens(inputs.to(torch.int32))
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                z_H, z_L = self.cycle(z_H, z_L, x, cos_sin, gen)
        z_H, z_L = self.cycle(z_H, z_L, x, cos_sin, gen)
        logits = self.lm_head(z_H)
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return z_H.detach(), z_L.detach(), logits, q[..., 0]


class EqR(nn.Module):
    def __init__(self, config: EqRConfig):
        super().__init__()
        self.config = config
        self.inner = EqRInner(config)

    @property
    def device(self) -> torch.device:
        return self.inner.lm_head.weight.device

    def initial_state(self, batch_size: int, gen: torch.Generator | None):
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        z_H = trunc_normal_init_(torch.empty(shape, dtype=self.inner.forward_dtype, device=self.device),
                                 std=self.config.H_init_std, generator=gen)
        z_L = trunc_normal_init_(torch.empty(shape, dtype=self.inner.forward_dtype, device=self.device),
                                 std=self.config.L_init_std, generator=gen)
        return z_H, z_L

    def step(self, z_H, z_L, inputs, gen: torch.Generator | None):
        return self.inner(z_H, z_L, inputs, gen)


def load_eqr(ckpt_dir, device="cpu", dtype: str | None = None) -> EqR:
    """A release EqR checkpoint (ics/checkpoint.py) on `device`, optionally in another forward dtype, in eval mode."""
    return load_checkpoint(EqR, EqRConfig, ckpt_dir, device, dtype)
