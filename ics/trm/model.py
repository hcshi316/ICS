# Adapted from github.com/SamsungSAILMontreal/TinyRecursiveModels@c0110373 (models/recursive_reasoning/trm.py; MIT
# License: see ics/trm/layers.py). Modified.
"""Tiny Recursive Model (TRM), its adaptive-computation wrapper replaced by two explicit calls:
  initial_state(B)                        the cold-start latent: fresh copies of H_init, L_init broadcast over the batch
  segment(z_H, z_L, inputs, puzzle_ids)   one recursion segment: H_cycles x (L_cycles z_L updates, then one z_H
                                          update); only the last H cycle carries gradient. Returns the new (detached)
                                          latent, the token logits of the grid and q_halt, the halting logit.
A release checkpoint (ics/checkpoint.py) holds every TRMConfig field under "model" in config.json and the weights under
the names of TRM.state_dict() ("inner.<name>"); load_trm reads it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ics.checkpoint import load_checkpoint
from ics.trm.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CastedSparseEmbedding,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
    trunc_normal_init_,
)


@dataclass
class TRMConfig:
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 512
    puzzle_emb_len: int = 16
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"          # "rope" | "learned" | "none"
    mlp_t: bool = False                  # MLP token mixer instead of attention (the Sudoku model)
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16             # training: segments per example
    halt_exploration_prob: float = 0.1   # training: ACT exploration
    forward_dtype: str = "bfloat16"
    batch_size: int = 1                  # training: rows of the puzzle-embedding buffer

    @property
    def prefix_len(self) -> int:
        """Positions the puzzle embedding occupies in front of the grid."""
        if self.puzzle_emb_len:
            return self.puzzle_emb_len
        return -(self.puzzle_emb_ndim // -self.hidden_size)


class Block(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        if config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=config.seq_len + config.prefix_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(hidden_size=config.hidden_size, head_dim=config.hidden_size // config.num_heads,
                                       num_heads=config.num_heads, num_key_value_heads=config.num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                                     variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        return rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)


class ReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMInner(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std,
                                            cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = config.prefix_len
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0,
                                                    cast_to=self.forward_dtype)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_heads,
                                              max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                                              base=config.rope_theta)
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(config.seq_len + self.puzzle_emb_len, config.hidden_size,
                                             init_std=embed_init_std, cast_to=self.forward_dtype)
        self.L_level = ReasoningModule([Block(config) for _ in range(config.L_layers)])
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype),
                                                          std=1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype),
                                                          std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                                  dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def forward(self, z_H, z_L, inputs, puzzle_identifiers):
        seq_info = {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}
        x = self._input_embeddings(inputs, puzzle_identifiers)
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return z_H.detach(), z_L.detach(), logits, q[..., 0]


class TRM(nn.Module):
    inner_class = TRMInner          # a variant swaps the recursion (ics_baselines/attractor/model.py)

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.inner = self.inner_class(config)

    @property
    def device(self) -> torch.device:
        return self.inner.H_init.device

    def initial_state(self, batch_size: int):
        shape = (batch_size, self.config.seq_len + self.inner.puzzle_emb_len, self.config.hidden_size)
        return self.inner.H_init.expand(shape).clone(), self.inner.L_init.expand(shape).clone()

    def segment(self, z_H, z_L, inputs, puzzle_identifiers):
        return self.inner(z_H, z_L, inputs, puzzle_identifiers)


def load_trm(ckpt_dir, device="cpu", dtype: str | None = None) -> TRM:
    """A release TRM checkpoint (ics/checkpoint.py) on `device`, optionally in another forward dtype, in eval mode."""
    return load_checkpoint(TRM, TRMConfig, ckpt_dir, device, dtype)
