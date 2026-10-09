import pytest
import torch

from ics.checkpoint import save_checkpoint
from ics.trm.layers import trunc_normal_init_
from ics_baselines.eqr.model import EqR, EqRConfig, load_eqr


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "H_cycles": 2, "L_cycles": 2, "L_layers": 1, "hidden_size": 32,
            "num_heads": 2, "expansion": 2.0, "pos_encodings": "rope", "forward_dtype": "float32"}
    base.update(kw)
    return EqRConfig(**base)


LAYER = ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight")


def test_state_dict_names_are_eqrs():
    names = {"embed_tokens.embedding_weight", "lm_head.weight", "q_head.weight", "q_head.bias",
             *(f"L_level.layers.0.{p}" for p in LAYER)}
    assert set(EqR(tiny()).state_dict()) == {f"inner.{n}" for n in names}
    mixer = EqR(tiny(mlp_t=True))                      # RoPE with the mixer, as the PPB checkpoints have it
    assert "inner.L_level.layers.0.mlp_t.gate_up_proj.weight" in mixer.state_dict()
    assert not any("self_attn" in k for k in mixer.state_dict())
    assert mixer.inner.L_level.layers[0].mlp_t.gate_up_proj.weight.shape[1] == 9     # grid positions only


def test_refuses_other_positional_encodings():
    with pytest.raises(ValueError, match="'rope2d'"):
        EqR(tiny(pos_encodings="rope2d"))


def test_initial_state_draws_z_H_then_z_L_truncated_normal_in_the_model_dtype():
    m = EqR(tiny(forward_dtype="bfloat16", L_init_std=0.5))
    z_H, z_L = m.initial_state(3, torch.Generator().manual_seed(4))
    replay = torch.Generator().manual_seed(4)
    H = trunc_normal_init_(torch.empty(3, 9, 32, dtype=torch.bfloat16), std=1.0, generator=replay)
    L = trunc_normal_init_(torch.empty(3, 9, 32, dtype=torch.bfloat16), std=0.5, generator=replay)
    assert torch.equal(z_H, H) and torch.equal(z_L, L) and z_H.dtype == torch.bfloat16


def test_step_draws_one_noise_tensor_per_update_in_the_model_dtype():
    # hidden_size 24: 3 x 9 x 24 = 648 values per noise tensor. A CPU bfloat16 draw differs from a float32 draw rounded
    # to bfloat16, in its values and the generator's state, only at 16 or more values that are not a multiple of 16
    torch.manual_seed(0)
    m = EqR(tiny(forward_dtype="bfloat16", L_layers=2, lambda_=0.9, hidden_size=24)).eval()
    x = torch.randint(0, 5, (3, 9), dtype=torch.int32)
    gen = torch.Generator().manual_seed(1)
    z_H, z_L = m.initial_state(3, gen)
    replay = torch.Generator().set_state(gen.get_state())
    new_H, new_L, logits, q = m.step(z_H, z_L, x, gen)
    inner, cos_sin = m.inner, m.inner.rotary_emb()

    def update(h, injection):                              # both layers, then one bfloat16 noise tensor per update
        out = h + injection
        for layer in inner.L_level.layers:
            out = layer(cos_sin=cos_sin, hidden_states=out)
        noise = torch.randn((3, 9, 24), generator=replay, dtype=torch.bfloat16) * 0.01
        return (1 - 0.9) * h + 0.9 * out + noise

    x_emb = inner.embed_scale * inner.embed_tokens(x)
    for _ in range(2):                                     # H_cycles x (L_cycles z_L updates, then one z_H update)
        for _ in range(2):
            z_L = update(z_L, z_H + x_emb)
        z_H = update(z_H, z_L)
    assert torch.equal(new_H, z_H) and torch.equal(new_L, z_L)
    assert torch.equal(gen.get_state(), replay.get_state())
    assert logits.shape == (3, 9, 5) and q.shape == (3,) and q.dtype == torch.float32


def test_an_update_is_damped_and_noisy():
    torch.manual_seed(0)
    m = EqR(tiny(noise_scale=0.3)).eval()
    h, inj = torch.randn(2, 9, 32), torch.randn(2, 9, 32)
    cos_sin = m.inner.rotary_emb()
    out = m.inner.L_level(h, inj, cos_sin, torch.Generator().manual_seed(2))
    blocks = h + inj
    for layer in m.inner.L_level.layers:
        blocks = layer(cos_sin=cos_sin, hidden_states=blocks)
    noise = torch.randn(h.shape, generator=torch.Generator().manual_seed(2)) * 0.3
    torch.testing.assert_close(out, (1 - 0.95) * h + 0.95 * blocks + noise, rtol=0, atol=0)


def test_load_eqr_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = tiny()
    m = EqR(cfg)
    save_checkpoint(m.state_dict(), cfg, tmp_path / "ck")
    back = load_eqr(tmp_path / "ck")
    assert not back.training and back.config == cfg
    for k, val in m.state_dict().items():
        torch.testing.assert_close(back.state_dict()[k], val, rtol=0, atol=0)


def test_only_the_last_cycle_carries_gradient():
    torch.manual_seed(0)
    m = EqR(tiny(H_cycles=2))
    x = torch.randint(0, 5, (3, 9), dtype=torch.int32)
    z = m.initial_state(3, torch.Generator().manual_seed(1))
    _, _, logits, q = m.step(*z, x, torch.Generator().manual_seed(2))
    (logits.sum() + q.sum()).backward()
    got = {name: p.grad.clone() for name, p in m.named_parameters()}
    m.zero_grad()
    inner, cos_sin, gen = m.inner, m.inner.rotary_emb(), torch.Generator().manual_seed(2)
    x_emb = inner.embed_scale * inner.embed_tokens(x)
    with torch.no_grad():                                        # upstream's deep_recursion: the first H - 1 cycles
        z_H, z_L = inner.cycle(*z, x_emb, cos_sin, gen)
    z_H, z_L = inner.cycle(z_H, z_L, x_emb, cos_sin, gen)
    (inner.lm_head(z_H).sum() + inner.q_head(z_H[:, 0]).to(torch.float32)[:, 0].sum()).backward()
    for name, p in m.named_parameters():
        assert torch.equal(p.grad, got[name]), name
