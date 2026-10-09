import torch

from ics.trm.model import TRM, TRMConfig


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "H_cycles": 2, "L_cycles": 2,
            "L_layers": 1, "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32"}
    base.update(kw)
    return TRMConfig(**base)


def test_segment_shapes_and_state():
    torch.manual_seed(0)
    m = TRM(tiny()).eval()
    x = torch.randint(0, 5, (3, 9), dtype=torch.int32)
    ids = torch.zeros(3, dtype=torch.int32)
    z_H, z_L = m.initial_state(3)
    assert z_H.shape == (3, 11, 32)
    z_H, z_L, logits, q = m.segment(z_H, z_L, x, ids)
    assert logits.shape == (3, 9, 5) and q.shape == (3,) and q.dtype == torch.float32
    assert not z_H.requires_grad


def test_segment_is_deterministic():
    torch.manual_seed(0)
    m = TRM(tiny()).eval()
    x = torch.randint(0, 5, (2, 9), dtype=torch.int32)
    ids = torch.zeros(2, dtype=torch.int32)
    a = m.segment(*m.initial_state(2), x, ids)
    b = m.segment(*m.initial_state(2), x, ids)
    torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)


def test_initial_state_is_a_fresh_copy():
    m = TRM(tiny()).eval()
    H, L = m.inner.H_init.clone(), m.inner.L_init.clone()
    z_H, z_L = m.initial_state(2)
    z_H[torch.tensor([True, False])] = 9.0          # masked write: must not reach the model's buffers
    z_L[0, 0] += 1.0
    torch.testing.assert_close(m.inner.H_init, H, rtol=0, atol=0)
    torch.testing.assert_close(m.inner.L_init, L, rtol=0, atol=0)


def test_state_dict_names_match_the_released_checkpoints():
    att = set(TRM(tiny(pos_encodings="rope")).state_dict())
    assert "inner.L_level.layers.0.self_attn.qkv_proj.weight" in att
    assert {"inner.H_init", "inner.L_init", "inner.puzzle_emb.weights", "inner.q_head.bias"} <= att
    assert not any("rotary" in k or "local_" in k for k in att)          # non-persistent buffers
    mlp = set(TRM(tiny(mlp_t=True, pos_encodings="none")).state_dict())
    assert "inner.L_level.layers.0.mlp_t.gate_up_proj.weight" in mlp
    assert len(att) == 11 and len(mlp) == 11
