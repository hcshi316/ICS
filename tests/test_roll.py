import re

import numpy as np
import pytest
import torch

from fakes import ScriptedTRM
from ics.trm.model import TRM, TRMConfig
from ics.trm.roll import chunk_rows, device_batch, roll, segments


def echo_model(vocab=12):
    """Decodes every row to itself; q = number of the segment (ScriptedTRM counts the segments in z_H)."""
    return ScriptedTRM(vocab, lambda row, t, zl: (row, float(t), 1.0))


def tiny_trm(**kw):
    """A tiny real TRM, float32 on the CPU. Its q head is drawn at random: TRM's init zeroes the weight, which would
    make q_halt a constant."""
    torch.manual_seed(0)
    cfg = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "H_cycles": 2, "L_cycles": 2,
           "L_layers": 1, "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32", **kw}
    model = TRM(TRMConfig(**cfg)).eval()
    torch.nn.init.normal_(model.inner.q_head.weight)
    return model


def test_chunk_rows():
    assert chunk_rows(768, 768) == 768
    assert chunk_rows(1000, 768) == 768
    assert chunk_rows(13, 768) == 16
    assert chunk_rows(8, 768) == 8
    assert chunk_rows(250, 250) == 250
    assert chunk_rows(5, 6) == 6


def test_a_short_chunk_is_padded_with_copies_of_its_first_row():
    rows = np.arange(13 * 3).reshape(13, 3)
    xb, n = device_batch(rows, 768, "cpu")
    assert n == 13 and xb.shape == (16, 3) and xb.dtype == torch.int32
    np.testing.assert_array_equal(xb[:13].numpy(), rows)
    np.testing.assert_array_equal(xb[13:].numpy(), np.repeat(rows[:1], 3, 0))


def test_roll_returns_real_rows_only():
    rows = np.arange(30).reshape(10, 3) % 12
    r = roll(echo_model(), rows, T=4, batch=4)             # chunks of 4, 4, 2 (the last padded to 8 -> capped at 4)
    np.testing.assert_array_equal(r.dec, rows)
    np.testing.assert_array_equal(r.q, np.full(10, 4.0))
    assert r.top_ids.shape == (10, 3, 10) and r.top_vals.shape == (10, 3, 10)
    np.testing.assert_array_equal(r.top_ids[:, :, 0], rows)


def test_roll_empty():
    r = roll(echo_model(), np.zeros((0, 3), np.int64), T=2, batch=4)
    assert r.dec.shape == (0, 3) and r.q.shape == (0,) and r.state is None
    kept = roll(echo_model(), np.zeros((0, 3), np.int64), T=2, batch=4, keep_state=True)
    assert [z.shape for z in kept.state] == [(0, 1), (0, 1)]


def test_the_noise_hook_sees_every_segment_of_every_chunk():
    calls = []

    def hook(t, z_L):
        calls.append(t)
        return z_L + 1.0

    m = ScriptedTRM(12, lambda row, t, zl: (row, zl, 1.0))   # q reports the accumulated noise
    r = roll(m, np.ones((6, 2), np.int64), T=3, batch=4, noise=hook)
    assert calls == [1, 2, 3, 1, 2, 3]
    np.testing.assert_array_equal(r.q, np.full(6, 3.0))


@pytest.mark.parametrize("T, batch", [(0, 4), (2, 0)], ids=["T_0", "batch_0"])
def test_rejects_out_of_range_settings(T, batch):
    with pytest.raises(ValueError, match=f"^roll needs T >= 1 and batch >= 1, got T={T}, batch={batch}$"):
        roll(echo_model(), np.ones((2, 3), np.int64), T=T, batch=batch)


def test_segments_yields_each_step_and_its_state():
    xb = torch.ones(2, 3, dtype=torch.int32)
    steps = list(segments(echo_model(), xb, T=5))
    assert [t for t, _logits, _q, _z in steps] == [1, 2, 3, 4, 5]
    assert [z_H[:, 0].tolist() for *_, (z_H, _z_L) in steps] == [[t, t] for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
    later = [(t, q.tolist()) for t, _logits, q, _z in segments(echo_model(), xb, T=2, state=steps[-1][3])]
    assert later == [(1, [6.0, 6.0]), (2, [7.0, 7.0])]           # segments 6 and 7 of the model, 1 and 2 of this call


@pytest.mark.parametrize("arch", [{}, {"mlp_t": True, "pos_encodings": "none"}], ids=["attention", "mlp_t"])
def test_a_roll_continued_from_its_kept_state_equals_the_uninterrupted_roll(arch):
    model = tiny_trm(**arch)
    rows = np.random.default_rng(0).integers(1, 5, (21, 9))      # chunks of 8, 8 and 5 (padded to 8)
    whole = roll(model, rows, T=5, batch=8, keep_state=True)
    first = roll(model, rows, T=2, batch=8, keep_state=True)
    rest = roll(model, rows, T=3, batch=8, state=first.state, keep_state=True)
    for name in ("dec", "q", "top_ids", "top_vals"):
        np.testing.assert_array_equal(getattr(rest, name), getattr(whole, name))
    for got, want in zip(rest.state, whole.state):
        assert got.shape == (21, 11, 32) and got.dtype == torch.float32 and got.device.type == "cpu"
        assert torch.equal(got, want)
    fresh = roll(model, rows, T=3, batch=8)                       # the state carries information: from the initial
    assert not np.array_equal(fresh.top_vals, rest.top_vals)      # state, the same 3 segments end elsewhere


def test_the_state_comes_back_in_the_forward_dtype():
    r = roll(tiny_trm(forward_dtype="bfloat16"), np.ones((3, 9), np.int64), T=1, batch=4, keep_state=True)
    assert all(z.shape == (3, 11, 32) and z.dtype == torch.bfloat16 and z.device.type == "cpu" for z in r.state)


def test_the_state_comes_back_only_when_kept():
    rows = np.ones((3, 2), np.int64)
    start = (torch.full((3, 1), 4.0), torch.zeros(3, 1))         # 4 segments into the echo model's count
    assert roll(echo_model(), rows, T=2, batch=4).state is None
    r = roll(echo_model(), rows, T=2, batch=4, state=start)
    np.testing.assert_array_equal(r.q, [6.0, 6.0, 6.0])
    assert r.state is None
    kept = roll(echo_model(), rows, T=2, batch=4, state=start, keep_state=True)
    assert kept.state[0][:, 0].tolist() == [6.0, 6.0, 6.0]


def test_a_short_chunk_pads_the_state_with_copies_of_its_first_row():
    seen = []

    def hook(t, z_L):
        seen.append(z_L[:, 0].tolist())
        return z_L

    z_L = torch.arange(6.0)[:, None]                             # row k's z_L is k
    r = roll(echo_model(), np.ones((6, 2), np.int64), T=1, batch=4, noise=hook, state=(torch.zeros(6, 1), z_L),
             keep_state=True)
    assert seen == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 4.0, 4.0]]  # chunks of 4 and 2 (padded to 4)
    assert r.state[1][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]     # the real rows' state, each in its place


@pytest.mark.parametrize("which, shape", [(0, (5, 11, 32)), (1, (6, 11, 16)), (0, (6, 10, 32)), (1, (6, 11))],
                         ids=["rows", "hidden", "positions", "rank"])
def test_a_state_of_another_shape_is_refused(which, shape):
    model = tiny_trm()
    state = list(model.initial_state(6))
    state[which] = torch.zeros(shape)
    error = f"roll needs a state of shape (6, 11, 32) for 6 rows, got {('z_H', 'z_L')[which]} of shape {shape}"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        roll(model, np.ones((6, 9), np.int64), T=1, batch=4, state=tuple(state))


@pytest.mark.parametrize("which", [0, 1], ids=["z_H", "z_L"])
def test_a_state_in_another_dtype_is_refused(which):
    # A float32 state would make a bfloat16 model's roll run in float32 (the latent's dtype carries through a segment).
    model = tiny_trm(forward_dtype="bfloat16")
    state = list(model.initial_state(6))
    state[which] = state[which].float()
    error = (f"roll needs a state in the model's forward dtype torch.bfloat16, got {('z_H', 'z_L')[which]} in "
             "torch.float32")
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        roll(model, np.ones((6, 9), np.int64), T=1, batch=4, state=tuple(state))
