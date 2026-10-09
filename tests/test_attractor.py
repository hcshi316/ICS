import dataclasses
import math

import torch

from ics.checkpoint import save_checkpoint
from ics.trm.model import TRM, TRMConfig
from ics_baselines.attractor.model import Attractor, AttractorConfig, AttractorInner, anderson, load_attractor


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "H_cycles": 2, "L_cycles": 6,
            "L_layers": 2, "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32",
            "deq_max_iter": 6, "deq_min_iter": 3}
    base.update(kw)
    return AttractorConfig(**base)


def test_the_weights_are_a_trms():
    cfg = tiny()
    trm = TRMConfig(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(TRMConfig)})
    assert set(Attractor(cfg).state_dict()) == set(TRM(trm).state_dict())


def drifting(scales):
    """f(y) = 1 + scale * 0.5**t on its t-th evaluation, whatever y is: a row's residual shrinks with its scale. The
    dtype of every evaluation's input is recorded."""
    calls = []

    def f(y):
        calls.append(y.dtype)
        return (torch.ones(y.shape) + scales.view(-1, 1, 1) * 0.5 ** len(calls)).to(y.dtype)
    return f, calls


def test_anderson_finds_the_fixed_point():
    a = torch.tensor([0.5, 0.9]).view(2, 1, 1)
    y = anderson(lambda y: a * y + 1, torch.zeros(2, 3, 4), max_iter=30, tol=1e-6, min_iter=4, m=5, beta=1.0)
    torch.testing.assert_close(y, (1 / (1 - a)).expand(2, 3, 4), rtol=1e-4, atol=1e-4)


def test_anderson_evaluates_first_in_the_latent_dtype_then_in_float32():
    f, calls = drifting(torch.tensor([1.0]))
    y = anderson(f, torch.zeros(1, 2, 3, dtype=torch.bfloat16), max_iter=5, tol=1e-3, min_iter=4, m=5, beta=1.0)
    assert y.dtype == torch.bfloat16
    assert calls[0] == torch.bfloat16 and set(calls[1:]) == {torch.float32}


def test_the_rows_of_a_batch_are_solved_together():
    kw = {"max_iter": 8, "tol": 1e-3, "min_iter": 4, "m": 5, "beta": 1.0}
    f, alone = drifting(torch.tensor([1e-4]))
    y_alone = anderson(f, torch.zeros(1, 2, 3), **kw)
    f, both = drifting(torch.tensor([1e-4, 1.0]))
    y_both = anderson(f, torch.zeros(2, 2, 3), **kw)
    assert len(alone) == 4 and len(both) == 8            # the batch runs until its slowest row converges ...
    assert not torch.equal(y_alone[0], y_both[0])        # ... so a row's result depends on the rest of its batch


def test_initial_state_adds_the_perturbation_in_the_model_dtype():
    m = Attractor(tiny(forward_dtype="bfloat16"))
    dH, dL = torch.randn(32), torch.randn(32)
    z_H, z_L = m.initial_state(2, dH, dL)
    torch.testing.assert_close(z_H[1, 5], m.inner.H_init + dH.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(z_L[0, 0], m.inner.L_init + dL.to(torch.bfloat16), rtol=0, atol=0)
    z_H, z_L = m.initial_state(1)                        # no perturbation: z0 itself
    torch.testing.assert_close(z_H[0, 0], m.inner.H_init, rtol=0, atol=0)
    assert z_H.shape == (1, 11, 32)


def test_a_segment_solves_then_applies_the_map_bptt_through_times():
    torch.manual_seed(0)
    cfg = tiny(deq_min_iter=3, deq_max_iter=3, bptt_through=2)     # every solve: exactly 3 map evaluations
    m = Attractor(cfg).eval()
    calls = []
    m.inner.L_level.register_forward_hook(lambda mod, args, out: calls.append(1))
    x = torch.randint(0, 5, (2, 9), dtype=torch.int32)
    z_H, _z_L, logits, q = m.segment(*m.initial_state(2), x, torch.zeros(2, dtype=torch.int32))
    assert len(calls) == cfg.H_cycles * (3 + 2 + 1)      # per H cycle: the solve, bptt_through, the z_H update
    assert logits.shape == (2, 9, 5) and q.shape == (2,) and q.dtype == torch.float32 and z_H.shape == (2, 11, 32)


def test_load_attractor_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = tiny()
    m = Attractor(cfg)
    save_checkpoint(m.state_dict(), cfg, tmp_path / "ck")
    back = load_attractor(tmp_path / "ck")
    assert not back.training and back.config == cfg and isinstance(back, Attractor)
    for k, val in m.state_dict().items():
        torch.testing.assert_close(back.state_dict()[k], val, rtol=0, atol=0)


def inputs_and_ids(n: int = 2):
    return torch.randint(0, 5, (n, 9), dtype=torch.int32), torch.zeros(n, dtype=torch.int32)


def test_the_regulariser_is_the_mean_square_of_j_transpose_v_for_a_probe_over_sqrt_hidden():
    torch.manual_seed(0)
    inner = Attractor(tiny()).train().inner
    z, ctx = torch.randn(2, 11, 32, requires_grad=True), torch.randn(2, 11, 32)
    seq_info = {"cos_sin": inner.rotary_emb()}
    torch.manual_seed(1)
    reg = inner.regulariser(z, ctx, seq_info)
    torch.manual_seed(1)
    v = torch.randn(2, 11, 32) / math.sqrt(32)
    J = torch.autograd.functional.jacobian(lambda y: inner.L_level(y, ctx, **seq_info), z)   # [2, 11, 32] x [2, 11, 32]
    torch.testing.assert_close(reg, torch.einsum("abc,abcdef->def", v, J).pow(2).mean(), rtol=1e-4, atol=1e-9)
    assert reg.requires_grad                                       # create_graph: the loss backpropagates through it
    assert torch.autograd.grad(reg, z, allow_unused=True)[0] is None   # but not into z: it is taken at z detached


def test_training_takes_a_regulariser_after_every_solve_and_returns_the_last_cycles(monkeypatch):
    torch.manual_seed(0)
    m = Attractor(tiny(H_cycles=3, batch_size=2)).train()
    taken, solved = [], []
    regulariser, solve = AttractorInner.regulariser, AttractorInner.solve

    def recorded_regulariser(self, z, ctx, seq_info):
        taken.append((z.detach().clone(), regulariser(self, z, ctx, seq_info)))
        return taken[-1][1]

    def recorded_solve(self, z_L, ctx, seq_info):
        z, reg = solve(self, z_L, ctx, seq_info)
        solved.append(z.detach().clone())
        return z, reg

    monkeypatch.setattr(AttractorInner, "regulariser", recorded_regulariser)
    monkeypatch.setattr(AttractorInner, "solve", recorded_solve)
    x, ids = inputs_and_ids()
    torch.manual_seed(1)
    *_, reg = m.regularised_segment(*m.initial_state(2), x, ids)
    after = torch.get_rng_state()
    assert len(taken) == len(solved) == 3                         # one per H cycle, the warm-up cycles' included
    assert all(torch.equal(z, s) for (z, _), s in zip(taken, solved))   # at the solve's result, after bptt_through
    assert reg is taken[-1][1] and reg.requires_grad              # the last cycle's, in the graph
    assert not taken[0][1].requires_grad and not taken[1][1].requires_grad   # the warm-ups': computed, discarded
    torch.manual_seed(1)
    for _ in range(3):
        torch.randn(2, 11, 32)                                    # the probes are the segment's only draws
    assert torch.equal(torch.get_rng_state(), after)
    taken.clear()
    assert m.eval().regularised_segment(*m.initial_state(2), x, ids)[4] is None and not taken   # none in eval mode
    m.train().config.jacobian_reg_lambda = 0.0
    assert m.regularised_segment(*m.initial_state(2), x, ids)[4] is None and not taken          # nor at weight 0


def test_the_regulariser_reaches_the_weights_and_the_embeddings_through_the_context_not_the_heads():
    torch.manual_seed(0)
    m = Attractor(tiny(batch_size=2)).train()
    *_, reg = m.regularised_segment(*m.initial_state(2), *inputs_and_ids())
    p = dict(m.named_parameters())
    wanted = [p["inner.embed_tokens.embedding_weight"], p["inner.L_level.layers.0.mlp.down_proj.weight"],
              m.inner.puzzle_emb.local_weights, p["inner.lm_head.weight"], p["inner.q_head.weight"]]
    tokens, block, puzzle, lm, q = torch.autograd.grad(reg, wanted, allow_unused=True)
    assert tokens.abs().sum() > 0 and puzzle.abs().sum() > 0      # through ctx = z_H + x, which keeps x's graph
    assert block.abs().sum() > 0 and lm is None and q is None


def test_only_the_last_cycles_map_applications_after_its_solve_carry_gradient():
    torch.manual_seed(0)
    cfg = tiny(batch_size=2, jacobian_reg_lambda=0.0)
    m = Attractor(cfg).train()
    x, ids = inputs_and_ids()
    _, _, logits, q, reg = m.regularised_segment(*m.initial_state(2), x, ids)
    assert reg is None
    (logits.sum() + q.sum()).backward()
    got = {name: p.grad.clone() for name, p in m.named_parameters()}
    m.zero_grad()
    inner, seq_info = m.inner, {"cos_sin": m.inner.rotary_emb()}
    kw = {"max_iter": cfg.deq_max_iter, "tol": cfg.deq_tol, "min_iter": cfg.deq_min_iter, "m": cfg.deq_anderson_m,
          "beta": cfg.deq_anderson_beta}
    e = inner._input_embeddings(x, ids)
    z_H, z_L = m.initial_state(2)
    for cycle in range(cfg.H_cycles):
        with torch.set_grad_enabled(cycle == cfg.H_cycles - 1):  # the warm-up cycles carry no gradient
            ctx = z_H + e
            with torch.no_grad():                                 # nor does the solve
                z_L = anderson(lambda z, ctx=ctx: inner.L_level(z, ctx, **seq_info), z_L, **kw)
            for _ in range(cfg.bptt_through):
                z_L = inner.L_level(z_L, ctx, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
    (inner.lm_head(z_H)[:, 2:].sum() + inner.q_head(z_H[:, 0]).to(torch.float32)[:, 0].sum()).backward()
    for name, p in m.named_parameters():
        assert torch.equal(p.grad, got[name]), name
