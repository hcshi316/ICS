import pytest
import torch
import torch.nn.functional as F

from ics.checkpoint import save_checkpoint
from ics.data import IGNORE
from ics_baselines.gram.model import GRAM, GRAMConfig, gaussian_kl, load_gram


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_len": 2, "H_cycles": 2, "L_cycles": 2, "L_layers": 1,
            "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "pos_encodings": "rope", "forward_dtype": "float32"}
    base.update(kw)
    return GRAMConfig(**base)


LAYER = ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight")


def test_state_dict_names_are_grams():
    names = {"H_init", "L_init", "embed_tokens.embedding_weight", "puzzle_emb.embedding_weight", "lm_head.weight",
             "q_head.weight", "q_head.bias", "v_head.weight", "v_head.bias",
             *(f"{head}.net.{p}.weight" for head in ("prior_head", "post_head") for p in ("gate_up_proj", "down_proj")),
             *(f"{core}.layers.0.{p}" for core in ("f_L", "f_H") for p in LAYER)}
    assert set(GRAM(tiny()).state_dict()) == {f"inner.{n}" for n in names}
    mixer = set(GRAM(tiny(mlp_t=True, pos_encodings="none")).state_dict())
    assert "inner.f_H.layers.0.mlp_t.gate_up_proj.weight" in mixer and not any("self_attn" in k for k in mixer)


def test_refuses_other_positional_encodings():
    with pytest.raises(ValueError, match="'learned'"):
        GRAM(tiny(pos_encodings="learned"))


def test_initial_state_is_a_fresh_copy_of_z0():
    m = GRAM(tiny()).eval()
    z_H, z_L = m.initial_state(2)
    assert z_H.shape == z_L.shape == (2, 11, 32)
    torch.testing.assert_close(z_H[1, 3], m.inner.H_init, rtol=0, atol=0)
    torch.testing.assert_close(z_L[0, 0], m.inner.L_init, rtol=0, atol=0)
    z_H[0] += 1.0                                            # must not reach the buffer
    torch.testing.assert_close(m.initial_state(1)[0][0, 0], m.inner.H_init, rtol=0, atol=0)
    assert not torch.equal(z_H[0, 0], m.inner.H_init)


def test_step_draws_one_float32_normal_per_transition_over_the_whole_latent():
    torch.manual_seed(0)
    m = GRAM(tiny()).eval()
    x = torch.randint(0, 5, (3, 9), dtype=torch.int32)
    gen, replay = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    z_H, z_L, logits, q, v = m.step(*m.initial_state(3), x, gen)
    for _ in range(2):                                       # H_cycles transitions
        torch.randn((3, 11, 32), generator=replay, dtype=torch.float32)
    assert torch.equal(gen.get_state(), replay.get_state())
    assert logits.shape == (3, 9, 5) and q.shape == v.shape == (3,) and q.dtype == v.dtype == torch.float32
    assert z_H.shape == z_L.shape == (3, 11, 32) and not z_H.requires_grad


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])      # bfloat16: e is drawn in float32, eps cast after
def test_a_transition_moves_z_H_to_the_proposal_plus_prior_noise(dtype):
    torch.manual_seed(0)
    cfg = tiny(H_cycles=1, forward_dtype=dtype)
    m = GRAM(cfg).eval()
    with torch.no_grad():
        m.inner.prior_head.net.down_proj.weight.normal_(0, 0.1)      # a prior with varied mean and scale
    x = torch.randint(0, 5, (2, 9), dtype=torch.int32)
    z_H, z_L = m.initial_state(2)
    new_H, new_L, *_ = m.step(z_H, z_L, x, torch.Generator().manual_seed(3))
    inner = m.inner
    e, cos_sin = inner.embed(x, torch.zeros(2, dtype=torch.int32)), inner.rotary_emb()
    zl = z_L
    for _ in range(cfg.L_cycles):
        zl = inner.f_L(zl, z_H + e, cos_sin=cos_sin)
    u = inner.f_H(z_H, zl, cos_sin=cos_sin)
    mu, raw = inner.prior_head(u)
    eps = torch.randn(u.shape, generator=torch.Generator().manual_seed(3))
    expected = u + (mu.float() + (torch.nn.functional.softplus(raw.float()) + cfg.min_std) * eps).to(u.dtype)
    torch.testing.assert_close(new_H, expected, rtol=0, atol=0)
    torch.testing.assert_close(new_L, zl, rtol=0, atol=0)


def test_the_generator_seed_determines_the_step():
    torch.manual_seed(0)
    m = GRAM(tiny()).eval()
    x = torch.randint(0, 5, (2, 9), dtype=torch.int32)
    a, b, c = (m.step(*m.initial_state(2), x, torch.Generator().manual_seed(s)) for s in (1, 1, 2))
    assert all(torch.equal(u, w) for u, w in zip(a, b))
    assert not torch.equal(a[0], c[0])


def test_load_gram_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = tiny()
    m = GRAM(cfg)
    save_checkpoint(m.state_dict(), cfg, tmp_path / "ck")
    back = load_gram(tmp_path / "ck")
    assert not back.training and back.config == cfg
    for k, val in m.state_dict().items():
        torch.testing.assert_close(back.state_dict()[k], val, rtol=0, atol=0)


def trained(**kw) -> GRAM:
    """A tiny GRAM of 3 transitions whose noise heads' output projections are random, so that the posterior differs
    from the prior and the KL is not 0."""
    torch.manual_seed(0)
    m = GRAM(tiny(H_cycles=3, **kw))
    with torch.no_grad():
        for head in (m.inner.prior_head, m.inner.post_head):
            head.net.down_proj.weight.normal_(0, 0.1)
    return m.train()


def data(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (3, 9), generator=g, dtype=torch.int32)
    labels[0, :4] = IGNORE
    return {"inputs": torch.randint(1, 5, (3, 9), generator=g, dtype=torch.int32), "labels": labels,
            "puzzle_identifiers": torch.zeros(3, dtype=torch.int32)}


def test_a_training_step_samples_the_posterior_from_the_global_generator_and_sums_every_transitions_kl():
    m, b = trained(), data()
    z = m.initial_state(3)
    torch.manual_seed(5)
    z_H, z_L, logits, _, _, kl = m.train_step(*z, b)
    after = torch.get_rng_state()
    inner, cfg = m.inner, m.config
    x, cos_sin = inner.embed(b["inputs"], b["puzzle_identifiers"]), inner.rotary_emb()
    y = torch.where(b["labels"] == IGNORE, 0, b["labels"])
    e_y = inner.embed_scale * torch.cat((torch.zeros(3, 2, 32), inner.embed_tokens(y)), dim=-2)
    sigma = lambda raw: F.softplus(raw) + cfg.min_std
    torch.manual_seed(5)
    h, l, kls = *z, []
    with torch.no_grad():
        for _ in range(3):
            for _ in range(cfg.L_cycles):
                l = inner.f_L(l, h + x, cos_sin=cos_sin)
            u = inner.f_H(h, l, cos_sin=cos_sin)
            p_mu, p_raw = inner.prior_head(u)
            q_mu, q_raw = inner.post_head(torch.cat((u, e_y), dim=-1))
            h = u + (q_mu + sigma(q_raw) * torch.randn(u.shape))
            kls.append(gaussian_kl(q_mu, sigma(q_raw), p_mu, sigma(p_raw)).sum(-1).sum(-1))
        assert torch.equal(logits, inner.lm_head(h)[:, 2:])
    assert torch.equal(torch.get_rng_state(), after)               # one float32 normal per transition
    assert torch.equal(z_H, h) and torch.equal(z_L, l) and not z_H.requires_grad
    assert kl.shape == (3,) and (kl > 0).all()
    torch.testing.assert_close(kl, kls[2] + (kls[0] + kls[1]))    # the last transition's KL, then the other two


def test_the_balanced_kl_trains_the_prior_with_kl_balance_of_its_gradient_and_the_posterior_with_the_rest():
    one, zero = torch.ones(1), torch.zeros(1)
    assert gaussian_kl(one, one, zero, one).item() == 0.5                       # KL(N(1, 1) || N(0, 1))
    torch.testing.assert_close(gaussian_kl(zero, 2 * one, zero, one), torch.tensor([1.5 - 0.6931472]))
    inner = GRAM(tiny(kl_balance=0.7)).inner
    g = torch.Generator().manual_seed(0)
    q_mu, p_mu = (torch.randn(3, 11, 32, generator=g).requires_grad_() for _ in range(2))
    q_sigma, p_sigma = (torch.rand(3, 11, 32, generator=g).add(0.5).requires_grad_() for _ in range(2))
    kl = inner.balanced_kl(q_mu, q_sigma, p_mu, p_sigma)
    whole = gaussian_kl(q_mu, q_sigma, p_mu, p_sigma)
    torch.testing.assert_close(kl, whole.sum(-1).sum(-1))         # [B]: summed over channels and positions
    prior = torch.autograd.grad(whole.sum(), (p_mu, p_sigma), retain_graph=True)
    post = torch.autograd.grad(whole.sum(), (q_mu, q_sigma))
    got = torch.autograd.grad(kl.sum(), (p_mu, p_sigma, q_mu, q_sigma))
    for g_, want in zip(got, [0.7 * t for t in prior] + [0.3 * t for t in post]):
        torch.testing.assert_close(g_, want)


def test_only_the_last_transition_carries_gradient_and_the_heads_read_a_detached_latent():
    m = trained()
    z_H, z_L = (t.requires_grad_() for t in m.initial_state(3))
    _, _, logits, q, v, _ = m.train_step(z_H, z_L, data())
    (q.sum() + v.sum()).backward()
    assert {n for n, p in m.named_parameters() if p.grad is not None} == {
        "inner.q_head.weight", "inner.q_head.bias", "inner.v_head.weight", "inner.v_head.bias"}
    logits.sum().backward()
    assert z_H.grad is None and z_L.grad is None                  # the first transitions ran without gradient


def test_the_kl_of_an_earlier_transition_trains_the_noise_heads_and_the_label_embedding_alone():
    grads = []
    for every in (True, False):
        m = trained(kl_all_transitions=every)
        torch.manual_seed(5)
        m.train_step(*m.initial_state(3), data())[-1].sum().backward()
        grads.append({n: p.grad for n, p in m.named_parameters()})
    differ = {n for n, g in grads[0].items() if (g is None) != (grads[1][n] is None)
              or (g is not None and not torch.equal(g, grads[1][n]))}
    heads = {f"inner.{h}.net.{p}.weight" for h in ("prior_head", "post_head") for p in ("gate_up_proj", "down_proj")}
    assert differ == heads | {"inner.embed_tokens.embedding_weight"}           # the cores' gradients are equal


def test_gradient_checkpointing_recomputes_the_cores_and_changes_no_value_or_gradient():
    out = {}
    for ckpt in (True, False):
        m, calls = trained(grad_checkpoint=ckpt), []
        m.inner.f_L.register_forward_pre_hook(lambda module, args, calls=calls: calls.append(1))
        torch.manual_seed(5)
        _, _, logits, q, v, kl = m.train_step(*m.initial_state(3), data())
        loss = logits.sum() + kl.sum() + q.sum() + v.sum()
        grads = torch.autograd.grad(loss, list(m.parameters()))   # reentrant checkpointing refuses autograd.grad
        out[ckpt] = (loss, len(calls), grads)
    assert torch.equal(out[True][0], out[False][0])
    assert out[False][1] == 6 and out[True][1] == 8               # the last transition's 2 f_L calls, again
    assert all(torch.equal(a, b) for a, b in zip(out[True][2], out[False][2]))
