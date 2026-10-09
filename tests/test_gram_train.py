import numpy as np
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounter

from fakes import sudoku_pool
from ics.data import IGNORE
from ics.tasks import make_task
from ics.trm.train import stablemax_cross_entropy
from ics_baselines.gram.model import GRAM, GRAMConfig
from ics_baselines.gram.predict import run_gram
from ics_baselines.gram.train import GRAMHead


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_len": 2, "H_cycles": 3, "L_cycles": 2, "L_layers": 1,
            "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "pos_encodings": "rope", "forward_dtype": "float32",
            "halt_max_steps": 3}
    base.update(kw)
    return GRAMConfig(**base)


def head(**kw) -> GRAMHead:
    """A tiny head whose noise heads' and value head's output weights are random: the posterior differs from the
    prior, and the LPRM's values from 0.5."""
    torch.manual_seed(0)
    h = GRAMHead(tiny(**kw)).train()
    inner = h.model.inner
    with torch.no_grad():
        for lin in (inner.prior_head.net.down_proj, inner.post_head.net.down_proj, inner.v_head):
            lin.weight.normal_(0, 0.1)
    return h


def batch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32)
    labels[0, :3] = IGNORE
    return {"inputs": torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32), "labels": labels,
            "puzzle_identifiers": torch.zeros(4, dtype=torch.int32)}


def test_building_the_head_draws_the_models_weights_and_nothing_else():
    torch.manual_seed(0)
    model = GRAM(tiny())
    state = torch.get_rng_state()
    torch.manual_seed(0)
    h = GRAMHead(tiny())
    assert torch.equal(torch.get_rng_state(), state)
    for name, value in model.state_dict().items():
        assert torch.equal(h.model.state_dict()[name], value), name
    z0 = GRAMHead(tiny(forward_dtype="bfloat16")).initial_carry(batch(0))["z0"]
    assert z0.shape == (3, 4, 32) and z0.dtype == torch.bfloat16 and not z0.any()


def test_a_step_restarts_from_z0_samples_the_posterior_and_no_row_halts_early():
    h, b = head(), batch(1)
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)                      # q_halt > 0: a TRM row would halt here
    torch.manual_seed(5)
    carry, loss, stats = h(h.initial_carry(b), b)
    after = torch.get_rng_state()
    torch.manual_seed(5)                                          # the same draws, replayed: the posterior's alone
    z_H, z_L, logits, q, _, kl = h.model.train_step(*h.model.initial_state(4), b)
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(carry["z_H"], z_H) and torch.equal(carry["z_L"], z_L)
    assert carry["steps"].tolist() == [1] * 4 and not carry["halted"].any() and "v_loss" not in stats
    mask = b["labels"] != IGNORE
    ce = stablemax_cross_entropy(logits, b["labels"], mask).sum(-1) / mask.sum(-1)
    exact = ((logits.argmax(-1) == b["labels"]) | ~mask).all(-1)
    bce = F.binary_cross_entropy_with_logits(q, exact.float(), reduction="sum")
    assert kl.sum() > 0 and torch.equal(stats["kl"][0], kl.sum()) and stats["kl"][1] == 4
    torch.testing.assert_close(loss, ce.sum() + 0.1 * kl.sum().double() + 0.5 * bce.double())


def test_every_row_restarts_from_z0_every_halt_max_steps_steps_taking_that_steps_batch():
    h = head()
    carry, seen = h.initial_carry(batch(0)), []
    for k in range(7):                                            # halt_max_steps 3: restarts at steps 1, 4 and 7
        if k == 3:
            state = torch.get_rng_state()
        carry, _, _ = h(carry, batch(k))
        seen.append((carry["steps"].tolist(), bool(carry["halted"].all()), carry["inputs"].clone()))
        if k == 3:
            fourth = carry["z_H"]
    assert [s for s, _, _ in seen] == [[1] * 4, [2] * 4, [3] * 4] * 2 + [[1] * 4]
    assert [h_ for _, h_, _ in seen] == [False, False, True] * 2 + [False]
    for k, (_, _, inputs) in enumerate(seen):
        assert torch.equal(inputs, batch(3 * (k // 3))["inputs"])   # the batch of the cycle's first step
    torch.set_rng_state(state)                                    # step 4 restarts every row from z0
    assert torch.equal(fourth, h.model.train_step(*h.model.initial_state(4), batch(3))[0])


def test_the_cycles_last_step_adds_grams_deferred_lprm_whose_gradient_reaches_v_head_alone():
    h = head()
    params = dict(h.model.named_parameters())
    torch.manual_seed(5)
    carry, combined = h.initial_carry(batch(0)), []
    for k in range(3):                                            # one cycle; the loop's backward of each step
        carry, loss, stats = h(carry, batch(k))
        ((1 / 768) * loss).backward()
        combined.append(({n: p.grad for n, p in params.items()}, loss.detach(), stats))
        h.zero_grad()
    # our GRAM reproduction's procedure: each step's loss backward, then at the cycle's end a second backward of
    # 0.5 * the sum over its steps of MSE(sigmoid(v_head(z0)), the last step's token accuracy), z0 viewed in each
    # step's latent
    torch.manual_seed(5)
    z, views, accuracies, inner, b = h.model.initial_state(4), [], [], h.model.inner, batch(0)
    labels = b["labels"]
    mask, counts = labels != IGNORE, (labels != IGNORE).sum(-1)
    for k in range(3):
        z_H, z_L, logits, q, _, kl = h.model.train_step(*z, b)    # every row keeps step 1's batch
        z = (z_H, z_L)
        correct = mask & (logits.argmax(-1) == labels)
        accuracies.append((correct.float() / counts.clamp_min(1).unsqueeze(-1)).sum(-1))
        lm = (stablemax_cross_entropy(logits, labels, mask) / counts.clamp_min(1).unsqueeze(-1)).sum()
        exact = (correct.sum(-1) == counts).float()
        main = lm + 0.1 * kl.sum() + 0.5 * F.binary_cross_entropy_with_logits(q, exact, reduction="sum")
        (main / 768).backward()
        views.append(z_H[:, 0])
        if k == 2:
            v_loss = sum(F.mse_loss(torch.sigmoid(inner.v_head(v).float().squeeze(-1)), accuracies[-1],
                                    reduction="sum") for v in views)
            (0.5 * v_loss / 768).backward()
        grads = combined[k][0]
        for n, p in params.items():
            assert (p.grad is None) == (grads[n] is None), (k, n)
            assert p.grad is None or torch.equal(p.grad, grads[n]), (k, n)
        h.zero_grad()
    assert any(not torch.equal(a, accuracies[-1]) for a in accuracies[:-1])   # the target is the last step's
    assert combined[1][0]["inner.v_head.weight"] is None and combined[2][0]["inner.v_head.weight"] is not None
    _, loss, stats = combined[2]
    assert torch.equal(loss, main.detach() + 0.5 * v_loss.detach())
    assert torch.equal(stats["v_loss"][0], v_loss.detach()) and stats["v_loss"][1] == 12


def test_evaluation_scores_one_prior_sample_of_halt_max_steps_steps_on_its_own_generators():
    model = head(seq_len=81, vocab_size=11, halt_max_steps=2).model.eval()
    pool = sudoku_pool([[0, 1], [5], [], [7, 8, 9]])
    state = torch.get_rng_state()
    values = GRAMHead.evaluate(model, pool, "sudoku", 4)
    assert torch.equal(torch.get_rng_state(), state)              # the training generator is not touched
    task = make_task("sudoku", pool)
    dec = run_gram(model, task, N=1, D=2, batch=4)["gram_lprm/model/raw"]
    assert np.array_equal(GRAMHead.decode(model, task, 4), dec)
    assert not np.array_equal(run_gram(model, task, N=1, D=3, batch=4)["gram_lprm/model/raw"], dec)  # D matters
    want = task.check("raw", np.arange(4), dec)
    assert values.dtype == np.float64 and values.tolist() == want.astype(np.float64).tolist()
    assert GRAMHead.evaluate(model, pool.take(slice(0, 0)), "sudoku", 4).shape == (0,)


def test_the_optimizer_is_adamw_on_every_weight_the_register_embedding_included():
    h = head()
    [(opt, lr)] = h.optimizers(lr=1e-4, weight_decay=1.0, betas=[0.9, 0.95])
    assert type(opt) is torch.optim.AdamW and lr == 1e-4
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in h.model.parameters()]
    group = opt.param_groups[0]
    assert (group["betas"], group["weight_decay"], group["eps"], group["amsgrad"]) == ((0.9, 0.95), 1.0, 1e-8, False)
    assert any(p is h.model.inner.puzzle_emb.embedding_weight for p in group["params"])


def test_a_compiled_head_trains_two_cycles_as_the_eager_one():
    def train(make) -> list:
        h = head(halt_max_steps=2)
        step, out = make(h), []
        torch.manual_seed(5)
        carry = h.initial_carry(batch(0))
        for k in range(4):                                        # two cycles: the LPRM at steps 2 and 4
            carry, loss, _ = step(carry, batch(k))
            ((1 / 768) * loss).backward()
            out.append((carry, loss.detach(), {n: p.grad for n, p in h.model.named_parameters()}))
            h.zero_grad()
        return out

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled, eager = train(lambda h: torch.compile(h, backend=counter)), train(lambda h: h)
    for (c, loss, g), (c0, loss0, g0) in zip(compiled, eager):
        assert torch.equal(loss, loss0) and all(torch.equal(c[k], c0[k]) for k in c0)
        assert all((g[n] is None) == (g0[n] is None) and (g[n] is None or torch.equal(g[n], g0[n])) for n in g0)
    assert counter.frame_count > 0                                # Dynamo compiled the step
    assert compiled[0][2]["inner.v_head.weight"] is None and compiled[1][2]["inner.v_head.weight"] is not None
