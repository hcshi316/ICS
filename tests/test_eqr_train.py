import numpy as np
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounter

from fakes import sudoku_pool
from ics.data import IGNORE
from ics.optim import AdamATan2
from ics.tasks import make_task
from ics.trm.layers import trunc_normal_init_
from ics.trm.train import stablemax_cross_entropy
from ics_baselines.eqr.model import EqR, EqRConfig
from ics_baselines.eqr.predict import run_eqr
from ics_baselines.eqr.train import EqRHead


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "H_cycles": 2, "L_cycles": 2, "L_layers": 1, "hidden_size": 32,
            "num_heads": 2, "expansion": 2.0, "pos_encodings": "rope", "forward_dtype": "float32",
            "halt_max_steps": 3, "halt_exploration_prob": 0.0}
    base.update(kw)
    return EqRConfig(**base)


def head(**kw) -> EqRHead:
    torch.manual_seed(0)
    return EqRHead(tiny(**kw)).train()


def batch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32)
    labels[0, :3] = IGNORE
    return {"inputs": torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32), "labels": labels,
            "puzzle_identifiers": torch.zeros(4, dtype=torch.int32)}


def draw(n: int, std: float = 1.0) -> torch.Tensor:
    """n rows of a fresh latent, as upstream's reset_carry draws them (here in float32, the tiny model's dtype)."""
    return trunc_normal_init_(torch.empty(n, 9, 32), std=std)


def test_building_the_head_draws_the_models_weights_and_nothing_else():
    torch.manual_seed(0)
    model = EqR(tiny())
    state = torch.get_rng_state()
    h = head()
    assert torch.equal(torch.get_rng_state(), state)
    for name, value in model.state_dict().items():
        assert torch.equal(h.model.state_dict()[name], value), name


def test_a_restart_draws_z_H_then_z_L_for_the_restarting_rows_only():
    h = head(L_init_std=0.5)
    carry = {"z_H": torch.randn(4, 9, 32), "z_L": torch.randn(4, 9, 32)}
    torch.manual_seed(3)
    z_H, z_L = h.restart(carry, torch.tensor([True, False, True, False]))
    torch.manual_seed(3)
    H, L = draw(2), draw(2, 0.5)
    assert torch.equal(z_H[[0, 2]], H) and torch.equal(z_L[[0, 2]], L)
    assert torch.equal(z_H[[1, 3]], carry["z_H"][[1, 3]]) and torch.equal(z_L[[1, 3]], carry["z_L"][[1, 3]])
    state = torch.get_rng_state()
    kept = h.restart(carry, torch.zeros(4, dtype=torch.bool))     # no row restarts: nothing is drawn
    assert kept[0] is carry["z_H"] and kept[1] is carry["z_L"] and torch.equal(torch.get_rng_state(), state)


def test_a_step_draws_the_restarts_then_the_update_noise_then_exploration():
    h, b = head(halt_max_steps=8, halt_exploration_prob=0.5), batch(1)
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)                      # q_halt > 0: a row halts unless exploration holds it
    torch.manual_seed(5)
    carry, loss, _ = h(h.initial_carry(b), b)
    after = torch.get_rng_state()
    torch.manual_seed(5)                                          # the same draws, replayed
    z_H, z_L, logits, q = h.model.step(draw(4), draw(4), b["inputs"], None)
    minimum = (torch.rand(4) < 0.5) * torch.randint(2, 9, (4,), dtype=torch.int32)
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(carry["z_H"], z_H) and torch.equal(carry["z_L"], z_L)
    assert carry["halted"].tolist() == (minimum <= 1).tolist()   # held back only by exploration
    mask = b["labels"] != IGNORE
    ce = stablemax_cross_entropy(logits, b["labels"], mask).sum(-1) / mask.sum(-1)
    exact = ((logits.argmax(-1) == b["labels"]) | ~mask).all(-1)
    bce = F.binary_cross_entropy_with_logits(q, exact.float(), reduction="sum")
    torch.testing.assert_close(loss, ce.sum() + 0.5 * bce.double())


def test_continuing_rows_keep_their_latent_and_example_and_restarting_rows_take_the_new_batch():
    h, b1, b2 = head(), batch(1), batch(2)
    carry, _, _ = h(h.initial_carry(b1), b1)
    carry["halted"] = torch.tensor([True, False, True, False])
    torch.manual_seed(9)
    after, _, _ = h(carry, b2)
    torch.manual_seed(9)
    z_H, z_L = h.restart(carry, carry["halted"])                 # the same draws: rows 0 and 2 restart
    inputs = torch.stack([b2["inputs"][0], b1["inputs"][1], b2["inputs"][2], b1["inputs"][3]])
    z_H, z_L, _, _ = h.model.step(z_H, z_L, inputs, None)
    assert torch.equal(after["z_H"], z_H) and torch.equal(after["z_L"], z_L)
    assert after["steps"].tolist() == [1, 2, 1, 2]
    for r, src in enumerate([b2, b1, b2, b1]):
        assert torch.equal(after["inputs"][r], src["inputs"][r]) and torch.equal(after["labels"][r], src["labels"][r])


def test_a_compiled_head_runs_the_restart_eagerly_and_the_rest_as_one_graph():
    h = head(halt_max_steps=8, halt_exploration_prob=0.5)
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)                      # q_halt > 0: a row halts unless exploration holds it

    def steps(step) -> list:
        torch.manual_seed(5)
        carry, out = h.initial_carry(batch(0)), []
        for k in range(4):
            carry, loss, _ = step(carry, batch(k))
            out.append((carry, loss))
        return out

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled, eager = steps(torch.compile(h, backend=counter)), steps(h)
    assert any(0 < c["halted"].sum() < 4 for c, _ in eager[:-1])  # some steps restart only some of the rows
    for (c, loss), (c0, loss0) in zip(compiled, eager):
        assert torch.equal(loss, loss0) and all(torch.equal(c[k], c0[k]) for k in c0)
    assert counter.frame_count == 1                               # restart's rows depend on the data: it runs eager


def test_evaluation_scores_one_restart_of_halt_max_steps_steps_on_its_own_generators():
    model = head(seq_len=81, vocab_size=11, halt_max_steps=2).model.eval()
    pool = sudoku_pool([[0, 1], [5], [], [7, 8, 9]])
    state = torch.get_rng_state()
    values = EqRHead.evaluate(model, pool, "sudoku", 4)
    assert torch.equal(torch.get_rng_state(), state)              # the training generator is not touched
    task = make_task("sudoku", pool)
    dec = run_eqr(model, task, R=1, steps=2, batch=4)["eqr/model/raw"]
    assert np.array_equal(EqRHead.decode(model, task, 4), dec)
    assert not np.array_equal(run_eqr(model, task, R=1, steps=3, batch=4)["eqr/model/raw"], dec)  # steps matter
    want = task.check("raw", np.arange(4), dec)
    assert values.dtype == np.float64 and values.tolist() == want.astype(np.float64).tolist()
    assert EqRHead.evaluate(model, pool.take(slice(0, 0)), "sudoku", 4).shape == (0,)


def test_the_optimizer_is_adam_atan2_on_every_weight():
    h = head()
    [(opt, lr)] = h.optimizers(lr=1e-4, weight_decay=1.0, betas=[0.9, 0.95])
    assert type(opt) is AdamATan2 and lr == 1e-4
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in h.model.parameters()]
    assert opt.param_groups[0]["betas"] == (0.9, 0.95) and opt.param_groups[0]["weight_decay"] == 1.0
