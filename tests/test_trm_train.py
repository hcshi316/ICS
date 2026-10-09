import numpy as np
import torch
import torch.nn.functional as F

from fakes import sudoku_pool
from ics.data import IGNORE
from ics.optim import AdamATan2, SparseSignSGD
from ics.trm.model import TRMConfig
from ics.trm.train import TRMHead, stablemax_cross_entropy


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "H_cycles": 2, "L_cycles": 2,
            "L_layers": 1, "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32",
            "batch_size": 4, "halt_max_steps": 3, "halt_exploration_prob": 0.0}
    base.update(kw)
    return TRMConfig(**base)


def head(**kw) -> TRMHead:
    torch.manual_seed(0)
    return TRMHead(tiny(**kw)).train()


def batch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32)
    labels[0, :3] = IGNORE
    return {"inputs": torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32), "labels": labels,
            "puzzle_identifiers": torch.zeros(4, dtype=torch.int32)}


def test_stablemax_cross_entropy():
    logits = torch.tensor([[[2.0, -1.0, 0.5], [0.0, -3.0, 1.0]]])
    labels = torch.tensor([[2, IGNORE]])
    got = stablemax_cross_entropy(logits, labels, labels != IGNORE)
    s = torch.tensor([3.0, 0.5, 1.5], dtype=torch.float64)                      # x + 1, or 1 / (1 - x) below 0
    assert got.dtype == torch.float64 and got[0, 1] == 0
    torch.testing.assert_close(got[0, 0], -torch.log(s[2] / s.sum()))


def test_loss_is_the_row_mean_cross_entropy_plus_half_the_q_halt_bce():
    h, b = head(), batch(3)
    _, loss, stats = h(h.initial_carry(b), b)
    _, _, logits, q = h.model.segment(*h.model.initial_state(4), b["inputs"], b["puzzle_identifiers"])
    mask = b["labels"] != IGNORE
    ce = stablemax_cross_entropy(logits, b["labels"], mask).sum(-1) / mask.sum(-1)
    exact = ((logits.argmax(-1) == b["labels"]) | ~mask).all(-1)
    bce = F.binary_cross_entropy_with_logits(q, exact.float(), reduction="sum")
    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, ce.sum() + 0.5 * bce.double())
    torch.testing.assert_close(stats["lm_loss"][0], ce.sum())
    assert stats["lm_loss"][1] == 4


def test_halted_rows_take_the_new_batch_and_restart_the_others_continue():
    h, b1, b2 = head(), batch(1), batch(2)
    carry, _, _ = h(h.initial_carry(b1), b1)
    assert torch.equal(carry["inputs"], b1["inputs"]) and carry["steps"].tolist() == [1, 1, 1, 1]
    carry["halted"] = torch.tensor([True, False, True, False])
    carry, _, _ = h(carry, b2)
    for r, src in enumerate([b2, b1, b2, b1]):
        assert torch.equal(carry["inputs"][r], src["inputs"][r]) and torch.equal(carry["labels"][r], src["labels"][r])
    assert carry["steps"].tolist() == [1, 2, 1, 2]
    fresh, _, _ = h(h.initial_carry(b2), b2)                     # a restarted row starts from (H_init, L_init)
    torch.testing.assert_close(carry["z_H"][[0, 2]], fresh["z_H"][[0, 2]], rtol=0, atol=0)
    z = h.model.initial_state(4)                                 # a continuing row keeps its latent: its two
    for _ in range(2):                                           # segments run one after the other
        *z, _, _ = h.model.segment(*z, b1["inputs"], b1["puzzle_identifiers"])
    for got, want in zip((carry["z_H"], carry["z_L"]), z):
        torch.testing.assert_close(got[[1, 3]], want[[1, 3]], rtol=0, atol=0)


def test_rows_halt_after_halt_max_steps():
    h = head()
    carry, steps = h.initial_carry(batch(0)), []
    for k in range(7):
        carry, _, _ = h(carry, batch(k))
        steps.append(int(carry["steps"][0]))
    assert steps == [1, 2, 3, 1, 2, 3, 1]


def test_q_halt_halts_a_row_unless_exploration_holds_it():
    for prob, halted in ((0.0, True), (1.0, False)):
        h = head(halt_max_steps=8, halt_exploration_prob=prob)
        with torch.no_grad():
            h.model.inner.q_head.bias.fill_(5.0)                  # q_halt > 0 on every row
        carry, _, _ = h(h.initial_carry(batch(1)), batch(1))
        assert carry["halted"].tolist() == [halted] * 4           # exploration: at least 2 segments


def test_exploration_draws_rand_then_randint_at_every_segment():
    h, b = head(halt_max_steps=8, halt_exploration_prob=0.5), batch(1)
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)                      # q_halt > 0: a row halts unless exploration holds it
    torch.manual_seed(7)
    carry, seen = h.initial_carry(b), []
    for _ in range(6):
        carry, _, _ = h(carry, b)
        seen.append(carry["halted"].tolist())
    torch.manual_seed(7)                                          # the same draws, replayed
    steps, halted, replayed = torch.zeros(4, dtype=torch.int32), torch.ones(4, dtype=torch.bool), []
    for _ in range(6):
        steps = torch.where(halted, 0, steps) + 1
        minimum = (torch.rand(4) < 0.5) * torch.randint(2, 9, (4,), dtype=torch.int32)
        halted = steps >= minimum
        replayed.append(halted.tolist())
    assert seen == replayed and 0 < sum(map(sum, seen)) < 24      # some rows held back, some halted


def test_statistics_count_the_rows_that_end_an_example():
    h, b = head(halt_max_steps=2), batch(3)
    z = h.model.initial_state(4)
    for _ in range(2):                                            # every row ends its example at its 2nd segment
        *z, logits, q = h.model.segment(*z, b["inputs"], b["puzzle_identifiers"])
    b["labels"][1:3] = logits[1:3].argmax(-1)                     # rows 1 and 2 decode exactly,
    b["labels"][3] = IGNORE                                       # and row 3 has no label
    carry = h.initial_carry(b)
    for _ in range(2):
        carry, _, stats = h(carry, b)
    exact = ((logits.argmax(-1) == b["labels"]) | (b["labels"] == IGNORE)).all(-1)
    assert exact.tolist() == [False, True, True, True]
    assert [int(x) for x in stats["exact"]] == [2, 3] and [int(x) for x in stats["segments"]] == [6, 3]
    bce = F.binary_cross_entropy_with_logits(q, exact.float(), reduction="sum")
    torch.testing.assert_close(stats["q_halt_loss"][0], bce, rtol=0, atol=0)
    assert stats["q_halt_loss"][1] == 4


def test_eval_mode_runs_every_row_halt_max_steps_without_drawing():
    h = head(halt_max_steps=2, halt_exploration_prob=1.0).eval()
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)
    state = torch.get_rng_state()
    carry, _, _ = h(h.initial_carry(batch(1)), batch(1))
    assert not carry["halted"].any()
    carry, _, _ = h(carry, batch(1))
    assert carry["halted"].all() and torch.equal(torch.get_rng_state(), state)


def test_optimizers_are_the_embedding_sign_sgd_then_adam_atan2():
    settings = {"lr": 1e-4, "weight_decay": 1.0, "betas": [0.9, 0.95], "puzzle_emb_lr": 1e-2,
                "puzzle_emb_weight_decay": 0.1}
    h = head()
    opts = h.optimizers(**settings)
    assert [type(o) for o, _ in opts] == [SparseSignSGD, AdamATan2] and [lr for _, lr in opts] == [1e-2, 1e-4]
    assert [id(p) for p in opts[1][0].param_groups[0]["params"]] == [id(p) for p in h.model.parameters()]
    assert [type(o) for o, _ in head(puzzle_emb_ndim=0).optimizers(**settings)] == [AdamATan2]


def test_the_carry_starts_with_zero_latents_of_the_models_shape_and_every_row_halted():
    h, b = head(forward_dtype="bfloat16"), batch(0)
    carry = h.initial_carry(b)
    assert carry["z_H"].shape == carry["z_L"].shape == (4, 11, 32)
    assert carry["z_H"].dtype == carry["z_L"].dtype == torch.bfloat16              # the model's forward dtype
    assert not carry["z_H"].any() and not carry["z_L"].any()      # never read: every row restarts at its first step
    assert carry["halted"].all() and not carry["steps"].any() and not carry["inputs"].any()


def test_a_variant_overrides_the_restart_the_segment_and_the_decode():
    seen = []

    class Variant(TRMHead):
        def restart(self, carry, halted):
            z_H, z_L = super().restart(carry, halted)
            return z_H + 1, z_L

        def segment(self, z_H, z_L, data):
            seen.append(z_H.clone())
            return super().segment(z_H, z_L, data)

        @staticmethod
        def decode(model, task, batch):
            return task.Y.astype(np.int16)                        # every board answered with its label

    torch.manual_seed(0)
    v, b = Variant(tiny()).train(), batch(1)
    v(v.initial_carry(b), b)
    assert torch.equal(seen[0], v.model.initial_state(4)[0] + 1)
    pool = sudoku_pool([[0], [1, 2]])
    assert Variant.evaluate(v.model.eval(), pool, "sudoku", 2).tolist() == [1.0, 1.0]


def test_a_variant_adds_its_loss_terms_with_their_weights_and_logs_them():
    class Variant(TRMHead):
        def segment_terms(self, z_H, z_L, data):
            *out, terms = super().segment_terms(z_H, z_L, data)
            assert terms == {}                                    # TRM's segment has none
            return (*out, {"square": (0.25, out[2].square().mean())})     # out[2]: the logits

    b = batch(1)
    torch.manual_seed(0)
    v, h = Variant(tiny()).train(), head()                        # the same weights (seed 0)
    _, loss, stats = v(v.initial_carry(b), b)
    _, base, base_stats = h(h.initial_carry(b), b)
    _, _, logits, _ = h.model.segment(*h.model.initial_state(4), b["inputs"], b["puzzle_identifiers"])
    term = logits.square().mean()
    assert torch.equal(loss, base + 0.25 * term)
    assert torch.equal(stats["square"][0], term.detach()) and stats["square"][1] == 1
    assert not stats["square"][0].requires_grad                   # logged detached, out of the graph
    assert set(stats) - {"square"} == set(base_stats)
    with_term = torch.autograd.grad(loss, v.model.inner.lm_head.weight)[0]
    assert not torch.equal(with_term, torch.autograd.grad(base, h.model.inner.lm_head.weight)[0])  # it trains too
