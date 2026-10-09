import numpy as np
import torch
import torch.nn.functional as F

from fakes import sudoku_pool
from ics.data import IGNORE
from ics.optim import SparseSignSGD
from ics.tasks import make_task
from ics.trm.train import stablemax_cross_entropy
from ics_baselines.attractor.model import AttractorConfig
from ics_baselines.attractor.predict import run_attractor
from ics_baselines.attractor.train import AttractorHead


def tiny(**kw):
    base = {"seq_len": 9, "vocab_size": 5, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "H_cycles": 2, "L_layers": 1,
            "hidden_size": 32, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32", "deq_max_iter": 4,
            "deq_min_iter": 3, "halt_max_steps": 3, "halt_exploration_prob": 0.0, "batch_size": 4}
    base.update(kw)
    return AttractorConfig(**base)


def head(**kw) -> AttractorHead:
    torch.manual_seed(0)
    return AttractorHead(tiny(**kw)).train()


def batch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32)
    labels[0, :3] = IGNORE
    return {"inputs": torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32), "labels": labels,
            "puzzle_identifiers": torch.zeros(4, dtype=torch.int32)}


def test_the_loss_adds_the_weighted_regulariser_of_the_last_cycle_and_logs_it():
    h, b = head(jacobian_reg_lambda=0.25), batch(1)
    torch.manual_seed(5)
    carry, loss, stats = h(h.initial_carry(b), b)
    torch.manual_seed(5)                                          # the same draws, replayed
    z_H, _, logits, q, reg = h.model.regularised_segment(*h.model.initial_state(4), b["inputs"],
                                                          b["puzzle_identifiers"])
    mask = b["labels"] != IGNORE
    lm = (stablemax_cross_entropy(logits, b["labels"], mask) / mask.sum(-1, keepdim=True)).sum()
    exact = ((logits.argmax(-1) == b["labels"]) | ~mask).all(-1)
    bce = F.binary_cross_entropy_with_logits(q, exact.float(), reduction="sum")
    assert torch.equal(loss, lm + 0.5 * bce + 0.25 * reg)         # upstream's sum, in its order
    assert torch.equal(stats["jacobian_reg"][0], reg.detach()) and stats["jacobian_reg"][1] == 1
    assert torch.equal(carry["z_H"], z_H)


def test_a_step_draws_a_probe_per_solve_then_exploration():
    h, b = head(halt_max_steps=8, halt_exploration_prob=0.5), batch(1)
    with torch.no_grad():
        h.model.inner.q_head.bias.fill_(5.0)                      # q_halt > 0: a row halts unless exploration holds it
    torch.manual_seed(5)
    carry, _, _ = h(h.initial_carry(b), b)
    after = torch.get_rng_state()
    torch.manual_seed(5)                                          # the same draws, replayed
    for _ in range(h.model.config.H_cycles):                     # the warm-up cycles' probes too
        torch.randn(4, 11, 32)
    minimum = (torch.rand(4) < 0.5) * torch.randint(2, 9, (4,), dtype=torch.int32)
    assert torch.equal(torch.get_rng_state(), after)
    assert carry["halted"].tolist() == (minimum <= 1).tolist()   # held back only by exploration


def test_the_optimizers_are_sign_sgd_on_the_puzzle_embedding_then_adamw_on_every_weight():
    h, b = head(), batch(1)
    (sign, sign_lr), (adamw, lr) = h.optimizers(lr=1e-4, weight_decay=1.0, betas=[0.9, 0.95], eps=8e-8,
                                                puzzle_emb_lr=1e-2, puzzle_emb_weight_decay=1.0)
    assert type(sign) is SparseSignSGD and sign_lr == 1e-2 and sign.param_groups[0]["weight_decay"] == 1.0
    assert type(adamw) is torch.optim.AdamW and lr == 1e-4
    group = adamw.param_groups[0]
    assert [id(p) for p in group["params"]] == [id(p) for p in h.model.parameters()]
    assert (group["betas"], group["eps"], group["weight_decay"], group["fused"]) == ((0.9, 0.95), 8e-8, 1.0, False)
    _, loss, _ = h(h.initial_carry(b), b)
    loss.backward()
    assert len(sign.pending) == 1                                 # the double backward accumulates no gradient


def test_evaluation_decodes_restart_0_of_halt_max_steps_segments_and_draws_nothing():
    model = head(seq_len=81, vocab_size=11, halt_max_steps=2).model.eval()
    pool = sudoku_pool([[0, 1], [5], [], [7, 8, 9]])
    task = make_task("sudoku", pool)
    state = torch.get_rng_state()
    values = AttractorHead.evaluate(model, pool, "sudoku", 4)
    assert torch.equal(torch.get_rng_state(), state)              # the training generator is not touched
    decoded = AttractorHead.decode(model, task, 4)
    np.testing.assert_array_equal(decoded, run_attractor(model, task, R=1, segments=2, batch=4)["attractor/model/raw"])
    assert not np.array_equal(decoded, run_attractor(model, task, R=1, segments=3, batch=4)["attractor/model/raw"])
    want = task.check("raw", np.arange(4), decoded).astype(np.float64)
    assert values.dtype == np.float64 and values.tolist() == want.tolist()
    assert AttractorHead.evaluate(model, pool.take(slice(0, 0)), "sudoku", 4).shape == (0,)
