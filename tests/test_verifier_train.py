"""The verifier's training (ics/verifier/train.py, VerifierHead; the loop is ics/train.py), and the pick among seeds
(ics/verifier/pick.py, pick_verifier)."""
import json
import re
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fakes import TINY_RECIPE, TINY_VERIFIER, sudoku_dataset, tiny_trm_checkpoint
from ics.checkpoint import save_checkpoint, sha256
from ics.cli import main
from ics.config import merge_config
from ics.data import Pool
from ics.train import train
from ics.trm.model import TRMConfig, load_trm
from ics.trm.roll import roll
from ics.trm.train import TRMHead
from ics.verifier.data import build_verifier_data
from ics.verifier.pick import pick_verifier
from ics.verifier.train import VerifierHead, auc


def head(T=3) -> VerifierHead:
    torch.manual_seed(0)
    cfg = TRMConfig(seq_len=9, vocab_size=5, puzzle_emb_ndim=32, puzzle_emb_len=2, H_cycles=2, L_cycles=2, L_layers=1,
                    hidden_size=32, num_heads=2, expansion=2.0, forward_dtype="float32", batch_size=4,
                    halt_max_steps=T)
    h = VerifierHead(cfg).train()
    with torch.no_grad():
        h.model.inner.q_head.weight.normal_()                  # a q_halt that depends on the input
        h.model.inner.puzzle_emb.weights.normal_()
    return h


def batch(seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"inputs": torch.randint(1, 5, (4, 9), generator=g, dtype=torch.int32),
            "labels": torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros(4, dtype=torch.int32)}


def test_an_update_runs_T_segments_from_a_cold_start_each_loss_the_bce_over_T():
    h, b = head(), batch()
    assert h.steps == 3
    carry, losses, segments = h.initial_carry(b), [], []
    for _ in range(4):                                         # an update of 3 segments, then the next update's first
        carry, loss, stats = h(carry, b)
        losses.append(loss)
        segments.append(int(carry["segment"]))
        assert stats["bce"][1] == 4
    assert segments == [1, 2, 0, 1]
    z_H, z_L = h.model.initial_state(4)
    for t in range(3):
        z_H, z_L, _, q = h.model.segment(z_H, z_L, b["inputs"], b["puzzle_identifiers"])
        expected = F.binary_cross_entropy_with_logits(q, b["labels"].float(), reduction="sum") / 3
        torch.testing.assert_close(losses[t], expected, rtol=0, atol=0)
    torch.testing.assert_close(losses[3], losses[0], rtol=0, atol=0)    # the next update starts cold again


def test_the_head_trains_the_weights_through_q_halt_only():
    h, b = head(T=2), batch()
    carry = h.initial_carry(b)
    for _ in range(2):
        carry, loss, _ = h(carry, b)
        loss.backward()
    inner = h.model.inner
    assert inner.lm_head.weight.grad is None and inner.puzzle_emb.local_weights.grad is None
    assert inner.q_head.weight.grad.abs().sum() > 0 and inner.L_level.layers[0].mlp.gate_up_proj.weight.grad is not None
    (opt, lr), = h.optimizers(lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95])
    assert type(opt) is torch.optim.AdamW and lr == 1e-4
    assert opt.defaults["betas"] == (0.9, 0.95) and opt.defaults["weight_decay"] == 0.1
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in h.model.parameters()]


def test_evaluation_scores_q_halt_after_T_segments_and_the_metric_is_the_auc():
    h = head().eval()
    rows = batch()["inputs"].numpy()
    pool = Pool(rows, np.array([1, 0, 1, 0]), np.arange(4), {"verifier": {"task": "maze"}})     # the verifier's data
    values = h.evaluate(h.model, pool, "maze", 2)
    np.testing.assert_array_equal(values, roll(h.model, rows, 3, 2, topk=1).q)
    assert h.metric(pool, values) == auc(values, pool.labels)


def test_auc_is_the_chance_a_valid_candidate_scores_above_an_invalid_one():
    rng = np.random.default_rng(0)
    scores, targets = rng.integers(0, 4, 40).astype(float), rng.integers(0, 2, 40)    # many ties
    pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for (a, s), (b, u) in product(zip(scores, targets), repeat=2)
             if s == 1 and u == 0]
    assert auc(scores, targets) == pytest.approx(np.mean(pairs))
    assert auc([0.1, 0.9], [0, 1]) == 1.0 and auc([0.9, 0.1], [0, 1]) == 0.0 and auc([3, 3], [0, 1]) == 0.5
    assert np.isnan(auc([0.1, 0.2], [1, 1]))


def verifier_data(tmp_path):
    """The verifier data of a tiny random solver (3 segments) on sudoku_dataset: the labels valid, the decodes not. The
    solver's puzzle embedding is random, unlike a freshly built TRM's (zero)."""
    solver = tiny_trm_checkpoint(tmp_path / "solver", halt_max_steps=3)
    model = load_trm(solver)
    model.inner.puzzle_emb.weights.normal_()
    save_checkpoint(model.state_dict(), model.config, solver)
    build_verifier_data("sudoku", sudoku_dataset(tmp_path / "data"), [solver], tmp_path / "vdata", hypotheses=2)
    return solver, tmp_path / "vdata"


def test_the_verifier_trains_from_its_solver_and_is_selected_by_val_auc(tmp_path):
    solver, data = verifier_data(tmp_path)
    result = train("verifier", "sudoku", data, tmp_path / "run", TINY_VERIFIER, device="cpu", init=solver)
    assert result["step"] == 4 and set(result) == {"step", "auc", "best"}
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "auc" in r] == [2, 4] and all("bce" in r for r in log if "lr" in r)
    assert [r["lr"] for r in log if "lr" in r] == [1e-4] * 4                     # constant: no warmup, no decay
    saved = json.loads((tmp_path / "run" / "last" / "config.json").read_text())
    expected = json.loads((solver / "config.json").read_text())["model"]
    assert saved["model"] == {**expected, "halt_max_steps": 2, "batch_size": 4}      # the solver's, at depth T = 2
    provenance = saved["provenance"]
    assert provenance["eval_split"] == "val" and not provenance["ema"] and provenance["init_sha256"] == sha256(
        solver / "model.safetensors")
    assert provenance["deterministic"] and provenance["recipe"]["train"]["deterministic"]       # the recipe's mode
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=True)
    assert state["ema"] == {} and torch.equal(load_trm(tmp_path / "run" / "last").inner.q_head.weight,
                                              state["model"]["inner.q_head.weight"])     # no EMA: the live weights
    kept = load_trm(solver).inner.puzzle_emb.weights
    assert kept.abs().sum() > 0 and torch.equal(load_trm(tmp_path / "run" / "last").inner.puzzle_emb.weights,
                                                kept)                                   # kept the solver's
    best = json.loads((tmp_path / "run" / "best" / "config.json").read_text())["provenance"]
    assert {"step": best["step"], "auc": best["auc"]} == result["best"]          # held out: best/ beside last/


def test_the_verifier_needs_its_solver(tmp_path):
    _, data = verifier_data(tmp_path)
    with pytest.raises(ValueError, match=r"verifier fine-tunes a checkpoint: pass init \(--init\)"):
        train("verifier", "sudoku", data, tmp_path / "run", TINY_VERIFIER, device="cpu")
    maze = tiny_trm_checkpoint(tmp_path / "maze", seq_len=900, vocab_size=6)
    error = "the checkpoint was built for another dataset (checkpoint vs data: seq_len 900 vs 81, vocab_size 6 vs 11)"
    with pytest.raises(ValueError, match=re.escape(f"{maze}: {error}")):
        train("verifier", "sudoku", data, tmp_path / "run", TINY_VERIFIER, device="cpu", init=maze)
    assert not (tmp_path / "run").exists()


def test_the_verifier_needs_its_data(tmp_path):
    solver = tiny_trm_checkpoint(tmp_path / "solver")
    data = sudoku_dataset(tmp_path / "data")                    # the solver's own data, a label per cell
    with pytest.raises(ValueError, match=r"the data of python -m ics verifier-data, whose dataset\.json holds a "
                                         r"\"verifier\" summary; this data has none"):
        train("verifier", "sudoku", data, tmp_path / "run", TINY_VERIFIER, device="cpu", init=solver)
    assert not (tmp_path / "run").exists()                      # refused before the first update


def test_the_verifier_trains_on_the_task_its_data_was_built_for(tmp_path):
    solver, data = verifier_data(tmp_path)                      # Sudoku's
    with pytest.raises(ValueError, match=r"^this is the verifier data of sudoku \(python -m ics verifier-data --task "
                                         r"sudoku\): train it with --task sudoku$"):
        train("verifier", "maze", data, tmp_path / "run", TINY_VERIFIER, device="cpu", init=solver)
    assert not (tmp_path / "run").exists()                      # refused before the first update


def test_a_solver_does_not_train_on_verifier_data(tmp_path, monkeypatch):
    _, data = verifier_data(tmp_path)
    monkeypatch.setattr(TRMHead, "forward", lambda *args: pytest.fail("a training step ran"))
    with pytest.raises(ValueError, match=r"^this is verifier data \(python -m ics verifier-data\): train it with "
                                         r"--model verifier$"):
        train("trm", "sudoku", data, tmp_path / "run", TINY_RECIPE, device="cpu")
    assert not (tmp_path / "run").exists()


def test_an_eval_pool_of_one_class_is_refused(tmp_path):
    solver, data = verifier_data(tmp_path)
    one = merge_config(TINY_VERIFIER, {"train": {"eval_boards": 1}})        # the val split's first candidate: valid
    with pytest.raises(ValueError, match=r"the eval pool holds 1 valid and 0 invalid candidates; its AUC needs both: "
                                         r"raise train\.eval_boards"):
        train("verifier", "sudoku", data, tmp_path / "run", one, device="cpu", init=solver)
    assert not (tmp_path / "run" / "last").exists()             # no evaluation was saved


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Verifier runs of one solver, data and recipe, seeds 0, 1 and 2 (v0, v1, v2), trained once for the module."""
    root = tmp_path_factory.mktemp("seeds")
    solver, data = verifier_data(root)
    for s in range(3):
        train("verifier", "sudoku", data, root / f"v{s}", merge_config(TINY_VERIFIER, {"train": {"seed": s}}),
              device="cpu", init=solver)
    return root


@pytest.fixture
def runs(trained, tmp_path) -> list:
    """Copies of the module's three runs under tmp_path, for a test to edit."""
    return [Path(shutil.copytree(trained / f"v{s}", tmp_path / f"v{s}")) for s in range(3)]


def best(run) -> dict:
    return json.loads((run / "best" / "config.json").read_text())


def edit(run, change) -> None:
    """change(meta) on the run's best/config.json, {"model": ..., "provenance": ...}, in place."""
    meta = best(run)
    change(meta)
    (run / "best" / "config.json").write_text(json.dumps(meta, indent=1))


def test_pick_verifier_keeps_the_run_with_the_highest_val_auc(runs, tmp_path, capsys):
    aucs = (0.75, 0.875, 0.5)
    for run, value in zip(runs, aucs):
        edit(run, lambda meta, value=value: meta["provenance"].update(auc=value))
    out = tmp_path / "picked"
    main(["pick-verifier", *map(str, runs), "--out", str(out)])
    meta = json.loads((out / "config.json").read_text())
    picked = meta["provenance"].pop("picked")
    assert meta == best(runs[1])                                         # the chosen best/, its provenance and all
    assert (out / "model.safetensors").read_bytes() == (runs[1] / "best" / "model.safetensors").read_bytes()
    load_trm(out)                                                       # strictly
    steps = [best(run)["provenance"]["step"] for run in runs]
    assert picked == {"chosen": 1, "candidates": [{"run": f"v{s}", "seed": s, "step": steps[s], "auc": aucs[s],
                                                   "sha256": sha256(run / "best" / "model.safetensors")}
                                                  for s, run in enumerate(runs)]}
    width = len(str(runs[0]))
    assert capsys.readouterr().out.splitlines() == [
        f"{'run':{width}}  seed    step  val AUC",
        *(f"{run}  {s:4d}  {steps[s]:6d}  {aucs[s]:.6f}{'  picked' if s == 1 else ''}" for s, run in enumerate(runs)),
        f"wrote {out}"]


def test_a_tie_goes_to_the_lower_seed_then_to_the_earlier_run(runs, tmp_path):
    for run in runs:
        edit(run, lambda meta: meta["provenance"].update(auc=0.75))
    assert pick_verifier([runs[2], runs[0], runs[1]], tmp_path / "a")["chosen"] == 1           # seed 0
    twin = Path(shutil.copytree(runs[0], tmp_path / "twin"))                                  # seed 0 again
    assert pick_verifier([runs[1], twin, runs[0]], tmp_path / "b")["chosen"] == 1             # the earlier of them


def test_the_runs_may_differ_in_the_seed_and_in_the_number_of_ranks(runs, tmp_path):
    edit(runs[1], lambda meta: (meta["model"].update(batch_size=2), meta["provenance"].update(world=2)))
    assert [c["seed"] for c in pick_verifier(runs, tmp_path / "out")["candidates"]] == [0, 1, 2]


@pytest.mark.parametrize("change, differ", [
    (lambda meta: meta["provenance"].update(task="maze"), "task"),
    (lambda meta: meta["provenance"]["train_sha256"].update(labels="0" * 64), "train_sha256.labels"),
    (lambda meta: meta["provenance"].update(init_sha256="0" * 64), "init_sha256"),
    (lambda meta: meta["provenance"].update(eval_boards=1), "eval_boards"),
    (lambda meta: meta["provenance"]["recipe"]["optim"].update(lr=1.0), "optim.lr"),
    (lambda meta: meta["provenance"]["recipe"]["train"].update(deterministic=False), "train.deterministic"),
    (lambda meta: meta["model"].update(halt_max_steps=3), "model.halt_max_steps"),
], ids=["task", "train-data", "init", "eval-pool", "optim", "train", "model"])
def test_runs_that_differ_in_more_than_the_seed_are_refused(runs, tmp_path, change, differ):
    edit(runs[2], change)
    error = f"{runs[2]} and {runs[0]} differ in {differ}: pick among runs that differ in train.seed alone"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        pick_verifier(runs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_run_whose_best_was_not_chosen_on_val_is_refused(runs, tmp_path):
    edit(runs[1], lambda meta: meta["provenance"].update(eval_split="test"))
    error = (f"{runs[1]}: its best/ was chosen on the test split; pick-verifier picks by the AUC on val, boards held "
             f"out from the train split")
    for given in ([runs[1]], runs):                                     # alone, or before any comparison
        with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
            pick_verifier(given, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_run_given_twice_or_without_a_verifiers_best_is_refused(runs, tmp_path, capsys):
    again = tmp_path / "v1" / ".." / "v0"                              # runs[0] again
    with pytest.raises(ValueError, match=f"^{re.escape(str(again))} is given twice$"):
        pick_verifier([runs[0], runs[1], again], tmp_path / "out")
    shutil.rmtree(runs[2] / "best")                                     # its last/ alone
    for other in (runs[2], tiny_trm_checkpoint(tmp_path / "ckpt")):    # a release checkpoint is no run
        error = f"{other}: no best/ checkpoint of a verifier's training run (python -m ics train --model verifier)"
        with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
            pick_verifier([runs[0], other], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["pick-verifier", str(runs[0]), "--out", str(tmp_path / "taken")])
    assert "--out " + str(tmp_path / "taken") + " is not empty (config.json)" in capsys.readouterr().err
