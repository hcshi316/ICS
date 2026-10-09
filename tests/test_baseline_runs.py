import json
import re

import pytest
import torch

import ics.train
from conftest import DATA_DIRS
from fakes import TINY_ATTRACTOR, TINY_EQR, TINY_GRAM, assert_identical, interrupted, sudoku_dataset
from ics.config import merge_config
from ics.train import recipe, train
from ics_baselines.attractor.model import load_attractor
from ics_baselines.eqr.model import load_eqr
from ics_baselines.gram.model import load_gram

# per baseline: its tiny recipe, its checkpoint loader, and the losses every training line of its log holds (each > 0)
MODELS = {"eqr": (TINY_EQR, load_eqr, ("lm_loss", "q_halt_loss")),
          "gram": (TINY_GRAM, load_gram, ("lm_loss", "kl", "q_halt_loss", "v_loss")),
          "attractor": (TINY_ATTRACTOR, load_attractor, ("lm_loss", "q_halt_loss", "jacobian_reg"))}
STATE = ("step", "position", "model", "optimizers", "ema", "ranks")


def run(tmp_path, model, name="run", **over):
    data = tmp_path / "data"
    if not data.exists():
        sudoku_dataset(data)
    return train(model, "sudoku", data, tmp_path / name, merge_config(MODELS[model][0], over), device="cpu")


@pytest.mark.parametrize("model", MODELS)
def test_a_run_keeps_the_release_checkpoint_of_its_ema_weights(tmp_path, model):
    result = run(tmp_path, model)
    assert result["step"] == 6 and 0.0 <= result["accuracy"] <= 1.0
    out = tmp_path / "run"
    state = torch.load(out / "state.pt", weights_only=True)
    loaded = MODELS[model][1](out / "last")
    assert (loaded.config.seq_len, loaded.config.vocab_size) == (81, 11)
    settings = recipe(model, "sudoku", MODELS[model][0])["model"]
    assert all(getattr(loaded.config, k) == v for k, v in settings.items())   # the recipe's; halt_max_steps 2 too
    for name, value in state["ema"].items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    provenance = json.loads((out / "last" / "config.json").read_text())["provenance"]
    assert provenance["model"] == model and provenance["step"] == 6 and provenance["ema"]
    assert sorted(p.name for p in out.iterdir()) == ["last", "log.jsonl", "state.pt"]       # no best/: scored on test
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    training = [r for r in lines if "lr" in r]
    assert len(training) == 3 and all("exact" in r and all(r[k] > 0 for k in MODELS[model][2]) for r in training)


@pytest.mark.parametrize("model, over", [
    ("eqr", {"accumulate": 1}), ("eqr", {"accumulate": 2}), ("gram", {"accumulate": 1}), ("gram", {"accumulate": 2}),
    ("attractor", {"micro_batch": 4}), ("attractor", {"micro_batch": 2}),
], ids=["eqr-1", "eqr-2", "gram-1", "gram-2", "attractor-1", "attractor-2"])
def test_an_interrupted_run_resumes_exactly(tmp_path, monkeypatch, model, over):
    # saved at update 3; the run dies in update 4 (after its backward, before its optimizers step), and the rerun
    # resumes after update 3, in one or two micro-batches per update. EqR: the rows continue at 4 and restart, drawing
    # fresh latents, at 5. GRAM: update 3 is a cycle's first step, and update 4 ends the cycle, its LPRM reading update
    # 3's buffered z0. The Attractor: the probes and the exploration draws continue from the saved generator.
    run(tmp_path, model, "whole", train=over)
    interrupted(lambda path, name, **kw: run(path, model, name, **kw), tmp_path, monkeypatch, "resumed", 4, train=over)
    run(tmp_path, model, "resumed", train=over)
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("whole", "resumed"))
    assert_identical(a, b)
    assert (tmp_path / "whole" / "last" / "model.safetensors").read_bytes() == \
           (tmp_path / "resumed" / "last" / "model.safetensors").read_bytes()


@pytest.mark.parametrize("model", MODELS)
def test_evaluating_draws_nothing_from_the_training_generator(tmp_path, model):
    run(tmp_path, model, "often", train={"eval_every": 1})
    run(tmp_path, model, "once", train={"eval_every": 10 ** 9})              # evaluated after the last update only
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("often", "once"))
    assert_identical({k: a[k] for k in STATE}, {k: b[k] for k in STATE})


def test_micro_batch_sets_the_micro_batches_as_accumulate_would(tmp_path):
    run(tmp_path, "attractor", "micro", train={"micro_batch": 2})
    run(tmp_path, "attractor", "accumulate", train={"micro_batch": None, "accumulate": 2})
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("micro", "accumulate"))
    assert_identical({k: a[k] for k in STATE}, {k: b[k] for k in STATE})
    carries = a["ranks"][0]["carries"]
    assert len(carries) == 2 and all(c["inputs"].shape == (2, 81) for c in carries)
    assert json.loads((tmp_path / "micro" / "last" / "config.json").read_text())["model"]["batch_size"] == 2


@pytest.mark.parametrize("world, micro_batch, accumulate, error", [
    (3, 2, 1, "global_batch 4 is not a multiple of ranks x micro_batch = 3 x 2"),
    (1, 3, 1, "global_batch 4 is not a multiple of ranks x micro_batch = 1 x 3"),
    (1, 2, 2, "set train.accumulate or train.micro_batch, not both"),
], ids=["ranks", "rows", "both"])
def test_a_micro_batch_the_ranks_cannot_share_or_beside_accumulate_is_refused(tmp_path, monkeypatch, world,
                                                                               micro_batch, accumulate, error):
    monkeypatch.setattr(ics.train, "_distributed", lambda device: (0, world, torch.device("cpu")))
    with pytest.raises(ValueError, match=re.escape(error)):
        run(tmp_path, "attractor", train={"micro_batch": micro_batch, "accumulate": accumulate})


@pytest.mark.data
@pytest.mark.parametrize("task", list(DATA_DIRS))
@pytest.mark.parametrize("model", MODELS)
def test_a_baseline_trains_and_evaluates_on_each_task(tmp_path, dataset, model, task):
    # two updates of 8 rows (the Attractor's: two solver calls of 4 rows each)
    over = merge_config(MODELS[model][0], {"train": {"global_batch": 8, "epochs": None, "updates": 2, "eval_every": 2}})
    result = train(model, task, dataset(task), tmp_path / "run", over, device="cpu")
    assert result["step"] == 2 and 0.0 <= result["accuracy"] <= 1.0
    loaded = MODELS[model][1](tmp_path / "run" / "last")
    assert all(getattr(loaded.config, k) == v for k, v in recipe(model, task, over)["model"].items())   # mlp_t too
