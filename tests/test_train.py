import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ics.data
import ics.train
from conftest import DATA_DIRS
from fakes import TINY_RECIPE, Stop, assert_identical, interrupted, stop, sudoku_dataset
from ics.checkpoint import sha256
from ics.config import merge_config
from ics.data import IGNORE, load_train, train_batches
from ics.registry import HEADS, TASKS
from ics.train import EMA, TrainConfig, lr_at, recipe, train
from ics.trm.model import load_trm
from ics.trm.train import TRMHead


def run(tmp_path, name="run", **over):
    data = tmp_path / "data"
    if not data.exists():
        sudoku_dataset(data)
    return train("trm", "sudoku", data, tmp_path / name, merge_config(TINY_RECIPE, over), device="cpu")


def logged(out) -> list[int]:
    """The step of every record of out/log.jsonl."""
    return [json.loads(line)["step"] for line in (out / "log.jsonl").read_text().splitlines()]


def test_the_first_update_runs_at_base_over_warmup():
    assert lr_at(1, 1e-4, 2000, 65104, 1.0) == 1e-4 / 2000
    assert lr_at(1999, 1e-4, 2000, 65104, 1.0) == 1e-4 * 1999 / 2000
    assert lr_at(2000, 1e-4, 2000, 65104, 1.0) == lr_at(65104, 1e-4, 2000, 65104, 1.0) == 1e-4
    assert lr_at(5, 1.0, 0, 10, 0.0) == pytest.approx(0.5) and lr_at(10, 1.0, 0, 10, 0.1) == pytest.approx(0.1)


def test_ema_averages_the_parameters_and_keeps_the_buffers():
    m = torch.nn.Linear(2, 1)
    m.register_buffer("b", torch.zeros(1))
    ema, p0 = EMA(m, 0.9), m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
        m.b.fill_(3.0)
    ema.update(m)
    torch.testing.assert_close(ema.shadow["weight"], 0.1 * (p0 + 1) + 0.9 * p0)
    averaged = ema.copy(m)
    assert set(ema.shadow) == {"weight", "bias"} and averaged.b.item() == 3.0
    torch.testing.assert_close(averaged.weight, ema.shadow["weight"])


def test_the_recipes():
    # the settings a run takes: the recipe's on top of the defaults of the head's config class and of TrainConfig
    model = lambda name, task: HEADS[name].config_class(seq_len=9, vocab_size=5, **recipe(name, task)["model"])
    train_config = lambda name, task, over=None: TrainConfig(**recipe(name, task, over)["train"])
    assert model("trm", "sudoku").mlp_t and model("trm", "sudoku").pos_encodings == "none"
    assert train_config("trm", "sudoku").eval_boards == 12288 and model("trm", "maze").L_cycles == 4
    assert train_config("trm", "tapa", {"train": {"seed": 3}}).seed == 3
    with pytest.raises(ValueError, match="unknown setting train.sed"):
        recipe("trm", "sudoku", {"train": {"sed": 1}})
    assert train_config("verifier", "maze").deterministic and not train_config("trm", "maze").deterministic
    assert train_config("trm", "maze", {"train": {"deterministic": True}}).deterministic          # TRM's to set


@pytest.mark.parametrize("model", HEADS)
def test_every_recipe_builds_its_configs_on_every_task(model):
    for task in TASKS:
        cfg = recipe(model, task)
        TrainConfig(**cfg["train"])
        HEADS[model].config_class(**{"seq_len": 9, "vocab_size": 5, **cfg["model"]})


def test_an_override_names_a_config_field_the_data_does_not_set():
    assert recipe("trm", "sudoku", {"train": {"micro_batch": 4}})["train"]["micro_batch"] == 4     # unlisted, a field
    assert recipe("attractor", "maze", {"model": {"deq_tol": 0.01}})["model"]["deq_tol"] == 0.01
    for over, name in (({"model": {"seq_len": 81}}, "model.seq_len"), ({"train": {"sed": 1}}, "train.sed"),
                       ({"optim": {"eps": 1e-8}}, "optim.eps"), ({"model": {"L_layers": {"a": 1}}}, "model.L_layers")):
        with pytest.raises(ValueError, match=re.escape(name)):
            recipe("trm", "sudoku", over)


@pytest.mark.data
def test_the_recipes_and_their_update_counts(dataset):
    counts = {}
    for task in DATA_DIRS:
        meta = json.loads((dataset(task) / "train" / "dataset.json").read_text())
        c = TrainConfig(**recipe("trm", task)["train"])
        counts[task] = int(c.epochs * meta["total_groups"] * meta["mean_puzzle_examples"] / c.global_batch)
    assert counts == {"sudoku": 65104, "maze": 65104, "lightup": 130206, "nurikabe": 195396, "tapa": 194979,
                      "heyawake": 195174}


def test_a_run_writes_release_checkpoints_the_state_and_a_log(tmp_path):
    assert run(tmp_path)["step"] == 6
    out = tmp_path / "run"
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "accuracy" in r] == [3, 6] and [r["step"] for r in log if "lr" in r] == [2, 4, 6]
    state = torch.load(out / "state.pt", weights_only=True)
    assert state["step"] == 6 and state["position"] == [2, 2] and len(state["ranks"]) == 1
    model = load_trm(out / "last")
    for name, value in state["ema"].items():                     # the release checkpoint holds the EMA weights
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    provenance = json.loads((out / "last" / "config.json").read_text())["provenance"]
    assert provenance["step"] == 6 and provenance["ema"] and provenance["format"] == "train"
    data = {k: sha256(tmp_path / "data" / "train" / f"all__{k}.npy") for k in ("inputs", "labels")}
    assert provenance["train_sha256"] == state["run"]["train_sha256"] == data
    assert sorted(p.name for p in out.iterdir()) == ["last", "log.jsonl", "state.pt"]   # no best/: scored on test


@pytest.mark.parametrize("held_out", [False, True], ids=["trm", "held-out"])
def test_best_changes_only_on_a_strict_improvement_and_is_kept_for_a_held_out_metric(tmp_path, monkeypatch, held_out):
    scores = iter([0.5, 0.75, 0.75])
    monkeypatch.setattr(ics.train, "_evaluate", lambda *args: next(scores))
    monkeypatch.setattr(TRMHead, "metric_held_out", held_out)
    assert run(tmp_path, train={"eval_every": 2})["best"] == {"step": 4, "accuracy": 0.75}
    step = lambda d: json.loads((tmp_path / "run" / d / "config.json").read_text())["provenance"]["step"]
    assert step("last") == 6 and (step("best") == 4 if held_out else not (tmp_path / "run" / "best").exists())


def test_any_finite_metric_beats_the_initial_best(tmp_path, monkeypatch):
    scores = iter([-2.0, -3.0, -2.5])                           # below -1, as a negative loss would be
    monkeypatch.setattr(ics.train, "_evaluate", lambda *args: next(scores))
    monkeypatch.setattr(TRMHead, "metric_held_out", True)       # so the run keeps best/
    assert run(tmp_path, train={"eval_every": 2})["best"] == {"step": 2, "accuracy": -2.0}
    assert json.loads((tmp_path / "run" / "best" / "config.json").read_text())["provenance"]["step"] == 2
    for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines():          # strict JSON: no -Infinity
        json.loads(line, parse_constant=lambda name: pytest.fail(f"{name} in the log"))


@pytest.mark.parametrize("accumulate, eval_every", [(1, 3), (2, 3), (1, 4)])
def test_an_interrupted_run_resumes_exactly(tmp_path, monkeypatch, accumulate, eval_every):
    # halt_max_steps 2: every row halts at updates 2, 4 and 6, so updates 3 and 5 take new rows. eval_every 4 saves at
    # the end of block 1 (4 batches), so the rerun starts block 2.
    over = {"model": {"halt_max_steps": 2}, "train": {"accumulate": accumulate, "eval_every": eval_every}}
    run(tmp_path, "whole", **over)
    interrupted(run, tmp_path, monkeypatch, "resumed", eval_every + 1, **over)    # dies in the update after the save
    real, steps = ics.train.lr_at, []
    monkeypatch.setattr(ics.train, "lr_at", lambda step, *args: steps.append(step) or real(step, *args))
    run(tmp_path, "resumed", **over)
    assert list(dict.fromkeys(steps)) == list(range(eval_every + 1, 7))      # it resumed after the save
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("whole", "resumed"))
    assert_identical(a, b)
    assert all(c["halted"].all() for c in a["ranks"][0]["carries"])
    assert (tmp_path / "whole" / "last" / "model.safetensors").read_bytes() == \
           (tmp_path / "resumed" / "last" / "model.safetensors").read_bytes()


def test_deterministic_mode_is_set_before_the_device_and_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seen, real = [], ics.train._distributed
    monkeypatch.setattr(ics.train, "_distributed", lambda device: seen.append(
        (torch.are_deterministic_algorithms_enabled(), os.environ.get("CUBLAS_WORKSPACE_CONFIG"))) or real(device))
    provenance = lambda name: json.loads((tmp_path / name / "last" / "config.json").read_text())["provenance"]
    run(tmp_path, train={"deterministic": True})
    assert seen == [(True, ":4096:8")]                  # on before the device is chosen, and so before CUDA starts
    assert torch.are_deterministic_algorithms_enabled()                     # for the rest of the process
    p = provenance("run")
    assert p["deterministic"] and p["cublas_workspace_config"] == ":4096:8" and p["recipe"]["train"]["deterministic"]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")                  # the environment's setting stays
    run(tmp_path, "set", train={"deterministic": True})
    assert seen[-1] == (True, ":16:8") and provenance("set")["cublas_workspace_config"] == ":16:8"


def test_the_default_mode_is_left_off_and_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    run(tmp_path)
    assert not torch.are_deterministic_algorithms_enabled() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    p = json.loads((tmp_path / "run" / "last" / "config.json").read_text())["provenance"]
    assert p["deterministic"] is False and p["cublas_workspace_config"] is None
    assert p["recipe"]["train"]["deterministic"] is False


def test_two_deterministic_runs_are_bit_equal(tmp_path):
    for name in ("a", "b"):
        run(tmp_path, name, train={"deterministic": True})
    a, b = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("a", "b"))
    assert_identical(a, b)
    assert (tmp_path / "a" / "last" / "model.safetensors").read_bytes() == \
           (tmp_path / "b" / "last" / "model.safetensors").read_bytes()


def test_resuming_with_another_recipe_or_other_data_is_refused(tmp_path):
    run(tmp_path)
    for change in ({"seed": 1}, {"deterministic": True}):
        with pytest.raises(ValueError, match="belongs to another run"):
            run(tmp_path, train=change)
    other = sudoku_dataset(tmp_path / "other", seed=1)                # the same shapes, other boards
    with pytest.raises(ValueError, match="belongs to another run"):
        train("trm", "sudoku", other, tmp_path / "run", TINY_RECIPE, device="cpu")


def test_rerunning_a_finished_run_says_so_and_returns_its_result(tmp_path, capsys):
    first = run(tmp_path)
    log = (tmp_path / "run" / "log.jsonl").read_text()
    capsys.readouterr()
    assert run(tmp_path) == first and "already complete at step 6" in capsys.readouterr().out
    assert (tmp_path / "run" / "log.jsonl").read_text() == log


def test_the_log_follows_the_saves_and_restarts_with_the_run(tmp_path, monkeypatch):
    run(tmp_path, "whole")
    assert logged(tmp_path / "whole") == [2, 3, 4, 6, 6]               # log_every 2, eval_every 3, 6 updates
    interrupted(run, tmp_path, monkeypatch, "resumed", 5)              # logged up to update 4, saved at update 3
    assert logged(tmp_path / "resumed") == [2, 3, 4]
    with open(tmp_path / "resumed" / "log.jsonl", "a") as f:
        f.write('{"step": 5, "lr"')                                    # a record a crash cut short
    run(tmp_path, "resumed")                                           # the resume drops both
    assert logged(tmp_path / "resumed") == logged(tmp_path / "whole")
    (tmp_path / "whole" / "state.pt").unlink()
    run(tmp_path, "whole")                                             # a fresh start begins a new log
    assert logged(tmp_path / "whole") == [2, 3, 4, 6, 6]
    monkeypatch.setattr(ics.train, "_save", stop)
    with pytest.raises(Stop):
        run(tmp_path, "unsaved")                                       # dies saving the evaluation of update 3,
    assert logged(tmp_path / "unsaved") == [2]                         # which is logged only once it is saved


DEGENERATE = [("eval_boards", 0), ("eval_every", 0), ("log_every", -1), ("epochs_per_block", 0), ("epochs", 0),
              ("clip", 0.0), ("clip", -1.0), ("ema", 1.0), ("ema", -0.5)]


@pytest.mark.parametrize("setting, value", DEGENERATE)
def test_degenerate_settings_are_refused_before_the_model_is_built(tmp_path, monkeypatch, setting, value):
    monkeypatch.setattr(TRMHead, "__init__", lambda *args: pytest.fail("the model was built"))
    with pytest.raises(ValueError, match=rf"train\.{setting}\b"):
        run(tmp_path, train={setting: value})
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("setting, value", [(s, v) for s, v in DEGENERATE if s != "epochs"] + [("micro_batch", 2)])
def test_train_config_refuses_its_degenerate_settings_on_its_own(setting, value):
    # epochs 0 is the loop's to refuse, once it counts the updates; micro_batch stands beside accumulate 2
    with pytest.raises(ValueError, match=rf"train\.{setting}\b"):
        TrainConfig(global_batch=4, updates=1, accumulate=2, **{setting: value})


@pytest.mark.parametrize("task, error", [("lightup", "Light-Up needs all__dims.npy"),
                                         ("nurikabe", r"PPB\('nurikabe'\) needs")])
def test_a_task_the_data_cannot_serve_is_refused_before_the_first_update(tmp_path, monkeypatch, task, error):
    monkeypatch.setattr(TRMHead, "forward", lambda *args: pytest.fail("a training step ran"))
    with pytest.raises(ValueError, match=error):                   # Sudoku data: no all__dims.npy, no vocab.json
        train("trm", task, sudoku_dataset(tmp_path / "data"), tmp_path / "run", TINY_RECIPE, device="cpu")
    assert not (tmp_path / "run").exists()


@dataclass
class BareConfig:                       # like EqR's config: neither batch_size nor num_puzzle_identifiers
    seq_len: int
    vocab_size: int


class Bare(torch.nn.Module):
    def __init__(self, config: BareConfig):
        super().__init__()
        self.config = config
        self.table = torch.nn.Parameter(torch.zeros(config.seq_len, config.vocab_size))


class BareHead(torch.nn.Module):
    """A head whose model answers every board with one learned table of logits; its metric is the share of boards
    answered exactly."""
    config_class = BareConfig
    steps = 1
    metric_name = "exact"

    def __init__(self, config: BareConfig):
        super().__init__()
        self.model = Bare(config)

    def initial_carry(self, batch):
        return {"rows": torch.zeros(len(batch["inputs"]))}

    def forward(self, carry, batch):
        labels = batch["labels"].long()
        keep = labels != IGNORE
        loss = F.cross_entropy(self.model.table.expand(len(labels), -1, -1)[keep], labels[keep], reduction="sum")
        return carry, loss, {"loss": (loss.detach(), len(labels))}

    def optimizers(self, lr):
        return [(torch.optim.SGD(self.model.parameters(), lr=lr), lr)]

    @staticmethod
    def evaluate(model, pool, task, batch):
        return (model.table.argmax(-1).numpy() == pool.labels).all(1).astype(float)

    @staticmethod
    def metric(pool, values):
        return float(values.mean())


def test_a_head_whose_config_has_no_batch_size_or_puzzle_identifiers_trains(tmp_path, monkeypatch):
    (tmp_path / "bare.yaml").write_text("model: {}\noptim: {lr: 0.1}\n"
                                        "train: {global_batch: 8, epochs: 1, epochs_per_block: 1, log_every: 1}\n")
    monkeypatch.setattr(ics.train, "CONFIG_DIR", tmp_path)
    monkeypatch.setitem(ics.train.HEADS, "bare", BareHead)
    data = sudoku_dataset(tmp_path / "data")
    out = train("bare", "sudoku", data, tmp_path / "run", device="cpu")
    assert out == {"step": 1, "exact": 1.0, "best": {"step": 1, "exact": 1.0}}       # the head's metric, by its name
    saved = json.loads((tmp_path / "run" / "last" / "config.json").read_text())
    assert saved["model"] == {"seq_len": 81, "vocab_size": 11}
    assert saved["provenance"]["exact"] == 1.0 and "accuracy" not in saved["provenance"]
    assert torch.load(tmp_path / "run" / "state.pt", weights_only=True)["model"]["table"].any()     # it trained
    assert train("bare", "sudoku", data, tmp_path / "run", device="cpu") == out      # the rerun resumes its metric


def test_a_head_runs_its_steps_on_every_micro_batch_and_then_the_optimizer_steps_once(tmp_path, monkeypatch):
    forwards, updates = [], []                  # rows of every forward call; forward calls made at every opt.step

    class TwoSteps(BareHead):
        steps = 2

        def forward(self, carry, batch):
            forwards.append(len(batch["inputs"]))
            return super().forward(carry, batch)

        def optimizers(self, lr):
            [(opt, base)] = super().optimizers(lr)
            step = opt.step
            opt.step = lambda: updates.append(len(forwards)) or step()
            return [(opt, base)]

    (tmp_path / "bare.yaml").write_text("model: {}\noptim: {lr: 0.1}\n"
                                        "train: {global_batch: 8, epochs: 3, epochs_per_block: 1, accumulate: 2}\n")
    monkeypatch.setattr(ics.train, "CONFIG_DIR", tmp_path)
    monkeypatch.setitem(ics.train.HEADS, "bare", TwoSteps)
    assert train("bare", "sudoku", sudoku_dataset(tmp_path / "data"), tmp_path / "run", device="cpu")["step"] == 3
    assert forwards == [4] * 12 and updates == [4, 8, 12]       # per update: 2 micro-batches of 4 rows, 2 steps each


def test_a_cpu_device_object_under_torchrun_joins_gloo(monkeypatch):
    backends = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: pytest.fail("a GPU was selected"))
    monkeypatch.setattr(ics.train.dist, "init_process_group", backends.append)
    monkeypatch.setattr(ics.train.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(ics.train.dist, "get_world_size", lambda: 2)
    assert ics.train._distributed(torch.device("cpu")) == (1, 2, torch.device("cpu")) and backends == ["gloo"]


@pytest.mark.parametrize("world, accumulate", [(3, 1), (1, 3), (2, 4), (1, 0)])
def test_a_global_batch_the_ranks_and_micro_batches_cannot_share_is_refused(tmp_path, monkeypatch, world, accumulate):
    monkeypatch.setattr(ics.train, "_distributed", lambda device: (0, world, torch.device("cpu")))
    with pytest.raises(ValueError, match=r"global_batch 4 is not a multiple of ranks x accumulate"):
        run(tmp_path, train={"accumulate": accumulate})


def test_each_rank_runs_its_virtual_ranks_slices_in_order_each_with_its_own_carry(tmp_path, monkeypatch):
    seen, rows = [], ics.data.TrainSplit.rows
    monkeypatch.setattr(ics.data.TrainSplit, "rows", lambda self, r: seen.append(r.copy()) or rows(self, r))
    run(tmp_path, train={"accumulate": 2})
    stream = train_batches(load_train(tmp_path / "data").group_starts, 0, 2, 4)
    for k in range(6):                                           # 6 updates of 2 micro-batches of 2 rows
        np.testing.assert_array_equal(np.concatenate(seen[2 * k:2 * k + 2]), next(stream)[2])
    assert len(seen) == 12 and {len(r) for r in seen} == {2}
    carries = torch.load(tmp_path / "run" / "state.pt", weights_only=True)["ranks"][0]["carries"]
    assert len(carries) == 2 and all(c["inputs"].shape == (2, 81) for c in carries)
    assert json.loads((tmp_path / "run" / "last" / "config.json").read_text())["model"]["batch_size"] == 2


def test_eval_boards_beyond_the_test_split_evaluate_the_whole_split(tmp_path):
    run(tmp_path, train={"eval_boards": 100, "epochs": 1})
    assert json.loads((tmp_path / "run" / "last" / "config.json").read_text())["provenance"]["eval_boards"] == 6


def test_a_run_from_a_checkpoint_starts_from_its_weights_and_config(tmp_path):
    run(tmp_path, "first")
    frozen = {"optim": {"lr": 0.0, "puzzle_emb_lr": 0.0}, "train": {"ema": 0.0, "epochs": 1}}     # nothing moves
    init = tmp_path / "first" / "last"
    out = train("trm", "sudoku", tmp_path / "data", tmp_path / "second", merge_config(TINY_RECIPE, frozen),
                device="cpu", init=init)
    assert out["step"] == 2
    a, b = (json.loads((d / "config.json").read_text()) for d in (init, tmp_path / "second" / "last"))
    assert a["model"] == b["model"] and b["provenance"]["init_sha256"] == sha256(init / "model.safetensors")
    assert not b["provenance"]["ema"]
    second = load_trm(tmp_path / "second" / "last").state_dict()                # no EMA: the live weights, unmoved
    assert all(torch.equal(second[name], value) for name, value in load_trm(init).state_dict().items())
    with pytest.raises(ValueError, match="belongs to another run"):                    # the init is part of the run
        train("trm", "sudoku", tmp_path / "data", tmp_path / "second", merge_config(TINY_RECIPE, frozen), device="cpu")


@pytest.mark.parametrize("edit, error", [
    (lambda model: {k: v for k, v in model.items() if k != "mlp_t"}, "missing fields ['mlp_t'], unknown fields []"),
    (lambda model: {**model, "depth": 2}, "missing fields [], unknown fields ['depth']"),
], ids=["missing", "unknown"])
def test_an_init_whose_config_is_not_exactly_the_heads_is_refused(tmp_path, monkeypatch, edit, error):
    run(tmp_path, "first")
    path = tmp_path / "first" / "last" / "config.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, "model": edit(meta["model"])}))
    monkeypatch.setattr(TRMHead, "__init__", lambda *args: pytest.fail("the model was built"))
    with pytest.raises(ValueError, match=re.escape(error)):
        train("trm", "sudoku", tmp_path / "data", tmp_path / "second", TINY_RECIPE, device="cpu", init=path.parent)
    assert not (tmp_path / "second").exists()


def test_a_training_run_as_init_is_refused_by_name_before_the_model_is_built(tmp_path, monkeypatch):
    run(tmp_path, "first")
    first = tmp_path / "first"
    monkeypatch.setattr(TRMHead, "__init__", lambda *args: pytest.fail("the model was built"))
    error = f"{first} is a training run; pass a checkpoint, e.g. {first / 'last'}"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        train("trm", "sudoku", tmp_path / "data", tmp_path / "second", TINY_RECIPE, device="cpu", init=first)
    assert not (tmp_path / "second").exists()


def test_the_gradient_norm_is_clipped_when_set(tmp_path, monkeypatch):
    calls, real = [], torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda ps, norm: calls.append(norm) or real(ps, norm))
    run(tmp_path, "unclipped")
    assert calls == []
    run(tmp_path, "clipped", train={"clip": 0.5})
    assert calls == [0.5] * 6


def test_every_evaluation_is_kept_when_asked(tmp_path):
    run(tmp_path, train={"keep_every_eval": True})
    kept = tmp_path / "run" / "snapshots"
    assert sorted(p.name for p in kept.iterdir()) == ["step_3", "step_6"]
    assert json.loads((kept / "step_3" / "config.json").read_text())["provenance"]["step"] == 3
    assert (kept / "step_6" / "model.safetensors").read_bytes() == \
           (tmp_path / "run" / "last" / "model.safetensors").read_bytes()
    run(tmp_path, "plain")
    assert not (tmp_path / "plain" / "snapshots").exists()


@pytest.mark.parametrize("epochs, updates, error", [(None, 5, None), (3, 5, "exactly one"), (None, None, "exactly one"),
                                                    (None, 0, r"train\.updates 0 gives 0 updates")])
def test_a_run_lasts_its_epochs_or_its_updates(tmp_path, epochs, updates, error):
    over = {"train": {"epochs": epochs, "updates": updates}}
    if error:
        with pytest.raises(ValueError, match=error):
            run(tmp_path, **over)
    else:
        assert run(tmp_path, **over)["step"] == 5
