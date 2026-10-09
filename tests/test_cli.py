import fnmatch
import inspect
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
import torch

from fakes import (
    TINY_ATTRACTOR,
    TINY_EQR,
    TINY_GRAM,
    TINY_RECIPE,
    TINY_VERIFIER,
    lightup_source,
    maze_source,
    ppb_source,
    ppbench_row,
    sudoku_dataset,
    sudoku_source,
    tiny_trm_checkpoint,
)
from ics.builders import ppb
from ics.cli import main
from ics.config import merge_config
from ics.data import load_pool, load_train
from ics.registry import TASKS
from ics.tasks.ppb import PPB
from ics.train import train
from ics.trm.model import load_trm

ROOT = Path(__file__).resolve().parents[1]


def flags(recipe: dict) -> list[str]:
    return [f"--set={part}.{key}={value}" for part, settings in recipe.items() for key, value in settings.items()]


def files(d) -> dict:
    """Every file under d: its relative path -> its bytes."""
    return {p.relative_to(d): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("command, option", [("data", "--seed"), ("download", "--repo"), ("train", "--init CKPT"),
                                             ("verifier-data", "--solver"), ("eval", "--verifier CKPT")])
def test_help(command, option, capsys):
    with pytest.raises(SystemExit) as stop:
        main([command, "--help"])
    assert stop.value.code == 0 and option in capsys.readouterr().out


@pytest.mark.parametrize("task, options", [("sudoku", ["--seed", "1"]), ("maze", []), ("lightup", []), ("tapa", []),
                                           ("nurikabe", ["--seed", "1"])])
def test_data_from_the_command_line(tmp_path, capsys, fake_ppbench, task, options):
    sources = {"sudoku": sudoku_source, "maze": maze_source, "lightup": lightup_source,
               "tapa": lambda root: ppb_source(root, "tapa"), "nurikabe": lambda root: ppb_source(root, "nurikabe")}
    source = sources[task](tmp_path / "src")
    main(["data", "--task", task, "--out", str(tmp_path / "cli"), "--source", str(source), *options])
    assert capsys.readouterr().out == f"wrote {tmp_path / 'cli'}\n"
    TASKS[task].build(tmp_path / "direct", source=source, **({"seed": 1} if options else {}))
    assert files(tmp_path / "cli") == files(tmp_path / "direct")
    rows = {"sudoku": 6 * 1001, "maze": 2, "lightup": 24, "tapa": 30, "nurikabe": 30}
    assert len(load_train(tmp_path / "cli").inputs) == rows[task]


@pytest.mark.parametrize("task", ["maze", "lightup"])
def test_a_build_without_random_draws_takes_no_seed(tmp_path, capsys, task):
    with pytest.raises(SystemExit):
        main(["data", "--task", task, "--out", str(tmp_path), "--seed", "1"])
    assert f"python -m ics data: error: the {task} build takes no seed: " in capsys.readouterr().err


@pytest.mark.parametrize("task", ["tapa", "maze"])
def test_a_negative_seed_is_a_usage_error(tmp_path, capsys, task):
    with pytest.raises(SystemExit) as stop:
        main(["data", "--task", task, "--out", str(tmp_path / "out"), "--seed", "-1"])
    assert stop.value.code == 2 and "error: argument --seed: expected 0 or more, got -1" in capsys.readouterr().err


def test_data_refuses_a_non_empty_out_and_leaves_it_as_it_was(tmp_path, capsys):
    out = tmp_path / "out"
    (out / "results").mkdir(parents=True)
    (out / "results" / "r.json").write_text("{}")
    (out / "notes.txt").write_text("mine")
    (out / ".DS_Store").write_bytes(b"\0")
    before = files(out)
    with pytest.raises(SystemExit) as stop:
        main(["data", "--task", "maze", "--out", str(out), "--source", str(maze_source(tmp_path / "src"))])
    assert stop.value.code == 2 and files(out) == before
    assert (f"python -m ics data: error: --out {out} is not empty (notes.txt, results); choose another --out, or "
            "remove it if it holds an earlier build") in capsys.readouterr().err


def test_data_builds_into_an_out_holding_only_dotfiles_and_leaves_them(tmp_path):
    out, src = tmp_path / "out", maze_source(tmp_path / "src")
    (out / ".git").mkdir(parents=True)
    dotfiles = {".DS_Store": b"\0", "._train": b"\1", ".gitkeep": b"", ".git/HEAD": b"ref: refs/heads/main\n"}
    for name, data in dotfiles.items():
        (out / name).write_bytes(data)
    main(["data", "--task", "maze", "--out", str(out), "--source", str(src)])
    main(["data", "--task", "maze", "--out", str(tmp_path / "fresh"), "--source", str(src)])
    assert files(out) == {**{Path(name): data for name, data in dotfiles.items()}, **files(tmp_path / "fresh")}


def test_data_refuses_an_out_that_is_a_file(tmp_path, capsys):
    (tmp_path / "out").write_text("mine")
    with pytest.raises(SystemExit):
        main(["data", "--task", "maze", "--out", str(tmp_path / "out"), "--source", str(maze_source(tmp_path / "s"))])
    assert f"python -m ics data: error: --out {tmp_path / 'out'} is not a directory" in capsys.readouterr().err
    assert (tmp_path / "out").read_text() == "mine"


def test_data_refuses_maze_into_a_former_tapa_folder(tmp_path, capsys, fake_ppbench):
    out = tmp_path / "out"
    main(["data", "--task", "tapa", "--out", str(out), "--source", str(ppb_source(tmp_path / "ppb", "tapa"))])
    before = files(out)
    with pytest.raises(SystemExit):
        main(["data", "--task", "maze", "--out", str(out), "--source", str(maze_source(tmp_path / "maze"))])
    assert (f"python -m ics data: error: --out {out} is not empty (PPB_MANIFEST.json, identifiers.json, test, ...); "
            "choose another --out, or remove it if it holds an earlier build") in capsys.readouterr().err
    assert files(out) == before


def test_data_reports_a_missing_extra_in_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ppb.shutil, "which", lambda name: None)
    monkeypatch.delenv("NODE_BIN", raising=False)
    with pytest.raises(SystemExit) as stop:
        main(["data", "--task", "heyawake", "--out", str(tmp_path / "out")])
    assert stop.value.code == 1 and not (tmp_path / "out").exists()
    assert capsys.readouterr().err == f"python -m ics data: error: {ppb.NEEDS}: Node.js not found (node on PATH)\n"


def test_a_build_that_fails_keeps_its_traceback(tmp_path, fake_ppbench):
    src = ppb_source(tmp_path / "src", "tapa")
    with open(src / "golden_300.jsonl", "a") as f:       # a golden board whose text and row disagree in size
        f.write(json.dumps({**ppbench_row("tapa", "x", [". 1", ". ."], []), "height": 3}) + "\n")
    with pytest.raises(ValueError, match="tapa x: the board text is 2x2, the row 3x2"):
        main(["data", "--task", "tapa", "--out", str(tmp_path / "out"), "--source", str(src)])


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A local Hugging Face: hub.root / <repo id> holds a repo's files. snapshot_download, taking only arguments the
    real one takes, copies the files its allow_patterns matches into a snapshot folder of its own, read-only as the
    cache holds them, and returns that folder; for a repo it does not hold, it raises an OSError, as the hub's errors
    are. hub.calls records each call's (repo id, repo_type, allow_patterns)."""
    root, calls, real = tmp_path / "hub", [], inspect.signature(huggingface_hub.snapshot_download)

    def snapshot_download(*args, **kwargs):
        real.bind(*args, **kwargs)
        repo, pattern = args[0], kwargs["allow_patterns"]
        calls.append((repo, kwargs.get("repo_type"), pattern))
        if not (root / repo).is_dir():
            raise OSError("Repository Not Found")
        snapshot = tmp_path / "cache" / str(len(calls))
        for p in (root / repo).rglob("*"):
            name = p.relative_to(root / repo).as_posix()
            if p.is_file() and fnmatch.fnmatch(name, pattern):
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, snapshot / name)
                (snapshot / name).chmod(0o444)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return types.SimpleNamespace(root=root, calls=calls)


def test_download_fetches_a_checkpoint_or_a_dataset_into_its_default_out(tmp_path, monkeypatch, capsys, hub,
                                                                         fake_ppbench):
    tiny_trm_checkpoint(hub.root / "hcshi/ICS/trm/maze/seed0", 900, 6)
    tiny_trm_checkpoint(hub.root / "hcshi/ICS/trm/maze/seed1", 900, 6, seed=1)
    ppb.build_ppb(hub.root / "hcshi/ICS-data/ppb-tapa-ms26", "tapa", source=ppb_source(tmp_path / "src", "tapa"))
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    main(["download", "--model", "trm", "--task", "maze", "--seed", "0"])
    main(["download", "--dataset", "--task", "tapa"])
    assert capsys.readouterr().out == "wrote ckpt/trm/maze/seed0\nwrote data/tapa\n"
    assert hub.calls == [("hcshi/ICS", "model", "trm/maze/seed0/*"), ("hcshi/ICS-data", "dataset", "ppb-tapa-ms26/*")]
    assert files(tmp_path / "ckpt/trm/maze/seed0") == files(hub.root / "hcshi/ICS/trm/maze/seed0")
    assert files(tmp_path / "data/tapa") == files(hub.root / "hcshi/ICS-data/ppb-tapa-ms26")
    assert all(os.access(p, os.W_OK) for d in ("ckpt", "data") for p in (tmp_path / d).rglob("*"))    # not read-only
    load_trm(tmp_path / "ckpt/trm/maze/seed0")
    pool = load_pool(tmp_path / "data/tapa")                    # the labels, decoded, solve their boards
    assert all(PPB(pool, "tapa").valid(i, pool.labels[i]) for i in range(len(pool)))


def test_download_takes_another_repo_and_out(tmp_path, hub):
    for repo, folder in (("me/ICS-last", "gram/sudoku/seed1"), ("me/data", "lightup-ms26/test")):
        (hub.root / repo / folder).mkdir(parents=True)
        (hub.root / repo / folder / "f").write_text(repo)
    main(["download", "--model", "gram", "--task", "sudoku", "--seed", "1", "--repo", "me/ICS-last", "--out",
          str(tmp_path / "ck")])
    main(["download", "--dataset", "--task", "lightup", "--repo", "me/data", "--out", str(tmp_path / "lu")])
    assert hub.calls == [("me/ICS-last", "model", "gram/sudoku/seed1/*"), ("me/data", "dataset", "lightup-ms26/*")]
    assert (tmp_path / "ck" / "f").read_text() == "me/ICS-last"
    assert (tmp_path / "lu" / "test" / "f").read_text() == "me/data"


@pytest.mark.parametrize("args, error", [
    (["--dataset", "--task", "sudoku"], ("error: the sudoku dataset is built from its source, not downloaded: "
                                         "python -m ics data --task sudoku --out data/sudoku")),
    (["--dataset", "--task", "maze", "--out", "m"], "not downloaded: python -m ics data --task maze --out m"),
    (["--model", "trm", "--task", "maze"], "error: --model needs --seed"),
    (["--dataset", "--task", "tapa", "--seed", "0"], "error: --seed goes with --model"),
    (["--task", "tapa"], "error: one of the arguments --model --dataset is required"),
    (["--model", "trm", "--dataset", "--task", "tapa"], "error: argument --dataset: not allowed with argument --model"),
    (["--data", "data/tapa", "--task", "tapa"], "error: unrecognized arguments: data/tapa"),    # --data DIR elsewhere
])
def test_download_refuses_a_request_it_cannot_serve_before_any_download(tmp_path, monkeypatch, capsys, hub, args,
                                                                         error):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        main(["download", *args])
    assert stop.value.code == 2 and error in capsys.readouterr().err
    assert not hub.calls and not list(tmp_path.iterdir())


def test_download_refuses_an_out_in_use_and_a_folder_the_repo_lacks(tmp_path, capsys, hub):
    tiny_trm_checkpoint(hub.root / "hcshi/ICS/trm/maze/seed0", 900, 6)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("mine")
    for seed, out, error in (("0", tmp_path / "used", f"--out {tmp_path / 'used'} is not empty (notes.txt)"),
                             ("7", tmp_path / "new", "hcshi/ICS holds no trm/maze/seed7/")):
        with pytest.raises(SystemExit) as stop:
            main(["download", "--model", "trm", "--task", "maze", "--seed", seed, "--out", str(out)])
        assert stop.value.code == 2 and error in capsys.readouterr().err
    assert hub.calls == [("hcshi/ICS", "model", "trm/maze/seed7/*")]           # the out in use: refused before
    assert (tmp_path / "used" / "notes.txt").read_text() == "mine" and not (tmp_path / "new").exists()


def test_a_failure_of_the_hub_ends_in_its_message_without_a_traceback(tmp_path, capsys, hub):
    with pytest.raises(SystemExit) as stop:                         # no network, or no such repo
        main(["download", "--model", "trm", "--task", "maze", "--seed", "0", "--repo", "me/none", "--out",
              str(tmp_path / "out")])
    assert stop.value.code == 1 and not (tmp_path / "out").exists()
    assert capsys.readouterr().err == "python -m ics download: error: me/none: Repository Not Found\n"


@pytest.mark.parametrize("command", ["train", "eval"])
def test_set_reads_the_floats_yaml_1_1_leaves_as_strings(command, monkeypatch):
    cases = [("3e-4", 0.0003), ("1e5", 100000.0), ("1.5e5", 150000.0), ("-.5", -0.5), ("[9e-1,95e-2]", [0.9, 0.95]),
             ("'3e-4'", "3e-4"), ("1", 1), ("rope", "rope")]              # quoted: a string; no dot or exponent: an int
    seen = {}

    def run(*args, overrides, **options):
        seen.update(overrides)
        return {"results": {}}

    monkeypatch.setattr("ics.cli.train", run)
    monkeypatch.setattr("ics.cli.evaluate", run)
    model = ["--model", "trm"] if command == "train" else ["--method", "trm", "--ckpt", "ckpt"]
    main([command, *model, "--task", "sudoku", "--data", "data", "--out", "out",
          *(f"--set=x.v{i}={text}" for i, (text, _) in enumerate(cases))])
    read = list(seen["x"].values())
    assert read == [value for _, value in cases] and list(map(type, read)) == [type(value) for _, value in cases]


def test_train_from_the_command_line(tmp_path):
    data = sudoku_dataset(tmp_path / "data")
    main(["train", "--model", "trm", "--task", "sudoku", "--data", str(data), "--out", str(tmp_path / "run"),
          "--device", "cpu", *flags(merge_config(TINY_RECIPE, {"train": {"epochs": 1, "eval_boards": 2}}))])
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert log[-1]["step"] == 2 and log[-1]["best"]["step"] == 2
    assert json.loads((tmp_path / "run" / "last" / "config.json").read_text())["model"]["hidden_size"] == 32
    assert not (tmp_path / "run" / "best").exists()


def test_two_ranks_under_torchrun_train_like_one_and_one_rank_cannot_resume_them(tmp_path):
    data = sudoku_dataset(tmp_path / "data")
    # exploration off: the ranks draw differently; halt_max_steps 2: updates 3 and 5 take new rows; one eval board:
    # rank 1 decodes an empty share
    settings = merge_config(TINY_RECIPE, {"model": {"halt_exploration_prob": 0.0, "halt_max_steps": 2},
                                          "train": {"eval_boards": 1}})
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", "-m", "ics",
                    "train", "--model", "trm", "--task", "sudoku", "--data", str(data), "--out", str(tmp_path / "two"),
                    "--device", "cpu", *flags(settings)], check=True, cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT)})
    train("trm", "sudoku", data, tmp_path / "one", settings, device="cpu")
    one, two = (torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("one", "two"))
    for name, value in one["model"].items():        # the live weights: in 6 updates the EMA's hardly leave the init
        torch.testing.assert_close(two["model"][name], value, rtol=1e-4, atol=1e-6)
    assert len(two["ranks"]) == 2
    load_trm(tmp_path / "two" / "last")             # rank 0 exported a release checkpoint
    with pytest.raises(ValueError, match="belongs to another run"):
        train("trm", "sudoku", data, tmp_path / "two", settings, device="cpu")


def one_process_and_two_ranks(tmp_path, model: str, one: dict, two: dict) -> tuple[dict, dict]:
    """The state.pt of a CPU run of `model` on sudoku_dataset in one process with the recipe overrides `one` (in
    tmp_path / "one"), and of one on two ranks under torchrun with `two` (in tmp_path / "two")."""
    data = sudoku_dataset(tmp_path / "data")
    args = ["-m", "ics", "train", "--model", model, "--task", "sudoku", "--data", str(data), "--device", "cpu"]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}   # torchrun's ranks run one thread each
    subprocess.run([sys.executable, *args, "--out", str(tmp_path / "one"), *flags(one)], check=True, cwd=ROOT, env=env)
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *args,
                    "--out", str(tmp_path / "two"), *flags(two)], check=True, cwd=ROOT, env=env)
    return tuple(torch.load(tmp_path / d / "state.pt", weights_only=True) for d in ("one", "two"))


def test_accumulation_on_one_process_trains_exactly_like_more_ranks(tmp_path):
    # the draws are the only difference; halt_max_steps 2: updates 3 and 5 take new rows
    quiet = merge_config(TINY_RECIPE, {"model": {"halt_exploration_prob": 0.0, "halt_max_steps": 2}})
    a, b = one_process_and_two_ranks(tmp_path, "trm", merge_config(quiet, {"train": {"accumulate": 2}}), quiet)
    for part in ("model", "ema"):                                   # two-term gradient sums: bitwise, either way
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part])
    adam_a, adam_b = a["optimizers"][1]["state"], b["optimizers"][1]["state"]
    assert all(torch.equal(adam_a[i][m], adam_b[i][m]) for i in adam_a for m in ("exp_avg", "exp_avg_sq"))
    for v, carry in enumerate(a["ranks"][0]["carries"]):          # virtual rank v = rank v of the 2-rank run
        assert all(torch.equal(carry[k], b["ranks"][v]["carries"][0][k]) for k in carry)


def test_a_verifier_from_its_data_to_its_use_on_the_command_line(tmp_path, capsys):
    data = sudoku_dataset(tmp_path / "data")
    solver = tiny_trm_checkpoint(tmp_path / "solver", halt_max_steps=3)
    main(["verifier-data", "--task", "sudoku", "--data", str(data), "--solver", str(solver), "--out",
          str(tmp_path / "vdata"), "--hypotheses", "2", "--device", "cpu"])
    out, s = capsys.readouterr().out, json.loads((tmp_path / "vdata" / "val" / "dataset.json").read_text())["verifier"]
    assert f"train: {s['train']['candidates']} candidates, {s['train']['valid']} valid" in out
    assert out.endswith(f"wrote {tmp_path / 'vdata'}\n")
    main(["train", "--model", "verifier", "--task", "sudoku", "--data", str(tmp_path / "vdata"), "--out",
          str(tmp_path / "run"), "--init", str(solver), "--device", "cpu", *flags(TINY_VERIFIER)])
    assert json.loads((tmp_path / "run" / "best" / "config.json").read_text())["model"]["halt_max_steps"] == 2
    main(["eval", "--method", "ics", "--task", "sudoku", "--ckpt", str(solver), "--data", str(data), "--out",
          str(tmp_path / "res"), "--verifier", str(tmp_path / "run" / "best"), "--regime", "model", "--device", "cpu",
          "--set", "restatements=2", "--set", "model.T=1", "--set", "model.verifier_T=1"])
    with np.load(tmp_path / "res" / "ics-sudoku.model.npz") as z:      # 2 decodes, 2 views per distinct candidate
        assert len(z["ics__model__restatement"]) == 6
        assert set(z["ics__model__segs"]) <= {2 * 1 + 2 * 1, 2 * 1 + 2 * 2 * 1}


def test_verifier_data_refuses_an_out_on_its_data_or_taken_and_writes_nothing(tmp_path, capsys):
    data, solver = sudoku_dataset(tmp_path / "data"), tiny_trm_checkpoint(tmp_path / "solver")
    (tmp_path / "file").write_text("mine")
    refused = {data / ".." / "data": f"is the dataset (--data {data}); choose another --out",     # compared resolved
               data / "verifier": f"is inside the dataset (--data {data}); choose another --out",
               tmp_path: f"holds the dataset (--data {data}); choose another --out",
               solver: "is not empty (config.json, model.safetensors); choose another --out, or remove it",  # as data
               tmp_path / "file": "is not a directory; choose another --out"}
    link = tmp_path / "link"
    try:
        link.symlink_to(data, target_is_directory=True)
    except OSError:                                                 # no symlinks here (Windows without the privilege)
        pass
    else:
        refused |= {link: f"is the dataset (--data {data}); choose another --out",
                    link / "v": f"is inside the dataset (--data {data}); choose another --out"}
    before = files(tmp_path)
    for out, why in refused.items():
        with pytest.raises(SystemExit) as stop:
            main(["verifier-data", "--task", "sudoku", "--data", str(data), "--solver", str(solver), "--out",
                  str(out), "--device", "cpu"])
        assert stop.value.code == 2
        assert f"python -m ics verifier-data: error: --out {out} {why}" in capsys.readouterr().err
    assert files(tmp_path) == before and not (data / "verifier").exists() and not (data / "v").exists()


def test_verifier_data_refuses_out_of_range_numbers_before_anything_runs(tmp_path, capsys):
    for option, value, least in (("--snapshots", -1, 0), ("--hypotheses", -1, 0), ("--boards", 1, 2),
                                 ("--batch", 0, 1)):
        with pytest.raises(SystemExit) as stop:                     # --data and --solver do not exist
            main(["verifier-data", "--task", "sudoku", "--data", str(tmp_path / "data"), "--solver",
                  str(tmp_path / "solver"), "--out", str(tmp_path / "vdata"), option, str(value)])
        assert stop.value.code == 2
        assert (f"python -m ics verifier-data: error: argument {option}: expected {least} or more, got {value}"
                in capsys.readouterr().err)
    assert not (tmp_path / "vdata").exists()


def test_a_verifier_run_without_init_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:                         # refused before the data is read
        main(["train", "--model", "verifier", "--task", "sudoku", "--data", str(tmp_path / "vdata"), "--out",
              str(tmp_path / "run"), "--device", "cpu"])
    assert stop.value.code == 2 and not (tmp_path / "run").exists()
    assert ("python -m ics train: error: the verifier fine-tunes a checkpoint: pass --init, the solver's checkpoint"
            in capsys.readouterr().err)


@pytest.mark.parametrize("model, tiny, length", [("eqr", TINY_EQR, {"updates": 2}), ("gram", TINY_GRAM, {"epochs": 1}),
                                                 ("attractor", TINY_ATTRACTOR, {"updates": 2})])
def test_train_a_baseline_from_the_command_line(tmp_path, model, tiny, length):
    data = sudoku_dataset(tmp_path / "data")
    main(["train", "--model", model, "--task", "sudoku", "--data", str(data), "--out", str(tmp_path / "run"),
          "--device", "cpu", *flags(merge_config(tiny, {"train": length}))])
    saved = json.loads((tmp_path / "run" / "last" / "config.json").read_text())
    assert all(saved["model"][k] == v for k, v in tiny["model"].items())       # the tiny recipe's model, 2 updates
    assert saved["provenance"]["model"] == model and saved["provenance"]["step"] == 2


def test_micro_batch_gives_two_ranks_the_solver_calls_of_one_process_with_accumulation(tmp_path):
    # micro-batches of 2 rows: 2 per update on one process, 1 per rank on two. The regulariser is off (its probes would
    # come from each rank's own generator), and so is exploration; halt_max_steps 2: updates 3 and 5 take new rows
    settings = merge_config(TINY_ATTRACTOR, {"model": {"halt_exploration_prob": 0.0, "jacobian_reg_lambda": 0.0},
                                             "train": {"micro_batch": 2}})
    a, b = one_process_and_two_ranks(tmp_path, "attractor", settings, settings)
    for part in ("model", "ema"):                                   # two-term gradient sums: bitwise, either way
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part])
    assert len(a["ranks"][0]["carries"]) == 2 and len(b["ranks"]) == 2
    for v, carry in enumerate(a["ranks"][0]["carries"]):          # virtual rank v = rank v of the 2-rank run
        assert all(torch.equal(carry[k], b["ranks"][v]["carries"][0][k]) for k in carry)
    assert {json.loads((tmp_path / d / "last" / "config.json").read_text())["model"]["batch_size"]
            for d in ("one", "two")} == {2}
