import dataclasses
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from fakes import SOLUTION, maze_pool, sudoku_pool
from ics.checkpoint import save_checkpoint
from ics.cli import main
from ics.config import load_yaml, merge_config
from ics.evaluate import CONFIG_DIR, REGIMES, eval_config, evaluate, summarize
from ics.methods.ics import ICSConfig
from ics.registry import METHODS, TASKS
from ics.tasks import make_task
from ics.tasks.base import SCORINGS
from ics.tasks.sudoku import Sudoku
from ics.trm.model import TRM, TRMConfig, load_trm
from ics.verifier.select import run_ics_verifier
from ics_baselines.attractor.model import Attractor, AttractorConfig
from ics_baselines.eqr.model import EqR, EqRConfig
from ics_baselines.gram.model import GRAM, GRAMConfig


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    cfg = TRMConfig(seq_len=81, vocab_size=11, puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=1, L_cycles=1, L_layers=1,
                    hidden_size=16, num_heads=2, expansion=2.0, forward_dtype="float32")
    save_checkpoint(TRM(cfg).state_dict(), cfg, tmp_path / "ck")
    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    X = np.stack([SOLUTION.copy() for _ in range(3)])
    X[:, :5] = 1
    (d / "dataset.json").write_text(json.dumps({"pad_id": 0, "ignore_label_id": 0, "vocab_size": 11, "seq_len": 81,
                                                "num_puzzle_identifiers": 1, "sets": ["all"]}))
    np.save(d / "all__inputs.npy", X.astype(np.uint8))
    np.save(d / "all__labels.npy", np.stack([SOLUTION] * 3).astype(np.uint8))
    return tmp_path


def test_config_resolution():
    c = eval_config("ics", "sudoku")
    assert c["batch"] == 768 and c["window"] == 6144 and c["model"]["cells"] == 24 and c["cert"]["cells"] == 48
    c = eval_config("ics", "tapa")
    assert c["model"]["terminal"] == "confirm" and c["model"]["patience"] == [] and c["batch"] == 252
    assert eval_config("trm", "maze")["batch"] == 1000 and eval_config("ptrm", "lightup")["batch"] == 256


def defaults(fn) -> dict:
    return {k: p.default for k, p in inspect.signature(fn).parameters.items() if p.default is not p.empty}


def test_the_python_defaults_are_the_eval_configs():
    configs = {m: {k: v for k, v in load_yaml(CONFIG_DIR / f"{m}.yaml").items() if k != "tasks"}
               for m in METHODS}                                       # each config before any task's entry
    for method, spec in METHODS.items():
        if spec.run is not None:                                       # every method but ICS, run per regime
            assert defaults(spec.run) == configs[method], method
    ics = configs["ics"]
    assert ICSConfig("cert") == ICSConfig.from_settings({**ics["cert"], "agree": ics["model"]["agree"],
                                                         "batch": ics["batch"]})
    maze = eval_config("ics", "maze")["model"]                         # the verifier's defaults are Maze's protocol
    assert defaults(run_ics_verifier) == {"T": maze["T"], "verifier_T": maze["verifier_T"], "batch": maze["batch"]}
    sudoku = eval_config("ics", "sudoku")                              # restatements are seeded per ICS batch
    assert defaults(make_task) == {"restatements": sudoku["restatements"], "seed": sudoku["seed"],
                                   "block": sudoku["batch"]}
    assert defaults(Sudoku) == {"n_restatements": sudoku["restatements"], "seed": sudoku["seed"],
                                "block": sudoku["batch"]}


@pytest.mark.parametrize("method,over,keys", [
    ("trm", {"T": 20}, ["standard/model/raw", "depth_scaling/cert/pinned"]),
    ("ptrm", {"K": 2, "D": 3}, ["ptrm/model/raw", "ptrm/cert/pinned"]),
    ("ics", {"cert": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]},
             "model": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}, "restatements": 2},
     ["ics/cert/raw", "ics/model/pinned"]),
])
def test_evaluate_writes_results(setup, method, over, keys):
    s = evaluate(method, "sudoku", setup / "ck", setup / "data", setup / "out", overrides=over)
    for key in keys:
        assert 0.0 <= s["results"][key]["accuracy"] <= 1.0 and s["results"][key]["n"] == 3
    with np.load(setup / "out" / f"{method}-sudoku.npz") as z:
        answers = z[keys[0].replace("/", "__")]
        assert answers.shape == (3, 81) and answers.dtype == np.int16
        np.testing.assert_array_equal(z["index"], [0, 1, 2])
    assert json.loads((setup / "out" / f"{method}-sudoku.json").read_text())["method"] == method


def test_empty_pool_is_rejected(setup):
    with pytest.raises(ValueError, match="no test boards"):                 # before the checkpoint loads
        evaluate("trm", "sudoku", setup / "missing", setup / "data", setup / "out", start=3)


@pytest.mark.parametrize("method, regimes, error", [
    ("ics", (), "no regime given"),
    ("ics", ("cert", "Cert"), "unknown regime 'Cert'"),
    ("trm", ("cert",), "trm has no regime choice"),
    ("ptrm", ("model", "model"), "ptrm has no regime choice"),
])
def test_regimes_are_checked_before_anything_loads(setup, method, regimes, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        evaluate(method, "sudoku", setup / "missing", setup / "data", setup / "out", regimes=regimes)
    assert not (setup / "out").exists()


@pytest.mark.parametrize("kept", ["npz", "json"])
def test_a_result_is_overwritten_only_when_asked(setup, kept):
    evaluate("trm", "sudoku", setup / "ck", setup / "data", setup / "out", overrides={"T": 16})
    (setup / "out" / f"trm-sudoku.{'json' if kept == 'npz' else 'npz'}").unlink()     # either file alone is kept
    before = (setup / "out" / f"trm-sudoku.{kept}").read_bytes()
    with pytest.raises(FileExistsError, match="trm-sudoku"):                        # before the checkpoint loads
        evaluate("trm", "sudoku", setup / "missing", setup / "data", setup / "out", overrides={"T": 17})
    assert (setup / "out" / f"trm-sudoku.{kept}").read_bytes() == before
    evaluate("trm", "sudoku", setup / "ck", setup / "data", setup / "out", overrides={"T": 17}, overwrite=True)
    assert json.loads((setup / "out" / "trm-sudoku.json").read_text())["config"]["T"] == 17


def test_cli_overwrites_only_with_the_flag(setup):
    args = ["eval", "--method", "trm", "--task", "sudoku", "--ckpt", str(setup / "ck"), "--data", str(setup / "data"),
            "--out", str(setup / "cli"), "--device", "cpu", "--set", "T=16"]
    main(args)
    with pytest.raises(FileExistsError):
        main(args)
    main([*args, "--overwrite"])


def test_a_ptrm_run_starts_on_a_seeding_block(setup):
    for start, batch in ((1, 2), (0, 0)):                                   # a start inside a block; no blocks at all
        with pytest.raises(ValueError, match=f"got start={start}, batch={batch}$"):   # before the checkpoint loads
            evaluate("ptrm", "sudoku", setup / "missing", setup / "data", setup / "out", start=start, batch=batch)
    assert not (setup / "out").exists()
    s = evaluate("ptrm", "sudoku", setup / "ck", setup / "data", setup / "out", start=2, batch=2,
                 overrides={"K": 2, "D": 2})
    assert s["n"] == 1


@pytest.fixture
def maze_setup(tmp_path):
    torch.manual_seed(0)
    cfg = TRMConfig(seq_len=900, vocab_size=6, puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=1, L_cycles=1, L_layers=1,
                    hidden_size=16, num_heads=2, expansion=2.0, forward_dtype="float32")
    for name in ("solver", "verifier"):
        save_checkpoint(TRM(cfg).state_dict(), cfg, tmp_path / name)
    pool = maze_pool([1, 2, 3], (1, 1), (1, 10))
    d = tmp_path / "data" / "test"
    d.mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({"pad_id": 0, "ignore_label_id": 0, "vocab_size": 6, "seq_len": 900,
                                                "num_puzzle_identifiers": 1, "sets": ["all"]}))
    np.save(d / "all__inputs.npy", pool.inputs.astype(np.uint8))
    np.save(d / "all__labels.npy", pool.labels.astype(np.uint8))
    return tmp_path


def test_maze_routes_its_model_regime_to_the_verifier(maze_setup):
    small = {"cert": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}, "model": {"T": 2, "verifier_T": 1}}
    s = evaluate("ics", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out",
                 verifier=maze_setup / "verifier", overrides=small)
    with np.load(maze_setup / "out" / "ics-maze.npz") as z:
        assert "ics__cert__stage" in z and "ics__model__restatement" in z      # cert: the search; model: the verifier
        assert set(z["ics__model__segs"]) <= {8 * 2 + 8 * 1 * d for d in range(1, 9)}   # d distinct candidates
    assert set(s["results"]) == {"ics/cert/raw", "ics/cert/pinned", "ics/model/raw", "ics/model/pinned"}
    assert s["verifier"] == str(maze_setup / "verifier")
    with pytest.raises(ValueError, match="verifier"):       # at once: before the checkpoint loads or a regime runs
        evaluate("ics", "maze", maze_setup / "missing", maze_setup / "data", maze_setup / "out2", overrides=small)
    assert not (maze_setup / "out2").exists()
    s = evaluate("ics", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out3", regimes=("cert",),
                 overrides=small)                                        # the certificate regime needs no verifier
    assert set(s["results"]) == {"ics/cert/raw", "ics/cert/pinned"}


def test_the_summary_names_the_gpu_and_the_weights(maze_setup):
    def sha(ckpt):
        return hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest()

    small = {"cert": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}, "model": {"T": 2, "verifier_T": 1}}
    s = evaluate("ics", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out",
                 verifier=maze_setup / "verifier", overrides=small)
    assert s["gpu"] is None                                             # a CPU run
    assert s["ckpt_sha256"] == sha(maze_setup / "solver") != s["verifier_sha256"] == sha(maze_setup / "verifier")
    assert json.loads((maze_setup / "out" / "ics-maze.json").read_text())["ckpt_sha256"] == s["ckpt_sha256"]
    s = evaluate("trm", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out", overrides={"T": 16},
                 verifier=maze_setup / "verifier")                     # the verifier is not used
    assert s["verifier"] is s["verifier_sha256"] is None


def test_cli(setup, capsys):
    main(["eval", "--method", "trm", "--task", "sudoku", "--ckpt", str(setup / "ck"), "--data", str(setup / "data"),
          "--out", str(setup / "cli"), "--device", "cpu", "--limit", "2", "--set", "T=20"])
    out = capsys.readouterr().out
    assert "standard/model/raw" in out and "(" in out


def test_cli_help():
    flags = ("--method", "--task", "--ckpt", "--data", "--out", "--verifier", "--regime", "--start", "--limit",
             "--batch", "--set", "--device", "--dtype", "--overwrite")
    for argv, words in ((["--help"], ("eval",)), (["eval", "--help"], flags)):
        res = subprocess.run([sys.executable, "-m", "ics", *argv], capture_output=True, text=True, check=True)
        assert res.stdout.startswith("usage: python -m ics") and not res.stderr
        assert all(word in res.stdout for word in words)


def test_summarize_scores_only_answer_arrays():
    pool = sudoku_pool([(0,), (1,)])
    pool.is_golden = np.array([1, 0])
    wrong = SOLUTION.copy()
    wrong[1] = SOLUTION[0]                                    # board 1's blank cell repeats a digit of its row
    Y = np.stack([SOLUTION, wrong])
    s = summarize(Sudoku(pool), {"a/cert/raw": Y, "a/cert/pinned": Y[:, 0], "a/cert/segs": Y, "a/raw": Y})
    assert s == {"a/cert/raw": {"accuracy": 0.5, "correct": 1, "n": 2, "golden_correct": 1}}


def test_ics_windows_cover_the_pool(setup):
    small = {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}
    evaluate("ics", "sudoku", setup / "ck", setup / "data", setup / "out", regimes=("cert",),
             overrides={"window": 2, "restatements": 2, "cert": small})             # windows of 2 and 1 boards
    with np.load(setup / "out" / "ics-sudoku.cert.npz") as z:
        assert z["ics__cert__raw"].shape == (3, 81) and z["ics__cert__segs"].shape == (3,)
        np.testing.assert_array_equal(z["index"], [0, 1, 2])


def test_cli_evaluates_the_selected_boards(setup):
    main(["eval", "--method", "trm", "--task", "sudoku", "--ckpt", str(setup / "ck"), "--data", str(setup / "data"),
          "--out", str(setup / "shard"), "--device", "cpu", "--start", "1", "--limit", "1", "--set", "T=16"])
    np.testing.assert_array_equal(np.load(setup / "shard" / "trm-sudoku.rows1-2.npz")["index"], [1])   # full-split rows
    s = json.loads((setup / "shard" / "trm-sudoku.rows1-2.json").read_text())
    assert s["n"] == 1 and s["dtype"] == "float32"                          # no --dtype: the checkpoint's own


def test_cli_nests_settings_and_overrides_every_batch(maze_setup):
    main(["eval", "--method", "ics", "--task", "maze", "--ckpt", str(maze_setup / "solver"),
          "--data", str(maze_setup / "data"), "--out", str(maze_setup / "cli"), "--device", "cpu",
          "--verifier", str(maze_setup / "verifier"), "--regime", "model", "--batch", "8", "--dtype", "bfloat16",
          "--set", "model.T=1", "--set", "model.verifier_T=1"])
    s = json.loads((maze_setup / "cli" / "ics-maze.model.json").read_text())
    assert {k: s["config"]["model"][k] for k in ("T", "verifier_T", "batch")} == {"T": 1, "verifier_T": 1, "batch": 8}
    assert s["dtype"] == "bfloat16" and set(s["results"]) == {"ics/model/raw", "ics/model/pinned"}
    with np.load(maze_setup / "cli" / "ics-maze.model.npz") as z:
        assert set(z["ics__model__segs"]) <= {8 * 1 + 8 * 1 * d for d in range(1, 9)}   # d distinct candidates


def test_unknown_settings_are_rejected(setup, capsys):
    args = ["eval", "--method", "ics", "--task", "sudoku", "--ckpt", str(setup / "ck"), "--data", str(setup / "data"),
            "--out", str(setup / "out"), "--device", "cpu", "--regime", "cert", "--set", "restatements=2",
            "--set", "cert.levels=1", "--set", "cert.cells=2", "--set", "cert.node_budget=2",
            "--set", "cert.patience=[21]"]
    for bad, error in (("cert.levles=1", "unknown setting cert.levles"),                  # a typo
                       ("model.selector=verifier", "unknown setting model.selector"),     # a Maze setting, on Sudoku
                       ("cert.T.x=1", "setting cert.T is a single value, not {'x': 1}"),  # a group for a value
                       ("cert.T={}", "setting cert.T is a single value, not {}"),
                       ("cert=5", "setting cert is a group of settings, not 5")):         # a value for a group
        with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
            main(args + ["--set", bad])
    with pytest.raises(SystemExit):
        main(args + ["--set", "cert.levels"])                                           # no "=": an argparse error
    assert "argument --set: expected KEY=VALUE, got 'cert.levels'" in capsys.readouterr().err
    main(args)                                                                          # valid nested settings pass
    assert json.loads((setup / "out" / "ics-sudoku.cert.json").read_text())["config"]["cert"]["levels"] == 1


def test_the_verifier_loads_up_front_and_only_when_used(maze_setup, monkeypatch):
    monkeypatch.setattr("ics.evaluate.run_ics", lambda *a, **k: pytest.fail("a search ran before the verifier loaded"))
    with pytest.raises(FileNotFoundError, match="missing"):
        evaluate("ics", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out",
                 verifier=maze_setup / "missing")
    assert not (maze_setup / "out").exists()
    s = evaluate("trm", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out",
                 verifier=maze_setup / "missing", overrides={"T": 16})       # unused: neither loaded nor recorded
    assert s["verifier"] is None


@pytest.mark.parametrize("kept", [("last",), ("best", "last")], ids=["trm", "verifier"])
def test_a_training_run_as_the_checkpoint_or_the_verifier_is_refused_by_name(maze_setup, monkeypatch, kept):
    run = maze_setup / "run"                                    # a training run's directory: its checkpoints inside
    for name in kept:
        shutil.copytree(maze_setup / "solver", run / name)
    monkeypatch.setattr("ics.evaluate.run_ics", lambda *a, **k: pytest.fail("a search ran"))
    error = f"{run} is a training run; pass a checkpoint, e.g. {run / 'last'}"
    for ckpt, verifier in ((run, maze_setup / "verifier"), (maze_setup / "solver", run)):
        with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
            evaluate("ics", "maze", ckpt, maze_setup / "data", maze_setup / "out", verifier=verifier)
    assert not (maze_setup / "out").exists()


def test_an_unknown_selector_is_refused(maze_setup):
    with pytest.raises(ValueError, match=r"^unknown selector 'verifer'$"):    # before the checkpoint loads
        main(["eval", "--method", "ics", "--task", "maze", "--ckpt", str(maze_setup / "missing"),
              "--data", str(maze_setup / "data"), "--out", str(maze_setup / "out"), "--device", "cpu",
              "--set", "model.selector=verifer"])


def test_shards_and_single_regimes_get_their_own_files(setup):
    small = {"cert": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]},
             "model": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}, "restatements": 2}
    runs = ({},                                          # the full run
            {"regimes": ("cert", "cert")},               # one regime (named twice, still one)
            {"start": 1},                                # a shard: boards 1-2 of the 3
            {"regimes": ("model",), "limit": 5})         # one regime of a limited run (all 3 boards)
    for kw in runs:
        evaluate("ics", "sudoku", setup / "ck", setup / "data", setup / "out", overrides=small, **kw)
    stems = ("ics-sudoku", "ics-sudoku.cert", "ics-sudoku.rows1-3", "ics-sudoku.model.rows0-3")
    names = sorted(p.name for p in (setup / "out").iterdir())
    assert names == sorted(f"{s}.{x}" for s in stems for x in ("json", "npz"))


ROWS = {"trm": ("standard", "greedy_depth", "depth_scaling"), "ptrm": ("ptrm",), "ics": ("ics",),
        "gram": ("gram_lprm", "gram_majority"), "eqr": ("eqr",), "attractor": ("attractor",)}


@pytest.mark.data
@pytest.mark.parametrize("task", TASKS)
def test_every_method_runs_on_real_boards(dataset, tmp_path, task):
    # Tiny random models of the task's shape; their answers are wrong, but every stage and selection path runs.
    data = dataset(task)
    meta = json.loads((data / "test" / "dataset.json").read_text())
    torch.manual_seed(0)
    cfg = TRMConfig(seq_len=meta["seq_len"], vocab_size=meta["vocab_size"], puzzle_emb_ndim=16, puzzle_emb_len=1,
                    H_cycles=1, L_cycles=1, L_layers=1, hidden_size=16, num_heads=2, expansion=2.0,
                    forward_dtype="float32")
    save_checkpoint(TRM(cfg).state_dict(), cfg, tmp_path / "ck")
    tiny_baselines(tmp_path, meta["seq_len"], meta["vocab_size"])
    small = {"T": 2, "T_greedy": 2, "patience": [3], "levels": 2, "cells": 3, "node_budget": 4}
    overrides = {"trm": {"T": 20}, "ptrm": {"K": 3, "D": 3},
                 "ics": {"restatements": 3, "cert": small,
                         "model": {"T": 2, "verifier_T": 1} if task == "maze" else small},
                 "gram": {"N": 3, "D": 2}, "eqr": {"R": 3, "steps": 2}, "attractor": {"R": 3, "segments": 2}}
    for method, over in overrides.items():
        s = evaluate(method, task, tmp_path / (method if method in SMALL else "ck"), data, tmp_path / "out", limit=12,
                     batch=8, overrides=over, verifier=tmp_path / "ck" if task == "maze" else None)
        assert set(s["results"]) == {f"{row}/{regime}/{scoring}" for row in ROWS[method] for regime in REGIMES
                                     for scoring in SCORINGS}
        assert all(rec["n"] == 12 for rec in s["results"].values())
        with np.load(tmp_path / "out" / f"{method}-{task}.rows0-12.npz") as z:
            np.testing.assert_array_equal(z["index"], np.arange(12))
            for key in set(z.files) - {"index"}:                        # every key follows the grammar
                row, regime, *rest = key.split("__")
                assert row in ROWS[method] and regime in REGIMES and len(rest) in (1, 2)
                if rest[0] in SCORINGS and len(rest) == 1:
                    assert z[key].shape == (12, meta["seq_len"]) and z[key].dtype == np.int16
                else:
                    assert z[key].shape == (12,) and (len(rest) == 1 or rest[0] in SCORINGS)


def test_baseline_configs():
    assert eval_config("gram", "tapa") == {"N": 100, "D": 64, "seed": 0, "batch": 128}
    assert eval_config("gram", "sudoku")["batch"] == 3072 and eval_config("gram", "maze")["batch"] == 768
    assert eval_config("eqr", "heyawake") == {"R": 128, "steps": 16, "seed": 0, "batch": 8}
    assert eval_config("eqr", "maze")["steps"] == eval_config("eqr", "sudoku")["steps"] == 64
    assert eval_config("eqr", "sudoku")["batch"] == 32
    assert eval_config("attractor", "sudoku") == {"R": 128, "sigma": 0.5, "segments": 16, "seed": 0, "batch": 768,
                                                  "rows": 3072}
    assert eval_config("attractor", "maze")["batch"] == 192 and "rows" not in eval_config("attractor", "lightup")


SMALL = {"gram": {"N": 2, "D": 2}, "eqr": {"R": 2, "steps": 2}, "attractor": {"R": 2, "segments": 2}}
BASELINE_ROWS = {"gram": ("gram_lprm", "gram_majority"), "eqr": ("eqr",), "attractor": ("attractor",)}


def tiny_baselines(out, seq_len, vocab_size):
    """Random GRAM, EqR and Attractor checkpoints of a task's shape under out/{gram,eqr,attractor}."""
    torch.manual_seed(0)
    small = {"seq_len": seq_len, "vocab_size": vocab_size, "H_cycles": 1, "L_cycles": 1, "L_layers": 1,
             "hidden_size": 16, "num_heads": 2, "expansion": 2.0, "forward_dtype": "float32"}
    for name, model, cfg in (
            ("gram", GRAM, GRAMConfig(**small, puzzle_emb_len=1, pos_encodings="rope")),
            ("eqr", EqR, EqRConfig(**small, pos_encodings="rope")),
            ("attractor", Attractor, AttractorConfig(**small, puzzle_emb_ndim=16, puzzle_emb_len=1, deq_max_iter=3,
                                                     deq_min_iter=2))):
        save_checkpoint(model(cfg).state_dict(), cfg, out / name)


@pytest.fixture
def baselines(setup):
    tiny_baselines(setup, 81, 11)
    return setup


@pytest.mark.parametrize("method", SMALL)
def test_baselines_write_results(baselines, method):
    s = evaluate(method, "sudoku", baselines / method, baselines / "data", baselines / "out", overrides=SMALL[method])
    assert set(s["results"]) == {f"{r}/{g}/{c}" for r in BASELINE_ROWS[method] for g in REGIMES for c in SCORINGS}
    assert all(rec["n"] == 3 for rec in s["results"].values()) and s["method"] == method
    with np.load(baselines / "out" / f"{method}-sudoku.npz") as z:
        assert z[f"{BASELINE_ROWS[method][0]}__model__raw"].shape == (3, 81)
        np.testing.assert_array_equal(z["index"], [0, 1, 2])


@pytest.mark.parametrize("method", SMALL)
def test_the_runner_gets_the_effective_settings(baselines, monkeypatch, method):
    seen = []

    def record(model, task, **settings):
        seen.append(settings)
        return {}

    monkeypatch.setitem(METHODS, method, dataclasses.replace(METHODS[method], run=record))
    evaluate(method, "sudoku", baselines / method, baselines / "data", baselines / "out", overrides={"seed": 5})
    cfg = merge_config(eval_config(method, "sudoku"), {"seed": 5})     # Sudoku's own entry (e.g. EqR's 64 steps)
    assert seen == [{k: v for k, v in cfg.items() if k != "rows"}]


def test_a_checkpoint_of_another_model_is_refused(baselines):
    with pytest.raises(ValueError, match=re.escape("exactly the GRAMConfig fields")):
        evaluate("gram", "sudoku", baselines / "eqr", baselines / "data", baselines / "out", overrides=SMALL["gram"])
    with pytest.raises(ValueError, match=re.escape("exactly the TRMConfig fields")):
        evaluate("trm", "sudoku", baselines / "attractor", baselines / "data", baselines / "out", overrides={"T": 16})
    assert not (baselines / "out").exists()


@pytest.mark.parametrize("method", SMALL)
@pytest.mark.parametrize("seq_len, vocab_size, wrong", [
    (81, 12, "vocab_size 12 vs 11"),                             # unchecked, it runs silently
    (900, 6, "seq_len 900 vs 81, vocab_size 6 vs 11"),           # a Maze model: unchecked, it fails mid-forward
], ids=["another_vocabulary", "another_task"])
def test_a_checkpoint_for_another_dataset_is_refused(setup, method, seq_len, vocab_size, wrong):
    tiny_baselines(setup / "other", seq_len, vocab_size)
    ck = setup / "other" / method
    error = f"{ck}: the checkpoint was built for another dataset (checkpoint vs data: {wrong})"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        evaluate(method, "sudoku", ck, setup / "data", setup / "out", overrides=SMALL[method])
    assert not (setup / "out").exists()


def test_a_verifier_for_another_dataset_is_refused(maze_setup):
    cfg = TRMConfig(seq_len=81, vocab_size=11, puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=1, L_cycles=1, L_layers=1,
                    hidden_size=16, num_heads=2, expansion=2.0, forward_dtype="float32")
    save_checkpoint(TRM(cfg).state_dict(), cfg, maze_setup / "sudoku")          # a Sudoku model as the Maze verifier
    small = {"cert": {"levels": 1, "cells": 2, "node_budget": 2, "patience": [21]}, "model": {"T": 2, "verifier_T": 1}}
    error = (f"{maze_setup / 'sudoku'}: the checkpoint was built for another dataset "
             f"(checkpoint vs data: seq_len 81 vs 900, vocab_size 11 vs 6)")
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        evaluate("ics", "maze", maze_setup / "solver", maze_setup / "data", maze_setup / "out",
                 verifier=maze_setup / "sudoku", overrides=small)
    assert not (maze_setup / "out").exists()


@pytest.mark.parametrize("method", SMALL)
def test_a_baseline_run_starts_on_a_block(baselines, method):
    with pytest.raises(ValueError, match="got start=1, batch=2$"):                  # before the checkpoint loads
        evaluate(method, "sudoku", baselines / "missing", baselines / "data", baselines / "out", start=1, batch=2)
    assert not (baselines / "out").exists()


def test_attractor_on_sudoku_evaluates_only_the_configured_rows(baselines):
    small = {**SMALL["attractor"], "rows": 2}
    ck, data, out = baselines / "attractor", baselines / "data", baselines / "out"
    s = evaluate("attractor", "sudoku", ck, data, out, overrides=small)
    assert s["n"] == 2 and (out / "attractor-sudoku.json").exists()          # all the configured rows: a full run
    s = evaluate("attractor", "sudoku", ck, data, out, overrides=small, start=1, limit=5, batch=1)
    assert s["n"] == 1 and (out / "attractor-sudoku.rows1-2.json").exists()  # the limit stops at the last row
    with pytest.raises(ValueError, match="evaluates the first 2 test boards; got start=2$"):
        evaluate("attractor", "sudoku", baselines / "missing", data, out, overrides=small, start=2, batch=1)
    s = evaluate("attractor", "sudoku", ck, data, baselines / "all", overrides={**small, "rows": None})
    assert s["n"] == 3                                                         # rows: null evaluates the whole split


def test_a_given_verifier_selects_without_a_certificate_on_any_task(setup):
    judge = load_trm(setup / "ck")                      # the verifier: the solver with a random q head (the solver's
    torch.manual_seed(1)                                # q_halt is -5 on every row, so every candidate would tie)
    torch.nn.init.normal_(judge.inner.q_head.weight)
    save_checkpoint(judge.state_dict(), judge.config, setup / "verifier")
    small = {"levels": 1, "cells": 2, "node_budget": 2, "patience": []}
    over = {"restatements": 2, "cert": small, "model": {"T": 1, "verifier_T": 1}}
    s = evaluate("ics", "sudoku", setup / "ck", setup / "data", setup / "out", verifier=setup / "verifier",
                 overrides=over)
    assert s["verifier"] == str(setup / "verifier") and set(s["results"]) == {f"ics/{r}/{c}" for r in REGIMES
                                                                                for c in SCORINGS}
    assert s["config"]["model"]["selector"] == "verifier"              # the summary records the selection
    with np.load(setup / "out" / "ics-sudoku.npz") as z:
        assert "ics__cert__stage" in z and "ics__model__restatement" in z      # cert: the search; model: the verifier
        assert set(z["ics__model__segs"]) <= {2 * 1 + 1 * 2 * 1, 2 * 1 + 2 * 2 * 1}      # 2 views per distinct one
        assert 1 in z["ics__model__restatement"]                        # the verifier chooses: not always the first
    s = evaluate("ics", "sudoku", setup / "ck", setup / "data", setup / "out2", overrides={"restatements": 2,
                 "cert": small, "model": small})                       # without one, the model regime searches
    assert "selector" not in s["config"]["model"]
    with np.load(setup / "out2" / "ics-sudoku.npz") as z:
        assert "ics__model__stage" in z and "ics__model__restatement" not in z
