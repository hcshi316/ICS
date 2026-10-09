"""The verifier's training data (ics/verifier/data.py, build_verifier_data)."""
import ast
import json
import re
import types

import numpy as np
import pytest

import ics.verifier.data
from fakes import (
    SOLUTION,
    ScriptedTRM,
    lightup_source,
    ppb_source,
    sudoku_dataset,
    sudoku_pool,
    tiny_trm_checkpoint,
    write_trm_split,
)
from ics.builders import write_split
from ics.builders.lightup import build_lightup
from ics.builders.ppb import build_ppb
from ics.data import Pool, load_pool, load_train
from ics.methods.ics import ICSConfig, Node, _Search, propose
from ics.registry import TASKS
from ics.tasks.base import on_canvas
from ics.tasks.lightup import WALL
from ics.tasks.ppb import T_SHADE, T_WHITE
from ics.tasks.sudoku import Sudoku
from ics.trm.roll import roll
from ics.verifier.data import build_verifier_data, decodes, placed, solver_checkpoints, train_boards

CANVAS = ("lightup", "nurikabe", "tapa", "heyawake")


def test_hints_are_the_least_confident_editable_cells_with_their_first_alternative():
    task = Sudoku(sudoku_pool([(0, 30, 60)]))
    margins = np.full(81, 5.0)
    margins[[60, 0, 30]] = [0.5, 1.0, 3.0]                      # the gaps to the runner-up (pad, logit 0)
    r = roll(ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 0.0, margins)), task.X, 1, 8, topk=2)
    answer = task.pin(0, r.dec[0])
    # SOLUTION holds 10 in cell 60 and 2 in cell 0: the first other digit is 2, then 3
    first = lambda k: propose(task, 0, task.X[0], answer, r.top_ids[0], r.top_vals[0], cells=k, alts=1)
    assert first(2) == [(60, 2), (0, 3)]
    assert first(5) == [(60, 2), (0, 3), (30, 2)]
    assert first(0) == []


def test_hypotheses_write_the_models_best_other_digit_as_stage_b_does():
    # A blank decodes to SOLUTION's digit (logit 5), a given cell to itself; the blank token is every cell's runner-up
    # (4), then come the digits from the highest (0.1 * token). The first alternative is the best other digit, which
    # the two best tokens alone do not show (they would give the lowest).
    others = [-1.0, 4.0, *(0.1 * v for v in range(2, 11))]
    model = ScriptedTRM(11, lambda row, t, zl: (np.where(row == 1, SOLUTION, row), 0.0, 5.0), others)
    task = Sudoku(sudoku_pool([(0, 30, 60)]), n_restatements=1)
    r = roll(model, task.X, 1, 8)
    root = Node((), task.pin(0, r.dec[0]), r.top_ids[0], r.top_vals[0])
    stage_b = _Search(model, task, ICSConfig("cert", alts=1, cells=3))._hints(0, root)
    assert stage_b == [(0, 10), (30, 10), (60, 9)]                  # SOLUTION holds 2, 6 and 10 there
    _, ys, source = decodes(task, model, 1, 3, 8)
    hinted = np.tile(SOLUTION, (3, 1))
    for row, (c, tok) in zip(hinted, stage_b):
        row[c] = tok
    np.testing.assert_array_equal(ys[source == 1], hinted)         # a re-decode keeps its hint


def test_decodes_are_greedy_then_the_hinted_and_the_restated_inputs_decodes():
    task = Sudoku(sudoku_pool([(0, 1, 2), (5, 6)]), n_restatements=2)        # the board and one rewriting
    echo = ScriptedTRM(11, lambda row, t, zl: (row, 0.0, 1.0))      # answers its input: blanks stay blank
    owners, ys, source = decodes(task, echo, 3, 2, 8)
    np.testing.assert_array_equal(owners, [0, 1, 0, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(source, [0, 0, 1, 1, 1, 1, 2, 2])
    hinted = [task.X[0].copy() for _ in range(2)] + [task.X[1].copy() for _ in range(2)]
    for row, (c, tok) in zip(hinted, [(0, 2), (1, 2), (5, 2), (6, 2)]):    # ties: the cells in order, digit 1
        row[c] = tok
    # each decode echoes the row it saw; a restatement's, mapped back, is the board again
    np.testing.assert_array_equal(ys, np.concatenate([task.X, hinted, task.X]))
    owners, ys, source = decodes(task, echo, 3, 2, 8, boards=[1])                    # board 1 alone
    np.testing.assert_array_equal(owners, [1, 1, 1, 1])
    np.testing.assert_array_equal(source, [0, 1, 1, 2])
    np.testing.assert_array_equal(ys, [task.X[1], *hinted[2:], task.X[1]])


def test_a_run_stands_for_its_snapshots_spread_from_first_to_last_or_for_its_last_checkpoint(tmp_path):
    for step in (2, 10, 30, 400):
        tiny_trm_checkpoint(tmp_path / "run" / "snapshots" / f"step_{step}")
    run = tiny_trm_checkpoint(tmp_path / "run" / "last").parent
    tiny_trm_checkpoint(run / "best")                               # a run of an earlier trainer: never taken
    plain = tiny_trm_checkpoint(tmp_path / "plain" / "last").parent
    ckpt = tiny_trm_checkpoint(tmp_path / "ckpt")
    names = lambda paths: [p.name for p in paths]
    assert names(solver_checkpoints([run], 3)) == ["step_2", "step_30", "step_400"]
    assert names(solver_checkpoints([run], 1)) == ["step_400"]                      # one: the last
    assert names(solver_checkpoints([ckpt, run], 10)) == ["ckpt", "step_2", "step_10", "step_30", "step_400"]
    assert solver_checkpoints([run, plain, ckpt], 0) == [run / "last", plain / "last", ckpt]
    assert solver_checkpoints([run]) == solver_checkpoints([run], 10)                # the default: the snapshots
    maze = tmp_path / "maze"            # trm.yaml's Maze run: an evaluation every 6,510 updates, then the last, 65,104
    for step in (*range(6510, 65101, 6510), 65104):
        (maze / "snapshots" / f"step_{step}").mkdir(parents=True)
    # spread by step: the middle evaluation stays, and of the last two, 4 updates apart, only the last
    assert names(solver_checkpoints([maze])) == [f"step_{s}" for s in (*range(6510, 58591, 6510), 65104)]
    with pytest.raises(ValueError, match=r"keep_every_eval=true, or take the run's last checkpoint alone"):
        solver_checkpoints([plain], 3)                                              # a run that kept none
    with pytest.raises(ValueError, match=r"neither a checkpoint nor a training run with a last checkpoint \(last/\)"):
        solver_checkpoints([tmp_path / "run" / "snapshots"], 0)
    with pytest.raises(ValueError, match=r"neither a checkpoint nor a training run with a last checkpoint"):
        solver_checkpoints([tiny_trm_checkpoint(tmp_path / "old" / "best").parent], 0)     # best/ alone: not taken
    with pytest.raises(FileNotFoundError, match=r"missing: no such directory"):
        solver_checkpoints([tmp_path / "missing"], 3)


def test_placement_is_read_from_each_boards_ring():
    rows = [on_canvas(np.ones(shape, np.int64), 26, WALL) for shape in ((3, 4), (2, 2))]
    pool = Pool(np.stack(rows), np.stack(rows), np.arange(2), {})
    np.testing.assert_array_equal(placed(pool).dims, [[3, 4, 1, 1], [2, 2, 1, 1]])


def undecided(task, pool) -> bool:
    """No cell of the pool's PPB inputs is decided, shaded or unshaded (a Heyawake cell's token holds its state)."""
    if task == "heyawake":
        state = {v: ast.literal_eval(k)[4] for k, v in pool.vocab.items()}
        return all(state.get(int(t), 0) == 0 for t in np.unique(pool.inputs))
    return not np.isin(pool.inputs, (T_SHADE, T_WHITE)).any()


def canvas_dataset(root, task):
    """A canvas task's dataset as python -m ics data builds it from fakes' PPBench files (the fake_ppbench fixture): 3
    train boards of 8 rows (Light-Up) or 10 (the PPB tasks)."""
    if task == "lightup":
        build_lightup(root / "data", source=lightup_source(root / "src"))
    else:
        build_ppb(root / "data", task, source=ppb_source(root / "src", task))
    return root / "data"


def test_the_boards_are_the_first_puzzle_of_each_group_on_sudoku_and_maze(tmp_path):
    data = sudoku_dataset(tmp_path / "sudoku")                                  # 8 groups of 2 boards
    np.testing.assert_array_equal(train_boards("sudoku", data).index, np.arange(0, 16, 2))
    np.testing.assert_array_equal(train_boards("sudoku", data, 3).index, [0, 2, 4])
    X = np.ones((5, 900), np.int64)
    write_trm_split(tmp_path / "maze", "train", X, X)                             # a board per group
    np.testing.assert_array_equal(train_boards("maze", tmp_path / "maze").index, np.arange(5))


@pytest.mark.parametrize("task", CANVAS)
def test_the_canvas_boards_are_each_boards_first_row_as_given(tmp_path, fake_ppbench, task):
    data, k = canvas_dataset(tmp_path, task), TASKS[task].board_rows
    pool = train_boards(task, data)
    np.testing.assert_array_equal(pool.index, [0, k, 2 * k])                        # one row of each board's k
    np.testing.assert_array_equal(pool.inputs, load_pool(data, "train").inputs[::k])
    assert pool.dims is not None and (task == "lightup" or undecided(task, pool))


def keep_rows(data, rows):
    """The train split of `data` rewritten to hold these of its rows, in this order (its inputs, labels and dims)."""
    for name in ("inputs", "labels", "dims"):
        if (path := data / "train" / f"all__{name}.npy").exists():
            np.save(path, np.load(path)[rows])


@pytest.mark.parametrize("task", CANVAS)
def test_a_canvas_split_not_in_whole_boards_is_refused(tmp_path, fake_ppbench, task):
    data, k = canvas_dataset(tmp_path, task), TASKS[task].board_rows
    keep_rows(data, np.arange(3 * k - 1))
    with pytest.raises(ValueError, match=rf"a {task} train split, as python -m ics data writes it, holds {k} rows per "
                                         rf"board; this one holds {3 * k - 1} rows$"):
        train_boards(task, data)


def test_a_canvas_split_of_another_block_size_is_refused(tmp_path, fake_ppbench):
    data, k = canvas_dataset(tmp_path, "nurikabe"), TASKS["nurikabe"].board_rows
    block = [*range(10), *range(1, 10), 1]                      # 20 rows: row 10 is the first board's prefix
    keep_rows(data, np.concatenate([b * k + np.array(block) for b in (0, 1, 2, 0)]))     # 4 boards of the block
    with pytest.raises(ValueError, match=r"row 10 of the nurikabe train split, a board's first, has decided cells; a "
                                         r"nurikabe train split, as python -m ics data writes it, holds 10 rows per "
                                         r"board$"):
        train_boards("nurikabe", data)


@pytest.mark.parametrize("task", CANVAS[1:])
def test_a_ppb_split_whose_board_rows_have_decided_cells_is_refused(tmp_path, fake_ppbench, task):
    data = canvas_dataset(tmp_path, task)
    keep_rows(data, np.roll(np.arange(30), -1))               # every board's rows start one later: at a move prefix
    with pytest.raises(ValueError, match=rf"row 0 of the {task} train split, a board's first, has decided cells; a "
                                         rf"{task} train split, as python -m ics data writes it, holds 10 rows per "
                                         rf"board$"):
        train_boards(task, data)


def test_a_heyawake_split_without_its_vocabulary_is_refused_clearly(tmp_path, fake_ppbench):
    data = canvas_dataset(tmp_path, "heyawake")
    (data / "vocab.json").unlink()
    with pytest.raises(ValueError, match=r"PPB\('heyawake'\) needs vocab.json in the data directory"):
        train_boards("heyawake", data)


@pytest.mark.data
@pytest.mark.parametrize("task", ["lightup", "nurikabe", "tapa", "heyawake"])
def test_the_released_canvas_splits_give_each_board_once_as_given(dataset, task):
    pool, manifest = train_boards(task, dataset(task)), dataset(task) / "PPB_MANIFEST.json"
    n = json.loads(manifest.read_text())["train_boards"] if manifest.exists() else 1431    # Light-Up keeps no manifest
    assert len(pool) == n and (task == "lightup" or undecided(task, pool))


def scripted(monkeypatch, scripts: dict):
    """load_trm returns, for a checkpoint named in `scripts`, a solver of 2 segments answering scripts[name](row); the
    boards have no restatement but themselves."""
    monkeypatch.setattr(ics.verifier.data,"make_task", lambda name, pool: Sudoku(pool, n_restatements=1))
    def load(path, device="cpu"):
        model = ScriptedTRM(11, lambda row, t, zl: (scripts[path.name](row), 0.0, 1.0))
        model.config = types.SimpleNamespace(halt_max_steps=2)
        return model
    monkeypatch.setattr(ics.verifier.data,"load_trm", load)


def test_the_data_holds_each_boards_distinct_candidates_by_target(tmp_path, monkeypatch):
    # weak answers its input (blanks stay: invalid), strong answers SOLUTION (valid, as the train label). With one
    # hypothesis each board has 1 valid candidate (label = strong greedy = strong re-decode) and 2 invalid ones (weak
    # greedy, weak re-decode).
    data = sudoku_dataset(tmp_path / "data")                         # 8 train groups of 2 boards
    scripted(monkeypatch, {"weak": lambda row: row, "strong": lambda row: SOLUTION})
    solvers = [tiny_trm_checkpoint(tmp_path / "weak"), tiny_trm_checkpoint(tmp_path / "strong", seed=1)]
    s = build_verifier_data("sudoku", data, solvers, tmp_path / "v", hypotheses=1)
    assert (s["boards"], s["val_boards"], s["train_label"]) == (8, 1, [8, 0])
    assert [(x["checkpoint"], x["greedy"], x["hypotheses"], x["restatements"]) for x in s["solvers"]] == [
        ("weak", [0, 8], [0, 8], [0, 0]), ("strong", [8, 0], [8, 0], [0, 0])]
    assert s["train"] == {"candidates": 21, "valid": 7} and s["val"] == {"candidates": 3, "valid": 1}
    train, val = load_train(tmp_path / "v"), load_pool(tmp_path / "v", "val")
    np.testing.assert_array_equal(train.labels, [1] * 7 + [0] * 14)    # the valid candidates, then the invalid ones,
    np.testing.assert_array_equal(train.group_starts, [0, 7, 21])      # each in board order: two groups
    source = load_train(data)
    first = source.inputs[source.group_starts[:-1]]                 # each group's first board
    np.testing.assert_array_equal(train.inputs[:7], np.tile(SOLUTION, (7, 1)))
    np.testing.assert_array_equal(train.inputs[7::2], first[:7])     # the weak greedy decode, pinned: the input
    np.testing.assert_array_equal(val.inputs[1], first[7])          # the last board is the val board
    assert np.isin(train.rows(np.arange(21))["labels"], [0, 1]).all()        # targets: no label is ignored
    meta = json.loads((tmp_path / "v" / "val" / "dataset.json").read_text())
    assert meta["ignore_label_id"] is None and meta["verifier"] == s and meta["vocab_size"] == 11


def test_the_data_is_the_same_decoded_window_by_window(tmp_path, monkeypatch):
    data = sudoku_dataset(tmp_path / "data")
    scripted(monkeypatch, {"weak": lambda row: row, "strong": lambda row: SOLUTION})
    monkeypatch.setattr(ics.verifier.data,"make_task", lambda name, pool: Sudoku(pool, n_restatements=3))
    solvers = [tiny_trm_checkpoint(tmp_path / "weak"), tiny_trm_checkpoint(tmp_path / "strong", seed=1)]
    whole = build_verifier_data("sudoku", data, solvers, tmp_path / "whole", hypotheses=2)
    monkeypatch.setattr(ics.verifier.data,"WINDOW", 3)                     # the 8 boards in windows of 3, 3, 2
    assert build_verifier_data("sudoku", data, solvers, tmp_path / "windows", hypotheses=2) == whole
    files = sorted(p.relative_to(tmp_path / "whole") for p in (tmp_path / "whole").rglob("*") if p.is_file())
    assert files
    for f in files:
        assert (tmp_path / "whole" / f).read_bytes() == (tmp_path / "windows" / f).read_bytes(), f


def test_a_split_whose_candidates_are_all_valid_is_refused(tmp_path, monkeypatch):
    data = sudoku_dataset(tmp_path / "data")
    scripted(monkeypatch, {"strong": lambda row: SOLUTION})
    with pytest.raises(ValueError, match="the train boards' candidates are all valid"):
        build_verifier_data("sudoku", data, [tiny_trm_checkpoint(tmp_path / "strong")], tmp_path / "v", hypotheses=2)
    assert not (tmp_path / "v").exists()


def test_labels_that_fail_the_rule_checker_are_refused_as_data_of_another_task(tmp_path, monkeypatch):
    data = sudoku_dataset(tmp_path / "data")
    np.save(data / "train" / "all__labels.npy", np.load(data / "train" / "all__inputs.npy"))   # blanks: invalid
    scripted(monkeypatch, {"weak": lambda row: row})
    with pytest.raises(ValueError, match=r"the labels of the train boards all fail the sudoku rule checker, as every "
                                         r"other candidate does: the data .* does not match the task sudoku"):
        build_verifier_data("sudoku", data, [tiny_trm_checkpoint(tmp_path / "weak")], tmp_path / "v", hypotheses=1)
    assert not (tmp_path / "v").exists()


def test_a_solver_built_for_other_data_is_refused_before_any_is_loaded(tmp_path, monkeypatch):
    data = sudoku_dataset(tmp_path / "data")
    monkeypatch.setattr(ics.verifier.data,"load_trm", lambda *args: pytest.fail("a solver was loaded"))
    solvers = [tiny_trm_checkpoint(tmp_path / "fits"), tiny_trm_checkpoint(tmp_path / "other", vocab_size=12)]
    with pytest.raises(ValueError, match=r"other: the checkpoint was built for another dataset \(checkpoint vs data: "
                                         r"vocab_size 12 vs 11\)"):
        build_verifier_data("sudoku", data, solvers, tmp_path / "v")
    assert not (tmp_path / "v").exists()


@pytest.mark.parametrize("where", ["is", "is inside", "holds"])
def test_the_data_is_never_written_over_its_dataset(tmp_path, monkeypatch, where):
    data = sudoku_dataset(tmp_path / "data")
    out = {"is": data, "is inside": data / "v", "holds": tmp_path}[where]
    monkeypatch.setattr(ics.verifier.data,"load_trm", lambda *args: pytest.fail("a solver was loaded"))
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=rf"^out {re.escape(str(out))} {where} the dataset \(data "
                                         rf"{re.escape(str(data))}\); choose another out$"):         # the API's names
        build_verifier_data("sudoku", data, [tiny_trm_checkpoint(tmp_path / "ckpt")], out)
    assert {p: p.read_bytes() for p in data.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("groups", [0, 1])
def test_a_train_split_of_fewer_than_2_boards_is_refused(tmp_path, groups):
    data = sudoku_dataset(tmp_path / "data", groups=groups)                     # a board per group
    with pytest.raises(ValueError, match=rf"the train split holds {groups} boards; the verifier's data needs at "
                                         rf"least 2"):
        build_verifier_data("sudoku", data, [tiny_trm_checkpoint(tmp_path / "ckpt")], tmp_path / "v")
    assert not (tmp_path / "v").exists()


@pytest.mark.parametrize("setting", [{"snapshots": -1}, {"hypotheses": -1}, {"batch": 0}, {"boards": 1}])
def test_settings_out_of_range_are_refused(tmp_path, setting):
    data = sudoku_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="build_verifier_data needs snapshots >= 0, hypotheses >= 0, batch >= 1 and "
                                         "boards >= 2"):
        build_verifier_data("sudoku", data, [tiny_trm_checkpoint(tmp_path / "ckpt")], tmp_path / "v", **setting)
    assert not (tmp_path / "v").exists()


def test_write_split_stores_tokens_past_a_byte_as_int32_and_takes_further_fields(tmp_path):
    for vocab, dtype in ((256, np.uint8), (300, np.int32)):
        X = np.array([[0, vocab - 1], [1, 2]])
        write_split(tmp_path / str(vocab), "train", X, np.array([1, 0]), np.array([1, 1]), vocab, ignore_label_id=None,
                    note="x")
        split = load_train(tmp_path / str(vocab))
        assert split.inputs.dtype == dtype and split.labels.dtype == dtype
        np.testing.assert_array_equal(split.inputs, X)
        np.testing.assert_array_equal(split.labels, [1, 0])
        assert (split.meta["ignore_label_id"], split.meta["note"], split.meta["vocab_size"]) == (None, "x", vocab)


@pytest.mark.data
@pytest.mark.parametrize("task", TASKS)
def test_the_data_builds_from_every_tasks_train_boards(dataset, tmp_path, task):
    # A tiny random solver's decodes are invalid; every train label must pass the task's rule checker.
    data = dataset(task)
    meta = json.loads((data / "train" / "dataset.json").read_text())
    ckpt = tiny_trm_checkpoint(tmp_path / "ckpt", meta["seq_len"], meta["vocab_size"])
    s = build_verifier_data(task, data, [ckpt], tmp_path / "v", hypotheses=1, boards=10, batch=16)
    assert s["train_label"] == [10, 0] and s["val_boards"] == 1
    assert s["solvers"][0]["greedy"][0] + s["solvers"][0]["greedy"][1] == 10
    for split in ("train", "val"):
        assert load_pool(tmp_path / "v", split).inputs.shape[1] == meta["seq_len"]
