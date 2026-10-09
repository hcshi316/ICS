import ast
import hashlib
import json
import random
import subprocess
import sys
import types

import numpy as np
import pytest

from fakes import (
    DIGITS,
    click,
    lightup_source,
    make_pool,
    maze_source,
    ppb_source,
    ppbench_row,
    ppbench_source,
    sudoku_source,
    write_csv,
)
from ics.builders import ppb, read_csv
from ics.builders.lightup import build_lightup, canonical
from ics.builders.maze import build_maze
from ics.builders.ppb import build_ppb
from ics.builders.sudoku import build_sudoku, shuffle_sudoku
from ics.data import IGNORE, load_pool, load_train
from ics.registry import TASKS
from ics.tasks.lightup import BULB, W0, LightUp
from ics.tasks.ppb import PPB


def is_sudoku(grid) -> bool:
    g = np.asarray(grid).reshape(9, 9)
    boxes = g.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return all(sorted(unit) == list(range(1, 10)) for unit in [*g, *g.T, *boxes])


def test_a_rewriting_keeps_the_solution_valid_and_the_givens_on_it():
    rs = np.random.RandomState(0)
    solution = np.array([int(c) for c in DIGITS])
    board = np.where(np.arange(81) % 3 == 0, 0, solution)
    for _ in range(20):
        b, s = shuffle_sudoku(rs, board, solution)
        assert is_sudoku(s) and ((b == 0) | (b == s)).all() and (b == 0).sum() == 27


def test_the_sudoku_build(tmp_path):
    src = sudoku_source(tmp_path / "src")
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        build_sudoku(tmp_path / name, seed=seed, subsample=4, augment=5, source=src)
    train = load_train(tmp_path / "a")
    X, Y = np.asarray(train.inputs), np.asarray(train.labels)
    np.testing.assert_array_equal(train.group_starts, [0, 6, 12, 18, 24])
    assert train.meta["total_groups"] == 4 and train.meta["total_puzzles"] == 24 and train.meta["vocab_size"] == 11
    assert X.dtype == np.uint8 and X.min() >= 1 and X.max() <= 10 and all(is_sudoku(y - 1) for y in Y)
    questions, _ = read_csv(src / "train.csv")
    first = ["".join("." if t == 1 else str(t - 1) for t in x) for x in X[::6]]
    assert set(first) <= set(questions)                         # each group opens with its board, unchanged
    test = load_pool(tmp_path / "a", "test")
    assert len(test) == 3 and json.loads((tmp_path / "a" / "test" / "dataset.json").read_text())["total_groups"] == 3
    files = lambda d: {p.relative_to(d): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
    assert files(tmp_path / "a") == files(tmp_path / "b")
    assert not np.array_equal(np.load(tmp_path / "a" / "train" / "all__inputs.npy"),
                              np.load(tmp_path / "c" / "train" / "all__inputs.npy"))
    malformed = {"length": (DIGITS[:-1], DIGITS), "characters outside": ("-" + DIGITS[1:], DIGITS),
                 "blank": (DIGITS, "." + DIGITS[1:])}                    # error message: (question, answer)
    for error, (question, answer) in malformed.items():
        write_csv(tmp_path / "bad" / "train.csv", [["src", question, answer, "0"]])
        with pytest.raises(ValueError, match=error):
            build_sudoku(tmp_path / "d", source=tmp_path / "bad")


def test_the_maze_build(tmp_path):
    build_maze(tmp_path / "m", source=maze_source(tmp_path / "src"))
    train = load_train(tmp_path / "m")
    np.testing.assert_array_equal(train.group_starts, [0, 1, 2])
    np.testing.assert_array_equal(train.inputs[0, :4], [1, 2, 3, 4])
    np.testing.assert_array_equal(train.labels[0, :4], [1, 5, 3, 4])
    assert train.meta["seq_len"] == 900 and train.meta["vocab_size"] == 6
    # the original script's format, which its byte-identity rests on: uint8 tokens, int32 indices, dataset.json's text
    assert train.inputs.dtype == train.labels.dtype == np.uint8 and len(load_pool(tmp_path / "m", "test")) == 1
    for split in ("train", "test"):
        for name in ("group_indices", "puzzle_indices", "puzzle_identifiers"):
            assert np.load(tmp_path / "m" / split / f"all__{name}.npy").dtype == np.int32
    assert (tmp_path / "m" / "test" / "dataset.json").read_text() == (
        '{"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 6, "seq_len": 900, '
        '"num_puzzle_identifiers": 1, "total_groups": 1, "mean_puzzle_examples": 1.0, "total_puzzles": 1, '
        '"sets": ["all"]}')
    maze, path = "# SG" * 225, "#oSG" * 225
    malformed = [("characters outside", maze[:-1] + "x", path), (r"length \[899\], not 900", maze[:-1], path),
                 (r"length \[901\], not 900", maze, path + "o")]              # (error message, question, answer)
    for error, question, answer in malformed:
        write_csv(tmp_path / "bad" / "train.csv", [["s", question, answer, "0"]])
        with pytest.raises(ValueError, match=error):
            build_maze(tmp_path / "m2", source=tmp_path / "bad")


@pytest.mark.parametrize("case", ["missing", "malformed"])
@pytest.mark.parametrize("task", ["sudoku", "maze"])
def test_a_missing_or_malformed_test_csv_stops_the_build_before_it_writes(tmp_path, task, case):
    """test.csv missing, or with a blank Sudoku answer or a maze of 899 cells: the build raises the error it always did
    (naming test.csv), but before the train split is written, so `out` stays missing, or empty if it was."""
    build, source = {"sudoku": (build_sudoku, sudoku_source), "maze": (build_maze, maze_source)}[task]
    row, error = {"sudoku": (["src", DIGITS, "." + DIGITS[1:], "0"], r"test\.csv: answers with a blank"),
                  "maze": (["s", ("# SG" * 225)[:-1], "#oSG" * 225, "0"],
                           r"test\.csv: rows of length \[899\], not 900")}[task]
    src = source(tmp_path / "src")
    if case == "missing":
        (src / "test.csv").unlink()
        kind, error = FileNotFoundError, r"test\.csv"
    else:
        write_csv(src / "test.csv", [row])
        kind = ValueError
    new, empty = tmp_path / "new", tmp_path / "empty"
    empty.mkdir()
    for out in (new, empty):
        with pytest.raises(kind, match=error):
            build(out, source=src)
    assert {out.name: sorted(p.name for p in out.iterdir()) for out in (new, empty) if out.exists()} == {"empty": []}


def test_a_csv_without_rows_or_with_a_short_row_is_refused(tmp_path):
    write_csv(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no rows"):
        read_csv(tmp_path / "empty.csv")
    write_csv(tmp_path / "short.csv", [["s", DIGITS, DIGITS, "0"], ["s", DIGITS]])
    with pytest.raises(ValueError, match="line 3: fewer than 3 columns"):
        read_csv(tmp_path / "short.csv")


def crop(pool, i, row) -> np.ndarray:
    """Board i of a canvas row, as pool.dims places it."""
    h, w, r0, c0 = pool.dims[i]
    return np.asarray(row).reshape(26, 26)[r0:r0 + h, c0:c0 + w]


def cells(pool, i, row, vocab, kind) -> np.ndarray:
    """Board i of a canvas row as an object grid: each cell's token, or for Heyawake its (room clue, state)."""
    rev = {v: ast.literal_eval(k) for k, v in vocab.items()}
    out = np.empty(tuple(pool.dims[i][:2]), object)
    for (r, c), t in np.ndenumerate(crop(pool, i, row)):
        out[r, c] = rev[t][3:] if kind == "heyawake" else int(t)
    return out


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", ["nurikabe", "tapa", "heyawake"])
def test_the_ppb_build(tmp_path, fake_ppbench, kind, seed):
    out, src = tmp_path / kind, ppb_source(tmp_path / "src", kind)
    build_ppb(out, kind, seed=seed, source=src)
    vocab = json.loads((out / "vocab.json").read_text())
    test, train_rows = load_pool(out, "test"), load_pool(out, "train")
    X, Y = train_rows.inputs, train_rows.labels
    side = lambda x, axis: int(x.reshape(26, 26).any(axis=axis).sum()) - 2         # inside the ring of walls
    train = make_pool(X, Y, np.array([(side(x, 1), side(x, 0), 1, 1) for x in X]), vocab)
    # test: 2 of t0..t2 (val), then golden g1 and g2; train: the third, s1, s2, 10 rows each
    assert len(test) == 4 and len(train) == 30
    np.testing.assert_array_equal(test.is_golden, [0, 0, 1, 1])
    np.testing.assert_array_equal(test.dims[:, :2], [[10, 10], [10, 10], [2, 2], [2, 2]])
    for pool in (test, train):                           # every label solves its board, as the task reads the tokens
        task = PPB(pool, kind)
        assert all(task.valid(i, pool.labels[i]) for i in range(len(pool)))
    manifest = json.loads((out / "PPB_MANIFEST.json").read_text())
    val = manifest.pop("val_keys")
    assert set(val) < {"t0", "t1", "t2"} and manifest == {
        "pid": kind, "canvas": 26, "pool": 6, "train_boards": 3, "aug": 10, "val": 2, "golden": 2,
        "vocab_size": 5 + len(vocab), "val_seed": 20260808, "golden_keys": ["g1", "g2"],
        **({"seed": 1} if seed else {})}                 # a seed other than 0 is recorded
    drawn = random.Random(20260808).sample(["t0", "t1", "t2"], 2)          # val, in the order drawn
    first_clue = [[c is not None for c in info["clues"][0]].index(True) for info in PPB(test, kind).info[:2]]
    assert [f"t{j}" for j in first_clue] == drawn                          # tj: a clue in cell (0, j)
    # a train board's rows: undecided -> solved, then 9 prefixes, each turned with its solution, drawn as documented
    rows = {r["sort_key"]: r for r in map(json.loads, (src / "full_dataset.jsonl").read_text().splitlines())}
    same = lambda a, b: a.shape == b.shape and bool((a == b).all())

    def board(solved, state):                            # the cells of a solved board, in `state`
        grid = solved.copy()
        for (r, c), s in np.ndenumerate(np.array(state)):
            if kind == "heyawake":
                grid[r, c] = (grid[r, c][0], int(s))
            elif grid[r, c] < 5:
                grid[r, c] = ppb.STATE_TOKENS[s]
        return grid

    for b, key in enumerate([*({"t0", "t1", "t2"} - set(val)), "s1", "s2"]):
        moves, solved = ppb.solution_moves(rows[key]), cells(train, 10 * b, Y[10 * b], vocab, kind)
        undecided = board(solved, ppb.replay([], *solved.shape, ppb.EMPTY))
        assert same(cells(train, 10 * b, X[10 * b], vocab, kind), undecided)
        rng = random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16) + (seed << 32))
        for i in range(10 * b + 1, 10 * b + 10):
            prefix = ppb.replay(moves[:rng.randint(1, max(1, len(moves) - 1))], *solved.shape, ppb.EMPTY)
            k, mirror = rng.randrange(4), rng.random() < 0.5
            want = [np.rot90(grid, k) for grid in (board(solved, prefix), solved)]
            for got, grid in zip((X[i], Y[i]), want):
                assert same(cells(train, i, got, vocab, kind), np.fliplr(grid) if mirror else grid)
    assert list(vocab.values()) == list(range(5, 5 + len(vocab)))           # numbered in order of first use:
    first = crop(test, 0, test.inputs[0]).ravel()
    assert first[first >= 5][0] == 5                                       # first, the first test row's first clue
    assert {"nurikabe": "('nu', -1)", "tapa": "('ta', '-1,1')", "heyawake": "('hy', 1, 0, None, 2)"}[kind] in vocab
    # the released datasets' format: int32 tokens, int64 indices, dims and is_golden in the test split only
    dtype = lambda split, name: np.load(out / split / f"all__{name}.npy").dtype
    for split in ("train", "test"):
        assert dtype(split, "inputs") == dtype(split, "labels") == np.int32 == dtype(split, "puzzle_identifiers")
        assert dtype(split, "group_indices") == np.int64
    assert not (out / "train" / "all__dims.npy").exists() and test.dims.dtype == np.int32
    assert json.loads((out / "identifiers.json").read_text()) == ["<blank>"]
    assert json.loads((out / "test" / "dataset.json").read_text())["total_samples"] == 4
    if seed:                                             # the seed redraws the train split; the test arrays stay
        build_ppb(tmp_path / "default", kind, source=src)
        files = lambda d: {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file()}
        new, old = files(out), files(tmp_path / "default")
        differ = {f for f in old if new[f] != old[f]}
        assert {"train/all__inputs.npy", "train/all__labels.npy"} <= differ
        # but Heyawake's stored test labels, whose encoding takes the size of its vocabulary; decoded, they stay too
        moved = {"test/all__labels.npy"} if kind == "heyawake" else set()
        assert {f for f in differ if f.startswith("test/all__")} <= moved
        np.testing.assert_array_equal(load_pool(tmp_path / "default", "test").labels, test.labels)
        if kind != "heyawake":                           # so does the rest, but for Heyawake's vocabulary (see ppb.py)
            assert differ == {"train/all__inputs.npy", "train/all__labels.npy", "PPB_MANIFEST.json"}
            assert json.loads(new["PPB_MANIFEST.json"]) == {**json.loads(old["PPB_MANIFEST.json"]), "seed": 1}


# A golden board "g3" of each kind that no map of the square but the identity leaves unchanged: its cell lines, its
# solution ("#": shaded) and Heyawake's rooms; "same", another text of its puzzle (Tapa: a clue's runs in another
# order; Heyawake: a room's number in another of its cells); "shared", another puzzle with the same solution.
TWINS = {"nurikabe": {"grid": ["2 . .", ". . 1"], "solved": [". # #", ". # ."], "shared": [". . .", "2 . 1"]},
         "tapa": {"grid": [". . . .", ". 2,1 . .", "0 . . ."], "solved": [". # # #", ". . . #", ". . # #"],
                  "same": [". . . .", ". 1,2 . .", "0 . . ."], "shared": [". . . .", ". 2,1 . .", ". . . ."]},
         "heyawake": {"grid": [". 1 .", ". . ."], "rooms": ["0 0 1", "0 1 1"], "solved": ["# . .", ". . #"],
                      "same": [". . .", "1 . ."], "shared": [". 1 .", ". 1 ."]}}


def twin_row(kind: str, key: str, grid: list[str], k: int = 0, mirror: bool = False) -> dict:
    """The PPBench row of the TWINS board of `kind` with the cell lines `grid`, under k quarter turns (np.rot90), then
    a left-right mirror if `mirror`; its solution's moves left-click the shaded cells, and Heyawake's rooms are
    numbered anew in reading order, as pzpr.js numbers them."""
    twin, ids = TWINS[kind], {}

    def mapped(lines):
        cells = np.rot90(np.array([line.split() for line in lines], object), k)
        return [" ".join(row) for row in (np.fliplr(cells) if mirror else cells)]

    solved = [row.split() for row in mapped(twin["solved"])]
    moves = [click("left", r, c) for r, row in enumerate(solved) for c, cell in enumerate(row) if cell == "#"]
    rooms = [" ".join(str(ids.setdefault(room, len(ids))) for room in row.split()) for row in mapped(twin["rooms"])] \
        if "rooms" in twin else None
    return ppbench_row(kind, key, mapped(grid), moves, rooms)


@pytest.mark.parametrize("kind", ["nurikabe", "tapa", "heyawake"])
def test_a_train_board_posing_a_test_boards_puzzle_is_dropped_with_its_rows(tmp_path, fake_ppbench, capsys, kind):
    twin, maps = TWINS[kind], [(k, mirror) for k in range(4) for mirror in (False, True)]
    planted = [twin_row(kind, f"d{i}", twin["grid"], k, mirror) for i, (k, mirror) in enumerate(maps)]
    planted += [twin_row(kind, "same", twin["same"])] if "same" in twin else []
    files = {}
    for name, extra in (("base", []), ("planted", planted)):    # the same source, but for the planted boards
        src, out = ppb_source(tmp_path / name, kind), tmp_path / name / "out"
        for file, rows in (("golden_300.jsonl", [twin_row(kind, "g3", twin["grid"])]),
                           ("full_dataset.jsonl", [*extra, twin_row(kind, "shared", twin["shared"])])):
            with open(src / file, "a") as f:
                f.write("".join(json.dumps(row) + "\n" for row in rows))
        build_ppb(out, kind, source=src)
        files[name] = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    # every planted board, under the map that takes it back to g3 (a mirrored map is its own inverse)
    dropped = [{"sort_key": f"d{i}", "test": "g3", "k": k if mirror else -k % 4, "mirror": mirror}
               for i, (k, mirror) in enumerate(maps)]
    if "same" in twin:
        dropped.append({"sort_key": "same", "test": "g3", "k": 0, "mirror": False})
    base = json.loads(files["base"].pop("PPB_MANIFEST.json"))
    assert base["train_boards"] == 4 and "dropped" not in base          # the third 10x10 board, s1, s2 and shared
    assert json.loads(files["planted"].pop("PPB_MANIFEST.json")) == {**base, "pool": base["pool"] + len(dropped),
                                                                     "dropped": dropped}
    assert files["planted"] == files["base"]            # the rest byte for byte: the planted boards' 10 rows each gone
    assert capsys.readouterr().out == "".join(
        f"{kind}: train board {d['sort_key']} dropped: its puzzle is test board g3's "
        f"(k={d['k']}{', mirrored' if d['mirror'] else ''})\n" for d in dropped)
    full = tmp_path / "planted" / "full_dataset.jsonl"            # val takes t0 and t1, and train is the planted boards
    keys = {"t0", "t1", *(row["sort_key"] for row in planted)}
    full.write_text("".join(f"{line}\n" for line in full.read_text().splitlines()
                            if json.loads(line)["sort_key"] in keys))
    with pytest.raises(ValueError, match=f"^{kind}: every train board poses a test board's puzzle; the build would "
                                         f"have no train split$"):
        build_ppb(tmp_path / "none", kind, source=tmp_path / "planted")
    assert not (tmp_path / "none").exists()


def test_a_ppb_build_refuses_a_negative_seed_before_reading_its_source(tmp_path):
    with pytest.raises(ValueError, match="seed -1"):
        build_ppb(tmp_path / "out", "heyawake", seed=-1, source=tmp_path / "nothing")


def test_a_ppb_board_wrongly_solved_or_malformed_is_refused(tmp_path, fake_ppbench):
    tapa = ppbench_row("tapa", "x", [". 1", ". ."], [click("left", 1, 0)])
    cases = [("nurikabe", ppbench_row("nurikabe", "x", ["3 .", ". ."], []), "breaks the rule 'island-size'"),
             ("heyawake", ppbench_row("heyawake", "x", [". . ."], [], rooms=["0 1 0"]), "not 2 connected regions"),
             ("tapa", {**tapa, "height": 3}, "the board text is 2x2, the row 3x2"),
             ("tapa", {k: v for k, v in tapa.items() if k != "solution_enc"}, "missing field 'solution_enc'")]
    for i, (kind, row, error) in enumerate(cases):       # a golden board: parsed after val is drawn from the fakes
        src = ppb_source(tmp_path / str(i), kind)
        with open(src / "golden_300.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=f"{kind} x: .*{error}"):             # the error names the row
            build_ppb(tmp_path / str(i) / "out", kind, source=src)


def test_without_node_or_ppbench_a_ppb_build_stops_before_reading_its_source(tmp_path, monkeypatch):
    nothing = tmp_path / "nothing"                       # no source files: reading them would raise FileNotFoundError
    build = lambda: build_ppb(tmp_path / "out", "tapa", source=nothing)
    monkeypatch.setattr(ppb.shutil, "which", lambda name: None)
    monkeypatch.delenv("NODE_BIN", raising=False)
    with pytest.raises(ppb.PPBenchUnavailable, match=r"Node.js not found \(node on PATH\)$"):
        build()
    monkeypatch.setenv("NODE_BIN", "/opt/node/bin/node")
    with pytest.raises(ppb.PPBenchUnavailable, match=r"Node.js not found \(\$NODE_BIN=/opt/node/bin/node\)$"):
        build()
    monkeypatch.setattr(ppb.shutil, "which", lambda name: "/usr/bin/node")

    def node_version(answer):                            # what `node --version` prints, or the error it raises
        def run(args, **kw):
            assert kw["timeout"] == 10 and kw["stdin"] is subprocess.DEVNULL
            if isinstance(answer, Exception):
                raise answer
            return subprocess.CompletedProcess(args, 0, answer, "")

        monkeypatch.setattr(ppb.subprocess, "run", run)

    for error in (subprocess.TimeoutExpired(["/usr/bin/node", "--version"], 10), PermissionError(13, "denied")):
        node_version(error)
        with pytest.raises(ppb.PPBenchUnavailable, match=r"/usr/bin/node --version failed: "):
            build()
    node_version("v12.22.12\n")                          # too old: refused before the import, which would hang
    with pytest.raises(ppb.PPBenchUnavailable, match=r"/usr/bin/node --version gives 'v12\.22\.12'$"):
        build()
    node_version("v16.0.0\n")
    monkeypatch.setitem(sys.modules, "ppbench", None)    # import ppbench fails
    with pytest.raises(ppb.PPBenchUnavailable, match=r"pip install ppbench==0\.1\.0.*: import of ppbench halted"):
        build()
    (tmp_path / "ppbench.py").write_text("raise Exception(\"Timed out accessing 'console'\")")  # JSPyBridge's timeout
    monkeypatch.delitem(sys.modules, "ppbench")
    monkeypatch.syspath_prepend(tmp_path)
    with pytest.raises(ppb.PPBenchUnavailable, match=r"pip install ppbench==0\.1\.0.*: Timed out accessing 'console'$"):
        build()


def test_the_lightup_build(tmp_path, fake_ppbench):
    out = tmp_path / "lu"
    build_lightup(out, source=lightup_source(tmp_path / "src"))
    test, train = load_pool(out, "test"), load_pool(out, "train")
    boards = lambda pool: [crop(pool, i, x) for i, x in enumerate(pool.inputs)]
    # test: 2 of t0..t2 (val), in file order, then golden g1 (2x2) and g2 (1x3); train: the third, s1, s2, 8 rows each
    assert len(test) == 4 and len(train) == 24
    np.testing.assert_array_equal(test.is_golden, [0, 0, 1, 1])
    np.testing.assert_array_equal(test.dims, [[10, 10, 1, 1], [10, 10, 1, 1], [2, 2, 1, 1], [1, 3, 1, 1]])
    val = [list(board[9]).index(W0) for board in boards(test)[:2]]           # t_j: its 0-wall is in column j
    assert val == sorted(np.random.default_rng(20260808).choice(3, size=2, replace=False))     # as drawn, sorted
    train_boards = boards(train)
    assert {list(train_boards[0][9]).index(W0), *val} == {0, 1, 2}         # the third t_j, its first row unturned
    assert [b.shape for b in train_boards[8::8]] == [(2, 3), (3, 2)]
    turns = lambda b: [v for k in range(4) for v in (np.rot90(b, k), np.fliplr(np.rot90(b, k)))]
    for b in range(3):                                   # its rows: k = 0..3 turns of np.rot90, each then mirrored
        for got, want in zip(train_boards[8 * b:8 * b + 8], turns(train_boards[8 * b])):
            np.testing.assert_array_equal(got, want)
    # t1m (t1 mirrored), g1 (golden) and s3 (golden g2 turned) are gone: no test board is a symmetry of a train board
    assert not {canonical(b) for b in boards(test)} & {canonical(b) for b in train_boards}
    assert canonical(train_boards[8].reshape(3, 2)) != canonical(train_boards[8])     # s1's bytes in 3x2: no symmetry
    for pool in (test, train):
        task = LightUp(pool)
        assert all(task.valid(i, pool.labels[i]) for i in range(len(pool)))
    solved = crop(test, 0, test.labels[0])
    assert solved[0, 0] == BULB and (solved == BULB).sum() == 1           # the right click at (0, 1) puts no bulb
    # the released dataset's format: uint8 tokens, int32 dims, int64 indices, total_samples, labels encoded, no
    # identifiers.json
    dtype = lambda split, name: np.load(out / split / f"all__{name}.npy").dtype
    for split in ("train", "test"):
        assert dtype(split, "inputs") == dtype(split, "labels") == np.uint8 and dtype(split, "dims") == np.int32
        assert dtype(split, "puzzle_indices") == dtype(split, "group_indices") == np.int64
    assert sorted(p.name for p in out.iterdir()) == ["test", "train"]
    assert (out / "test" / "dataset.json").read_text() == (
        '{"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 9, "seq_len": 676, '
        '"num_puzzle_identifiers": 1, "total_groups": 4, "mean_puzzle_examples": 1.0, "total_puzzles": 4, '
        '"total_samples": 4, "sets": ["all"], "encoded_labels": true}')


def test_a_lightup_board_wrongly_solved_or_of_the_wrong_size_is_refused(tmp_path, fake_ppbench):
    board, bulb = [". .", ". -"], [click("left", 0, 0)]
    cases = {"breaks the Akari rules": ppbench_row("lightup", "x", board, []),           # no bulb: 3 cells unlit
             "does not have 2 lines of 3 cells": {**ppbench_row("lightup", "x", board, bulb), "width": 3},
             "unknown cell '5'": ppbench_row("lightup", "x", [". 5", ". -"], bulb),
             "missing field 'puzzle_url'": {k: v for k, v in ppbench_row("lightup", "x", board, bulb).items()
                                            if k != "puzzle_url"}}
    for error, row in cases.items():
        with pytest.raises(ValueError, match=f"lightup x: .*{error}"):               # the error names the row
            build_lightup(tmp_path / "out", source=ppbench_source(tmp_path / "bad", [row], []))


def test_without_node_the_lightup_build_stops_before_reading_its_source(tmp_path, monkeypatch):
    monkeypatch.setattr(ppb.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Node.js not found"):
        build_lightup(tmp_path / "out", source=tmp_path / "nothing")


def test_the_lightup_build_reads_both_source_files_before_it_parses_a_board(tmp_path, monkeypatch):
    src = lightup_source(tmp_path / "src")
    (src / "golden_300.jsonl").unlink()
    monkeypatch.setattr(ppb, "load_ppbench", lambda: types.SimpleNamespace(Puzzle=None))     # a parse: TypeError
    with pytest.raises(FileNotFoundError, match="golden_300.jsonl"):
        build_lightup(tmp_path / "out", source=src)


def test_a_source_too_small_for_the_splits_is_refused(tmp_path, fake_ppbench):
    """Fewer 10x10 boards than val takes (2 with the fakes), or no board left for train."""
    def keep(src, keys):                                 # the source with only these rows in full_dataset.jsonl
        full = src / "full_dataset.jsonl"
        full.write_text("".join(f"{line}\n" for line in full.read_text().splitlines()
                                if json.loads(line)["sort_key"] in keys))
        return src

    for keys, tens in ((["t0", "s1"], 1), (["t0", "t1"], 2)):
        counts = (f"too small a source: 2 boards outside golden_300.jsonl, {tens} of them 10x10; the build needs 2 "
                  "10x10 boards for val and 1 more board for train")
        with pytest.raises(ValueError, match=f"tapa: {counts}"):
            build_ppb(tmp_path / "out", "tapa", source=keep(ppb_source(tmp_path / f"ppb{tens}", "tapa"), keys))
        with pytest.raises(ValueError, match=f"lightup: {counts}"):
            build_lightup(tmp_path / "out", source=keep(lightup_source(tmp_path / f"lightup{tens}"), keys))


@pytest.mark.parametrize("task", ["sudoku", "maze", "lightup", "nurikabe", "tapa", "heyawake"])
def test_lightup_and_ppb_store_their_labels_encoded_sudoku_and_maze_plain(tmp_path, fake_ppbench, task):
    """Encoded: dataset.json declares it, and most stored tokens are not the labels the loader gives (the build tests
    check those against the rules)."""
    out, sources = tmp_path / "out", {"sudoku": sudoku_source, "maze": maze_source, "lightup": lightup_source}
    TASKS[task].build(out, source=sources.get(task, lambda root: ppb_source(root, task))(tmp_path / "src"))
    encoded = task not in ("sudoku", "maze")
    for split in ("train", "test"):
        meta, labels = json.loads((out / split / "dataset.json").read_text()), load_pool(out, split).labels
        # the share of stored tokens that are not the labels as a plain build stores them
        differ = (np.load(out / split / "all__labels.npy") != np.where(labels == IGNORE, 0, labels)).mean()
        assert meta.get("encoded_labels", False) == encoded and (differ > 0.5 if encoded else differ == 0)


@pytest.mark.data
@pytest.mark.parametrize("task", ["lightup", "nurikabe", "tapa", "heyawake"])
def test_a_ppbench_build_has_the_layout_of_the_released_dataset(tmp_path, fake_ppbench, dataset, task):
    """The files, the arrays' dtypes and ranks and the JSON keys a real rebuild needs to be byte-identical."""
    if task == "lightup":
        build_lightup(tmp_path / "b", source=lightup_source(tmp_path / "src"))
    else:
        build_ppb(tmp_path / "b", task, source=ppb_source(tmp_path / "src", task))

    def layout(root):
        out = {}
        hidden = lambda p: any(part.startswith(".") for part in p.relative_to(root).parts)     # ._*, .DS_Store, ...
        for p in (p for p in root.rglob("*") if p.is_file() and not hidden(p)):
            if p.suffix == ".npy":
                a = np.load(p, mmap_mode="r")
                out[str(p.relative_to(root))] = (a.dtype.str, a.ndim)
            elif p.suffix == ".json" and p.name != "vocab.json":                  # vocab.json's keys are the data's own
                value = json.loads(p.read_text())
                # optional keys: the manifest's seed and dropped, only when set; encoded_labels, which a dataset of
                # plain labels lacks
                if isinstance(value, dict):
                    value = [k for k in value if k not in ("seed", "dropped", "encoded_labels")]
                out[str(p.relative_to(root))] = value
        return out

    assert layout(tmp_path / "b") == layout(dataset(task))
    assert (tmp_path / "b" / "vocab.json").exists() == (dataset(task) / "vocab.json").exists()
