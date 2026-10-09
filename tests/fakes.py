"""Test stand-ins and helpers:
- Scripted, and ScriptedTRM, ScriptedSampler, ScriptedRestarts and ScriptedAttractor on it: models with the interfaces
  of TRM, GRAM, EqR and the Attractor whose answers a script gives;
- assert_identical, and digest, a golden value for a test to pin;
- small boards and datasets: make_pool, SOLUTION, sudoku_pool, maze_pool, solve_maze, write_trm_split, sudoku_dataset;
- the dataset builders' source files: sudoku_source, maze_source, lightup_source, ppb_source;
- tiny_trm_checkpoint, the training tests' tiny recipes (TINY_*), and interrupted, a run killed mid-update."""
import base64
import csv
import hashlib
import json
import types
from collections import deque

import numpy as np
import pytest
import torch

import ics.train
from ics.checkpoint import save_checkpoint
from ics.data import Pool, encode_labels
from ics.trm.model import TRM, TRMConfig


class Scripted(torch.nn.Module):
    """What the scripted models share: a buffer that places them on a device, and zero latents for a batch."""

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1))

    @property
    def device(self):
        return self.anchor.device

    def initial_state(self, batch_size):
        z = torch.zeros(batch_size, 1, device=self.device)
        return z, z.clone()

    def one_hot(self, answers) -> torch.Tensor:
        """Logits that rank each cell's answer token first."""
        return torch.nn.functional.one_hot(torch.as_tensor(np.stack(answers)), self.vocab).float()


class ScriptedTRM(Scripted):
    """Has TRM's rolling interface. After segment t, row `row` decodes to script(row, t, zl)[0] with halting logit
    script(...)[1]; zl is the row's scalar z_L (noise hooks add to it). The answer token of a cell gets logit `margin`
    (script(...)[2], scalar or per cell); every other token v gets others[v] (by default -0.01 * v: lower ids rank
    higher)."""

    def __init__(self, vocab: int, script, others=None):
        super().__init__()
        self.vocab, self.script = vocab, script
        self.others = (-0.01 * torch.arange(vocab, dtype=torch.float32) if others is None
                       else torch.tensor(others, dtype=torch.float32))

    def segment(self, z_H, z_L, inputs, puzzle_identifiers):
        t = int(z_H[0, 0].item()) + 1
        answers, qs, margins = [], [], []
        for row, zl in zip(inputs.cpu().numpy().astype(np.int64), z_L[:, 0].cpu().numpy()):
            a, q, m = self.script(row, t, float(zl))
            answers.append(a)
            qs.append(q)
            margins.append(np.broadcast_to(np.asarray(m, np.float32), a.shape))
        A = torch.as_tensor(np.stack(answers), dtype=torch.long)
        logits = self.others.expand(*A.shape, self.vocab).clone()
        logits.scatter_(-1, A.unsqueeze(-1), torch.as_tensor(np.stack(margins)).unsqueeze(-1))
        return z_H + 1, z_L, logits, torch.tensor(qs, dtype=torch.float32)


class ScriptedSampler(Scripted):
    """GRAM's sampling interface. Each initial_state call starts the next sample i (counted modulo N); after a step, a
    row with input `row` decodes to script(row, i, gen)[0] with LPRM value logit script(row, i, gen)[1], gen the
    sample's generator. Every step's (batch size, gen) is recorded in `calls`."""

    def __init__(self, vocab: int, script, N: int = 1):
        super().__init__()
        self.vocab, self.script, self.N, self.calls, self.i = vocab, script, N, [], -1

    def initial_state(self, batch_size):
        self.i = (self.i + 1) % self.N
        return super().initial_state(batch_size)

    def step(self, z_H, z_L, inputs, gen):
        self.calls.append((inputs.shape[0], gen))
        answers, values = zip(*(self.script(row, self.i, gen) for row in inputs.cpu().numpy().astype(np.int64)))
        return z_H, z_L, self.one_hot(answers), torch.zeros(len(values)), torch.tensor(values, dtype=torch.float32)


class ScriptedRestarts(Scripted):
    """EqR's restart interface. Rows arrive board-major, R restarts per board; after a step, restart r of a board with
    input `row` decodes to script(row, r)[0] with q_halt script(row, r)[1]. Every call's generator is recorded in
    `gens`, and every step's row count in `rows`."""

    def __init__(self, vocab: int, R: int, script):
        super().__init__()
        self.vocab, self.R, self.script, self.gens, self.rows = vocab, R, script, [], []

    def initial_state(self, batch_size, gen):
        self.gens.append(gen)
        return super().initial_state(batch_size)

    def step(self, z_H, z_L, inputs, gen):
        self.gens.append(gen)
        self.rows.append(inputs.shape[0])
        rows = inputs.cpu().numpy().astype(np.int64)
        answers, qs = zip(*(self.script(row, j % self.R) for j, row in enumerate(rows)))
        return z_H, z_L, self.one_hot(answers), torch.tensor(qs, dtype=torch.float32)


class ScriptedAttractor(Scripted):
    """Attractor's restart interface. Each initial_state call starts the next restart k (counted modulo R); after a
    segment, restart k of a board with input `row` decodes to script(row, k)[0] with q_halt script(row, k)[1]. Every
    initial_state call's (batch size, dH, dL) is recorded in `starts`, and every segment's batch size in `rows`."""

    def __init__(self, vocab: int, R: int, script, hidden: int = 4):
        super().__init__()
        self.vocab, self.R, self.script, self.starts, self.rows, self.k = vocab, R, script, [], [], -1
        self.config = types.SimpleNamespace(hidden_size=hidden)

    def initial_state(self, batch_size, dH=None, dL=None):
        self.k = (self.k + 1) % self.R
        self.starts.append((batch_size, dH, dL))
        return super().initial_state(batch_size)

    def segment(self, z_H, z_L, inputs, puzzle_identifiers):
        self.rows.append(inputs.shape[0])
        answers, qs = zip(*(self.script(row, self.k) for row in inputs.cpu().numpy().astype(np.int64)))
        return z_H, z_L, self.one_hot(answers), torch.tensor(qs, dtype=torch.float32)


def assert_identical(got, want):
    """`got` is `want` in kind as in value: dicts key by key, lists and tuples item by item, arrays with their dtype,
    shape and layout, tensors with their dtype and shape (torch.equal), anything else with its type (True is not 1, nor
    is np.int64(1))."""
    if isinstance(want, dict):
        assert type(got) is type(want) and got.keys() == want.keys(), (got, want)
        for k in want:
            assert_identical(got[k], want[k])
    elif isinstance(want, (list, tuple)):
        assert type(got) is type(want) and len(got) == len(want), (got, want)
        for a, b in zip(got, want):
            assert_identical(a, b)
    elif isinstance(want, np.ndarray):
        assert type(got) is np.ndarray and (got.dtype, got.shape) == (want.dtype, want.shape), (got, want)
        assert got.flags.c_contiguous == want.flags.c_contiguous and np.array_equal(got, want), (got, want)
    elif isinstance(want, torch.Tensor):
        assert type(got) is torch.Tensor and (got.dtype, got.shape) == (want.dtype, want.shape), (got, want)
        assert torch.equal(got, want), (got, want)
    else:
        assert type(got) is type(want) and got == want, (got, want)


def digest(arrays) -> str:
    """sha256 over arrays, each with its dtype and shape: a golden value for a test to pin."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def make_pool(X, Y, dims=None, vocab=None):
    X = np.asarray(X, np.int64)
    return Pool(X, np.asarray(Y, np.int64), np.arange(len(X), dtype=np.int64), {}, dims, None, vocab)


# A valid Sudoku (digits 1..9 as tokens 2..10): row r is the digit sequence shifted by 3r + r // 3.
SOLUTION = np.array([(r * 3 + r // 3 + c) % 9 + 2 for r in range(9) for c in range(9)], np.int64)


def sudoku_pool(hole_sets):
    """One board per set of blanked cells (token 1); every board has the solution SOLUTION."""
    X = []
    for holes in hole_sets:
        x = SOLUTION.copy()
        x[list(holes)] = 1
        X.append(x)
    return make_pool(X, [SOLUTION] * len(X))


def maze_pool(open_rows, start, goal):
    """One 30x30 maze: walls everywhere except the rectangle open_rows x columns 1..10; start/goal are (r, c)."""
    g = np.ones((30, 30), np.int64)
    for r in open_rows:
        g[r, 1:11] = 2
    g[start] = 3
    g[goal] = 4
    x = g.reshape(-1)
    return make_pool([x], [solve_maze(x)])


def solve_maze(row):
    """The row with PATH (5) on one BFS shortest start-goal path (neighbour order: up, down, left, right)."""
    walls = row == 1
    s, g = int(np.flatnonzero(row == 3)[0]), int(np.flatnonzero(row == 4)[0])
    prev, q = {s: None}, deque([s])
    while q:
        c = q.popleft()
        r, co = divmod(c, 30)
        for nr, nc in ((r - 1, co), (r + 1, co), (r, co - 1), (r, co + 1)):
            n = nr * 30 + nc
            if 0 <= nr < 30 and 0 <= nc < 30 and not walls[n] and n not in prev:
                prev[n] = c
                q.append(n)
    out = row.copy()
    c = prev[g]
    while c != s:
        out[c] = 5
        c = prev[c]
    return out


def write_trm_split(root, split, inputs, labels, group_sizes=None, index_dtype=np.int32, encoded=False):
    """One split of a dataset in the TRM layout (ics/data.py): a puzzle per row, group g holding group_sizes[g]
    consecutive puzzles (default: one each), label 0 ignored; `encoded`: the labels stored encoded, as dataset.json
    then declares."""
    d = root / split
    d.mkdir(parents=True)
    n = len(inputs)
    sizes = np.ones(n, np.int64) if group_sizes is None else np.asarray(group_sizes)
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 11,
            "seq_len": np.shape(inputs)[1], "num_puzzle_identifiers": 1, "total_groups": len(sizes),
            "mean_puzzle_examples": 1.0, "total_puzzles": n, "sets": ["all"],
            **({"encoded_labels": True} if encoded else {})}
    (d / "dataset.json").write_text(json.dumps(meta))
    np.save(d / "all__inputs.npy", np.asarray(inputs, np.uint8))
    labels = np.asarray(labels, np.uint8)
    np.save(d / "all__labels.npy", encode_labels(labels, 11) if encoded else labels)
    np.save(d / "all__puzzle_identifiers.npy", np.zeros(n, np.int32))
    np.save(d / "all__puzzle_indices.npy", np.arange(n + 1, dtype=index_dtype))
    np.save(d / "all__group_indices.npy", np.concatenate([[0], np.cumsum(sizes)]).astype(index_dtype))


def sudoku_dataset(root, groups=8, per_group=2, test=6, seed=0):
    """A small Sudoku dataset: boards of SOLUTION with random blanks, `groups` train groups of `per_group` boards and
    `test` test boards."""
    rng = np.random.default_rng(seed)
    boards = np.where(rng.random((groups * per_group + test, 81)) < 0.5, 1, SOLUTION)
    write_trm_split(root, "train", boards[test:], np.tile(SOLUTION, (groups * per_group, 1)), [per_group] * groups)
    write_trm_split(root, "test", boards[:test], np.tile(SOLUTION, (test, 1)))
    return root


DIGITS = "".join(str(t - 1) for t in SOLUTION)                    # a solved grid as a CSV answer


def write_csv(path, rows):
    """A source CSV of the builders: a header, then rows of (source, question, answer, rating)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["source", "question", "answer", "rating"])
        out.writerows(rows)


def sudoku_source(root, train=6, test=3):
    """train.csv and test.csv of Sudoku boards: SOLUTION with about 40% of the cells blank ('.')."""
    rng = np.random.default_rng(0)
    board = lambda: ["src", "".join(c if rng.random() < 0.6 else "." for c in DIGITS), DIGITS, "0"]
    write_csv(root / "train.csv", [board() for _ in range(train)])
    write_csv(root / "test.csv", [board() for _ in range(test)])
    return root


def maze_source(root, train=2, test=1):
    """train.csv and test.csv of one 30x30 maze, '# SG' (wall, empty, start, goal) repeated, with its path '#oSG'."""
    row = ["s", "# SG" * 225, "#oSG" * 225, "0"]
    write_csv(root / "train.csv", [row] * train)
    write_csv(root / "test.csv", [row] * test)
    return root


def click(button: str, r: int, c: int) -> str:
    """A PPBench move clicking cell (r, c)."""
    return f"mouse,{button},{2 * c + 1},{2 * r + 1}"


def ppbench_row(pid: str, key: str, grid: list[str], moves: list[str], rooms: list[str] | None = None) -> dict:
    """A PPBench row of the board whose cell lines are `grid` (Heyawake: its clue lines, after the room lines `rooms`).
    Its puzzle_url is the board text itself, which the fake_ppbench fixture's Puzzle gives back."""
    h, w = len(grid), len(grid[0].split())
    lines = ["pzprv3", pid, str(h), str(w)]
    if rooms is not None:
        lines += [str(1 + max(int(t) for row in rooms for t in row.split())), *rooms]
    raw = json.dumps({"moves_full": moves}).encode()
    enc = base64.b64encode(bytes(b ^ b"ppbench"[i % 7] for i, b in enumerate(raw))).decode()
    return {"pid": pid, "sort_key": key, "width": w, "height": h, "puzzle_url": "\n".join(lines + grid) + "\n",
            "solution_enc": enc}


def ppbench_source(root, pool: list[dict], golden: list[dict]):
    """full_dataset.jsonl (the pool's rows, then a sudoku row) and golden_300.jsonl (the golden rows, then the same)."""
    root.mkdir(parents=True, exist_ok=True)
    other = ppbench_row("sudoku", "other", [". ."], [])
    for name, rows in (("full_dataset.jsonl", pool), ("golden_300.jsonl", golden)):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in [*rows, other]))
    return root


def lightup_source(root):
    """PPBench files of Light-Up boards. The pool, in file order: three 10x10 boards "t0".."t2" (an empty top row, walls
    below, a wall numbered 0 at (9, j); a bulb at (0, 0), and a right click at (0, 1)), "t1m", t1 mirrored (a repeat);
    "s1" (2x3, the bulb next to a 1) and "s2" (3x2, and a click off the board); "g1" (2x2), also golden; "s3" (3x1), a
    turn of golden "g2" (1x3); "wide" (1x25, too wide). Golden: g1, g2 and the wide board."""
    row = lambda key, grid, *bulbs: ppbench_row("lightup", key, grid, [click("left", r, c) for r, c in bulbs])
    ten = lambda j: [". " * 10] + ["- " * 10] * 8 + [" ".join("0" if c == j else "-" for c in range(10))]
    tens = [ppbench_row("lightup", f"t{j}", ten(j), [click("left", 0, 0), click("right", 0, 1)]) for j in range(3)]
    g1, wide = row("g1", [". .", ". -"], (0, 0)), row("wide", [" ".join("." * 25)], (0, 0))
    pool = [*tens, row("t1m", ten(8), (0, 9)), row("s1", [". . .", "- 1 -"], (0, 1)),
            row("s2", [". -"] * 3, (0, 0), (5, 5)), g1, row("s3", [".", ".", "-"], (0, 0)), wide]
    return ppbench_source(root, pool, [g1, row("g2", ["- . ."], (0, 2)), wide])


def ppb_source(root, kind: str):
    """PPBench files of one shading puzzle type. The pool, in file order: three 10x10 boards "t0".."t2" (a clue in
    cell (0, j), the corner (9, 9) shaded; Heyawake's also a room of the cells (9, 0..j)), boards "s1" and "s2" whose
    solutions take several moves (s2's last one is off the board), "g1", also golden, and "wide" (1x25, too wide).
    Golden: g1, "g2" and the wide board. No train board poses a test board's puzzle (the build would drop it). Only
    Nurikabe's s1 clicks a cell twice: (0, 1) left then right, (1, 1) left twice; its label is valid only if the last
    click wins (not the first, and not cycling through the states)."""
    left = lambda *cells: [click("left", r, c) for r, c in cells]
    right = lambda *cells: [click("right", r, c) for r, c in cells]
    clue = {"nurikabe": "-", "tapa": "0", "heyawake": "1"}[kind]
    ten = lambda j: [" ".join(clue if c == j else "." for c in range(10))] + [". " * 10] * 9
    corner = lambda j: [" ".join("0" * 10)] * 9 + [" ".join("1" * (j + 1) + "0" * (9 - j))]
    if kind == "heyawake":                          # one room (s1, t<j>: two), its clue the number of shaded cells
        def row(key, grid, moves, rooms=None):
            return ppbench_row(kind, key, grid, moves, rooms or [" ".join("0" * len(grid[0].split()))] * len(grid))
        s1 = row("s1", ["1 . .", ". . ."], left((1, 1)) + right((0, 0), (0, 2)), rooms=["0 0 1", "0 0 1"])
        s2 = row("s2", [". . .", ". 2 ."], left((0, 0), (0, 2)) + right((1, 1)) + left((4, 4)))
        g1, g2 = row("g1", ["1 .", ". ."], left((1, 1))), row("g2", [". .", ". ."], right((0, 0)))
    else:
        row = lambda key, grid, moves, rooms=None: ppbench_row(kind, key, grid, moves)
        if kind == "nurikabe":                      # s1, s2: an island of 2 (seas of 4 and 2); g1: an island of 3
            s1 = row("s1", ["2 . .", ". . ."],
                     left((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)) + right((0, 1)) + left((1, 1)))
            s2 = row("s2", ["2 .", ". ."], right((0, 1)) + left((1, 0), (1, 1), (4, 4)))
            g1, g2 = row("g1", [". 3", ". ."], left((1, 0)) + right((0, 0))), row("g2", ["- .", ". ."], [])
        else:                                       # s1: two runs around "1,-"; s2 (any), g1: a run of 1
            s1 = row("s1", ["1,- . .", ". . .", ". . ."],
                     left((0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)) + right((1, 1)))
            s2 = row("s2", ["- .", ". ."], left((1, 1)) + right((0, 1)) + left((4, 4)))
            g1, g2 = row("g1", [". 1", ". ."], left((1, 0))), row("g2", ["0 .", ". ."], right((1, 1)))
    tens = [row(f"t{j}", ten(j), left((9, 9)) + right((0, 0)), corner(j)) for j in range(3)]
    wide = row("wide", [" ".join("." * 25)], [])
    return ppbench_source(root, [*tens, s1, s2, g1, wide], [g1, g2, wide])


# A tiny TRM recipe for training tests: on sudoku_dataset's 8 groups of 2 boards, global batch 4 gives 2 updates per
# epoch, 6 in all, in blocks of 2 epochs (4 batches); evaluations after updates 3 and 6.
TINY_RECIPE = {"model": {"hidden_size": 32, "num_heads": 2, "expansion": 2.0, "H_cycles": 2, "L_cycles": 2,
                         "L_layers": 1, "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "forward_dtype": "float32"},
               "train": {"global_batch": 4, "epochs": 3, "epochs_per_block": 2, "lr_warmup": 2, "eval_every": 3,
                         "log_every": 2}}


def tiny_trm_checkpoint(out, seq_len=81, vocab_size=11, seed=0, **kw):
    """A random tiny TRM release checkpoint in `out` (forward in float32), for the data of sudoku_dataset by default."""
    torch.manual_seed(seed)
    cfg = TRMConfig(**{"seq_len": seq_len, "vocab_size": vocab_size, "puzzle_emb_ndim": 16, "puzzle_emb_len": 1,
                       "H_cycles": 1, "L_cycles": 1, "L_layers": 1, "hidden_size": 16, "num_heads": 2,
                       "expansion": 2.0, "halt_max_steps": 2, "forward_dtype": "float32", **kw})
    save_checkpoint(TRM(cfg).state_dict(), cfg, out)
    return out


# The verifier's recipe shrunk for tests (ics/configs/train/verifier.yaml): 4 updates of 4 rows, 2 segments each, an
# evaluation after updates 2 and 4.
TINY_VERIFIER = {"model": {"halt_max_steps": 2},
                 "train": {"global_batch": 4, "updates": 4, "eval_every": 2, "log_every": 1}}


# A tiny EqR recipe for training tests (ics/configs/train/eqr.yaml): on sudoku_dataset's 8 groups, global batch 4 and 6
# updates in blocks of 2 epochs (4 batches); evaluations after updates 3 and 6 on 4 test boards; every row halts at its
# 2nd step, so all rows restart, drawing fresh latents, at updates 1, 3 and 5.
TINY_EQR = {"model": {"hidden_size": 32, "num_heads": 2, "expansion": 2.0, "H_cycles": 2, "L_cycles": 2, "L_layers": 1,
                      "forward_dtype": "float32", "halt_max_steps": 2},
            "train": {"global_batch": 4, "updates": 6, "epochs_per_block": 2, "lr_warmup": 2, "eval_every": 3,
                      "eval_boards": 4, "log_every": 2}}


# A tiny GRAM recipe for training tests (ics/configs/train/gram.yaml): on sudoku_dataset's 8 groups, global batch 4
# gives 2 updates per epoch, 6 in all, in blocks of 2 epochs (4 batches); evaluations after updates 3 and 6 on 4 test
# boards. Every row runs 2 supervision steps: all rows restart at updates 1, 3 and 5, and the cycles' last steps, 2, 4
# and 6, add the deferred LPRM; the save after update 3 holds a cycle's first step.
TINY_GRAM = {"model": {"hidden_size": 32, "num_heads": 2, "expansion": 2.0, "L_cycles": 2, "L_layers": 1,
                       "puzzle_emb_len": 2, "forward_dtype": "float32", "halt_max_steps": 2},
             "train": {"global_batch": 4, "epochs": 3, "epochs_per_block": 2, "lr_warmup": 2, "eval_every": 3,
                       "eval_boards": 4, "log_every": 2}}


# A tiny Attractor recipe for training tests (ics/configs/train/attractor.yaml): on sudoku_dataset's 8 groups, global
# batch 4 in one micro-batch of 4 rows (one solver call) and 6 updates in blocks of 2 epochs (4 batches); evaluations
# after updates 3 and 6 on 4 test boards; every row halts at its 2nd segment; solves of 3 or 4 map evaluations.
TINY_ATTRACTOR = {"model": {"hidden_size": 32, "num_heads": 2, "expansion": 2.0, "H_cycles": 2, "L_layers": 1,
                            "puzzle_emb_ndim": 32, "puzzle_emb_len": 2, "forward_dtype": "float32",
                            "halt_max_steps": 2, "deq_max_iter": 4, "deq_min_iter": 3},
                  "train": {"global_batch": 4, "micro_batch": 4, "updates": 6, "epochs_per_block": 2, "lr_warmup": 2,
                            "eval_every": 3, "eval_boards": 4, "log_every": 2}}


class Stop(Exception):
    """A run killed mid-update (stop), in the training tests."""


def stop(*args):
    raise Stop


def interrupted(run, tmp_path, monkeypatch, name: str, update: int, **over) -> None:
    """run(tmp_path, name, **over) until it dies in `update`, after the update's backward and before its optimizers
    step."""
    real = ics.train.lr_at
    monkeypatch.setattr(ics.train, "lr_at", lambda step, *args: stop() if step == update else real(step, *args))
    with pytest.raises(Stop):
        run(tmp_path, name, **over)
    monkeypatch.setattr(ics.train, "lr_at", real)
