"""Training data (build_verifier_data; python -m ics verifier-data), from the train split only. The boards are its
boards as built, each once and unaugmented (train_boards): Sudoku's first puzzle of each group, every Maze row, and on
the canvas tasks the first of each board's rows (Light-Up's 8, the PPB tasks' 10). The last tenth of the boards is
held out (val), for selection. A board's candidates, each pinned onto the board and kept once: its train label, then
for each solver checkpoint (solver_checkpoints: a solver's last checkpoint, or the snapshots its run kept, weak to
strong) the greedy decode (halt_max_steps segments from a cold start), `hypotheses` re-decodes of the input with one
hypothesis written in, as ICS Stage B proposes them (ics/methods/ics.py), and the decodes of the board's other
restatements, mapped back. A candidate's target is the task's rule checker on it: 1 valid, 0 not. The data is in the
TRM layout (ics/data.py), splits train and val: inputs are the candidates, labels [N] their targets, the valid
candidates one group and the invalid ones another, so a batch of an even number of rows is half valid and half invalid.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ics.builders import overlap, token_dtype, write_split
from ics.checkpoint import CONFIG_FILE, WEIGHTS_FILE, model_section, sha256
from ics.data import Pool, check_dims, dataset_dims, load_pool
from ics.methods.ics import propose
from ics.registry import TASKS
from ics.tasks import make_task
from ics.tasks.base import Task
from ics.tasks.ppb import KINDS, decided_tokens
from ics.trm.model import TRMConfig, load_trm
from ics.trm.roll import roll

VAL_FRACTION = 0.1                      # the share of the train boards held out for selection
SOURCES = ("greedy", "hypotheses", "restatements")        # a candidate's source, by its number in decodes
WINDOW = 1024                           # boards a checkpoint decodes together (bounds the memory of their decodes)


def solver_checkpoints(paths, snapshots: int = 10) -> list[Path]:
    """The solver checkpoints `paths` name: a release checkpoint stands for itself, a training run directory for
    `snapshots` of the checkpoints it kept (snapshots/step_<update>), spread evenly by step from the last back to the
    first (1: the last), or with snapshots 0 for its last checkpoint (last/) alone."""
    out = []
    for path in map(Path, paths):
        if not path.is_dir():
            raise FileNotFoundError(f"{path}: no such directory")
        kept = {int(d.name.removeprefix("step_")): d for d in (path / "snapshots").glob("step_*")}
        if (path / CONFIG_FILE).exists():
            out.append(path)
        elif snapshots == 0 and (path / "last" / CONFIG_FILE).exists():
            out.append(path / "last")
        elif snapshots > 0 and kept:
            left, taken = sorted(kept), []
            for target in np.linspace(left[-1], left[0], min(snapshots, len(left))):
                taken.append(left.pop(int(np.abs(np.array(left) - target).argmin())))
            out += [kept[step] for step in sorted(taken)]
        elif snapshots == 0:
            raise ValueError(f"{path} is neither a checkpoint nor a training run with a last checkpoint (last/)")
        else:
            raise ValueError(f"{path} is neither a checkpoint nor a training run with snapshots (keep them with --set "
                             f"train.keep_every_eval=true, or take the run's last checkpoint alone: --snapshots 0)")
    return out


def placed(pool: Pool) -> Pool:
    """`pool` with each board's placement (h, w, 1, 1), read off its ring of walls, where its split stores none (the PPB
    train splits)."""
    side = round(pool.inputs.shape[1] ** 0.5)
    used = pool.inputs.reshape(-1, side, side) != 0
    h, w = used.any(2).sum(1) - 2, used.any(1).sum(1) - 2
    return dataclasses.replace(pool, dims=np.stack([h, w, np.ones_like(h), np.ones_like(w)], 1))


def decodes(task: Task, solver, T: int, hypotheses: int, batch: int,
            boards=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A solver's candidates for the boards `boards` of `task` (indices, by default all), unpinned, by source
    (SOURCES): 0, each board's greedy decode (T segments from a cold start); 1, its hypothesis re-decodes; 2, its
    decodes of each of its other restatements, mapped back (a decode that cannot be mapped back is dropped). Returns
    (board, decode, source) per candidate."""
    idx = list(range(len(task))) if boards is None else [int(i) for i in boards]
    X = task.X[idx]
    decode = lambda rows: roll(solver, np.stack(rows), T, batch, topk=1).dec if rows else np.zeros((0, X.shape[1]))
    r = roll(solver, X, T, batch)                                   # the top-k of Stage B (roll's default k)
    hinted = []                                                     # (board, input with one hint)
    for j, i in enumerate(idx):
        for c, tok in propose(task, i, X[j], task.pin(i, r.dec[j]), r.top_ids[j], r.top_vals[j], hypotheses, 1):
            row = X[j].copy()
            row[c] = tok
            hinted.append((i, row))
    restated = [(i, tag, row) for i, x in zip(idx, X) for tag, row in task.restatements(i, x)
                if not np.array_equal(row, x)]
    mapped = [(i, task.restate_back(i, tag, y)) for (i, tag, _), y in zip(restated, decode([z for *_, z in restated]))]
    mapped = [(i, y) for i, y in mapped if y is not None]
    owners = [*idx, *(i for i, _ in hinted), *(i for i, _ in mapped)]
    ys = [r.dec, decode([row for _, row in hinted]), np.reshape([y for _, y in mapped], (-1, X.shape[1]))]
    return (np.array(owners, np.int64), np.concatenate(ys).astype(np.int64),
            np.repeat([0, 1, 2], [len(idx), len(hinted), len(mapped)]))


def train_boards(task: str, data, boards: int | None = None) -> Pool:
    """The verifier's boards, placed: the first `boards` (by default all) of the train split's boards, each once and as
    given; a board is a group of puzzles, or TaskEntry.board_rows rows (Light-Up 8, the PPB tasks 10). Refused: a split
    that is not a whole number of such boards, and one of fewer than 2 boards (a train board and a val board)."""
    d, k = Path(data) / "train", TASKS[task].board_rows
    if k:                                                       # every row its own group: board b is k rows from b*k
        total = len(np.load(d / "all__inputs.npy", mmap_mode="r"))
        if total % k:
            raise ValueError(f"{data}: a {task} train split, as python -m ics data writes it, holds {k} rows per "
                             f"board; this one holds {total} rows")
        rows = np.arange(0, total, k)
    else:                                                       # a group is a board: its first puzzle
        rows = np.load(d / "all__group_indices.npy").astype(np.int64)[:-1]
    if len(rows) < 2:
        raise ValueError(f"{data}: the train split holds {len(rows)} boards; the verifier's data needs at least 2 (a "
                         f"train board and a val board)")
    pool = load_pool(data, "train", rows=rows[:boards])
    if pool.dims is None and k:
        pool = placed(pool)
    if task in KINDS and (decided := np.isin(pool.inputs, decided_tokens(task, pool.vocab)).any(1)).any():
        raise ValueError(f"{data}: row {pool.index[decided.argmax()]} of the {task} train split, a board's first, has "
                         f"decided cells; a {task} train split, as python -m ics data writes it, holds {k} rows per "
                         f"board")
    return pool


def build_verifier_data(task: str, data, solvers, out, snapshots: int = 10, hypotheses: int = 4,
                        boards: int | None = None, batch: int = 256, device="cpu") -> dict:
    """Write the verifier's training data for `task` from the train split of `data` and the checkpoints `solvers` name
    (solver_checkpoints) into `out`; returns its summary, also in each split's dataset.json under "verifier". Nothing
    is written unless each split holds valid and invalid candidates."""
    if snapshots < 0 or hypotheses < 0 or batch < 1 or (boards is not None and boards < 2):
        raise ValueError(f"build_verifier_data needs snapshots >= 0, hypotheses >= 0, batch >= 1 and boards >= 2, got "
                         f"snapshots={snapshots}, hypotheses={hypotheses}, batch={batch}, boards={boards}")
    if why := overlap(out, data, ("out", "data")):
        raise ValueError(why)
    checkpoints, dims = solver_checkpoints(solvers, snapshots), dataset_dims(data, "train")
    for ckpt in checkpoints:
        check_dims(ckpt, model_section(ckpt, TRMConfig), dims)
    pool = train_boards(task, data, boards)
    t, n, tokens = make_task(task, pool), len(pool), token_dtype(pool.meta["vocab_size"])
    found = [{} for _ in range(n)]          # per board: pinned candidate (bytes, in the split's tokens) -> target

    def offer(owners, ys) -> list[int]:
        """Each candidate onto its board; returns the [valid, invalid] counts of the offered candidates."""
        tally = [0, 0]
        for i, y in zip(owners, ys):
            c = t.pin(int(i), y)
            key = c.astype(tokens).tobytes()
            if key not in found[i]:
                found[i][key] = int(t.valid(int(i), c))
            tally[1 - found[i][key]] += 1
        return tally

    summary = {"task": task, "boards": n, "val_boards": max(1, round(VAL_FRACTION * n)), "hypotheses": hypotheses,
               "train_label": offer(range(n), t.Y), "solvers": []}
    t0 = time.time()
    for ckpt in checkpoints:
        solver = load_trm(ckpt, device)
        tally = np.zeros((len(SOURCES), 2), np.int64)
        for a in range(0, n, WINDOW):
            owners, ys, source = decodes(t, solver, solver.config.halt_max_steps, hypotheses, batch,
                                         range(a, min(a + WINDOW, n)))
            tally += [offer(owners[source == s], ys[source == s]) for s in range(len(SOURCES))]
        provenance = json.loads((ckpt / CONFIG_FILE).read_text())["provenance"]
        summary["solvers"].append({"checkpoint": ckpt.name, "step": provenance.get("step"),
                                   "sha256": sha256(ckpt / WEIGHTS_FILE), **dict(zip(SOURCES, tally.tolist()))})
        print(json.dumps({**summary["solvers"][-1], "seconds": round(time.time() - t0, 1)}), flush=True)
        del solver
    cut, splits = n - summary["val_boards"], {}
    for split, part in (("train", range(cut)), ("val", range(cut, n))):
        valid, invalid = ([key for i in part for key, ok in found[i].items() if ok == target] for target in (1, 0))
        if not valid:
            raise ValueError(f"the labels of the {split} boards all fail the {task} rule checker, as every other "
                             f"candidate does: the data {data} does not match the task {task}")
        if not invalid:
            raise ValueError(f"the {split} boards' candidates are all valid; a verifier needs both (more or weaker "
                             f"solver checkpoints, or more hypotheses)")
        splits[split] = valid, invalid
        summary[split] = {"candidates": len(valid) + len(invalid), "valid": len(valid)}
    for split, (valid, invalid) in splits.items():
        inputs = np.frombuffer(b"".join(valid + invalid), tokens).reshape(len(valid) + len(invalid), -1)
        write_split(out, split, inputs, np.repeat([1, 0], [len(valid), len(invalid)]),
                    np.array([len(valid), len(invalid)]), pool.meta["vocab_size"], ignore_label_id=None,
                    verifier=summary)
    return summary
