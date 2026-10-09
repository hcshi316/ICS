"""Input-Conditioned Search (ICS) on a frozen TRM.

ICS never changes the weights or the latent state: it changes the input the recursive update is conditioned on. Stage P
continues the roll on a board's original input; every other roll starts from the model's initial state.
  A  roll every board on its original input for T_greedy segments
  P  patience: continue the still-open boards' roll on their original input to each depth in `patience` (reaching depth
     d rolls d minus the previous depth more segments; the first depth's previous depth is T_greedy)
  B  hypotheses, for up to `levels` rounds: for each parent, write the model's own alternative tokens into its least
     confident editable cells (smallest gap between its two best logits first; one cell per hypothesis, stacked on the
     parent's earlier hints), roll each hinted input for T segments, rank the children and keep the best `beam` as the
     next parents. At most `node_budget` hinted rolls per board over all rounds. A board's root parent is Stage A's
     pinned answer, with the top-k token ids and logits of its roll on the original input at the last depth it reached.
  C  restatements: roll every still-open board under each of its restatements for T segments and map the decode back.
Every decode is pinned (the given cells restored) before a terminal sees it. Terminals decide when a board commits:
  cert     the task certificate accepts the pinned decode
  q        q_halt > 0 and the pinned decode is consistent (below). Boards still open after C commit the answer that at
           least `agree` restatements agree on, else the restatement answer with the largest q_halt.
  confirm  a candidate becomes the board's incumbent when its q_halt beats the incumbent's; each new incumbent is
           re-derived once from the next restatement of the input it came from, and the board locks when the pinned
           re-derivation reproduces it. Boards still open after C commit their incumbent.
Children are ranked by (hint cells the decode overwrote, 1 if the raw decode is not consistent, -q_halt), ties in
proposal order. A decode y of board i is consistent when Task.consistent(i, y) holds: pin(i, y) == y, i.e. y keeps
every given cell and holds no token pin clears; the PPB tasks check only their clue cells and Heyawake's room structure.
segs[i] counts every segment rolled for board i, refuted branches included.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import torch

from ics.methods import ANSWER_DTYPE
from ics.tasks.base import SCORINGS, Task
from ics.trm.roll import roll

TERMINALS = ("cert", "q", "confirm")


@dataclass
class ICSConfig:
    terminal: str
    T: int = 16
    T_greedy: int = 20
    patience: tuple[int, ...] = (64, 256)   # Stage P: the depths the roll on x continues to (increasing, > T_greedy)
    levels: int = 6
    beam: int = 4
    cells: int = 48              # hypotheses proposed per parent
    alts: int = 2                # alternative tokens per cell
    node_budget: int = 96        # hinted rolls per board, all levels
    agree: int = 2               # q terminal: restatements that must agree after C
    batch: int = 252

    def __post_init__(self):
        if self.terminal not in TERMINALS:
            raise ValueError(f"unknown terminal {self.terminal!r}; expected one of {TERMINALS}")
        for name in ("T", "T_greedy", "beam", "cells", "alts", "node_budget", "agree", "batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"ICSConfig needs {name} >= 1, got {name}={getattr(self, name)}")
        if self.levels < 0:
            raise ValueError(f"ICSConfig needs levels >= 0, got levels={self.levels}")
        if any(deeper <= depth for depth, deeper in pairwise((self.T_greedy, *self.patience))):
            raise ValueError(f"ICSConfig needs strictly increasing patience depths above T_greedy={self.T_greedy}, "
                             f"got patience={self.patience}")

    @classmethod
    def from_settings(cls, settings: dict) -> ICSConfig:
        """The config of a regime's settings (ics/configs/eval/ics.yaml); keys that are not fields are dropped."""
        kw = {f.name: settings[f.name] for f in fields(cls) if f.name in settings}
        if "patience" in kw:
            kw["patience"] = tuple(kw["patience"])
        return cls(**kw)


class Node(NamedTuple):
    """A node of Stage B: hints written into a board's input, and the roll of that hinted input. A root node has no
    hints, Stage A's pinned answer and the top-k of the roll on the board's input at the last depth it reached (T_greedy
    or a patience depth); only children have the last three fields."""
    stack: tuple                    # the hints ((cell, token), ...), in the order they were written
    answer: np.ndarray              # the pinned decode
    ids: np.ndarray                 # [L, k] the k best tokens per cell, best first
    vals: np.ndarray                # [L, k] their logits
    rank: tuple | None = None       # the ranking key among the board's children of one level
    row: np.ndarray | None = None   # the hinted input
    q: float | None = None          # q_halt


def propose(task: Task, i: int, row: np.ndarray, answer: np.ndarray, ids: np.ndarray, vals: np.ndarray, cells: int,
            alts: int) -> list[tuple[int, int]]:
    """Stage B's hypotheses for board i: the editable cells of `row` (the input with the hints written so far), least
    confident first (the smallest gap between the decode's two best logits, vals[:, 0] - vals[:, 1]), each with up to
    `alts` alternative tokens (Task.alternatives, ranked by ids); the first `cells` (cell, token) pairs. `answer` is the
    pinned decode."""
    margin = (vals[:, 0] - vals[:, 1]).astype(np.float64)
    out = []
    for c in sorted(task.editable(i, row), key=lambda c: margin[c]):
        out += [(int(c), int(tok)) for tok in task.alternatives(i, c, int(answer[c]), ids[c], answer)[:alts]]
        if len(out) >= cells:
            break
    return out[:cells]


class _Search:
    def __init__(self, model, task: Task, cfg: ICSConfig):
        self.model, self.task, self.cfg = model, task, cfg
        n = len(task)
        self.segs = np.zeros(n, np.int64)
        self.done = np.zeros(n, bool)
        self.stage = np.full(n, "", dtype=object)
        self.committed = None
        self.incumbent = {}           # confirm: board -> (answer, q)
        self.next_restatement = {}    # confirm: board -> index of the next restatement to confirm with

    def roll(self, rows, T, batch=None, state=None, keep_state=False):
        return roll(self.model, np.asarray(rows), T, batch or self.cfg.batch, state=state, keep_state=keep_state)

    def commit(self, i, answer, stage):
        self.committed[i] = answer
        self.done[i] = True
        self.stage[i] = stage

    def consider(self, i, x_in, answer, q, stage) -> bool:
        """Offer pinned candidate `answer` (from input `x_in`) to the terminal: True if board i commits on it."""
        if self.done[i]:
            return False
        if self.cfg.terminal == "confirm":
            cur = self.incumbent.get(i)
            if cur is not None and not q > cur[1]:
                return False
            self.incumbent[i] = (answer.copy(), q)
            if not self._reproduced(i, x_in, answer):
                return False
        elif self.cfg.terminal == "cert":
            if not self.task.valid(i, answer):          # answer is pinned and pin is idempotent: the pinned scoring
                return False
        elif not (q > 0 and self.task.consistent(i, answer)):
            return False
        self.commit(i, answer, stage)
        return True

    def _reproduced(self, i, x_in, answer) -> bool:
        restated = self.task.restatements(i, x_in)
        fresh = [(tag, row) for tag, row in restated if not np.array_equal(row, x_in)] or restated
        if not fresh:
            return False
        j = self.next_restatement.get(i, 0) % len(fresh)
        self.next_restatement[i] = j + 1
        tag, row = fresh[j]
        r = self.roll(row[None], self.cfg.T, batch=1)
        self.segs[i] += self.cfg.T
        back = self.task.restate_back(i, tag, r.dec[0])
        return back is not None and np.array_equal(self.task.pin(i, back), answer)

    def _hinted(self, i, stack):
        """Board i's input with the hints of `stack` written into it."""
        row = self.task.X[i].copy()
        for c, tok in stack:
            row[c] = tok
        return row

    def _hints(self, i, node: Node) -> list[tuple[int, int]]:
        """The (cell, token) hints proposed for the children of `node`, in proposal order (propose)."""
        return propose(self.task, i, self._hinted(i, node.stack), node.answer, node.ids, node.vals, self.cfg.cells,
                       self.cfg.alts)

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(the committed answers in ANSWER_DTYPE, segs, stage as str), for an empty task too."""
        X = self.task.X
        if len(X) == 0:
            return np.zeros(X.shape, ANSWER_DTYPE), self.segs, self.stage.astype(str)
        roots = self._stage_a_and_patience()
        self._stage_b(roots)
        self._stage_c()
        return self.committed.astype(ANSWER_DTYPE), self.segs, self.stage.astype(str)

    def _stage_a_and_patience(self) -> dict[int, Node]:
        """Stages A and P. Returns the root node of every board still open."""
        task, cfg, X = self.task, self.cfg, self.task.X
        r = self.roll(X, cfg.T_greedy, keep_state=bool(cfg.patience))
        self.segs += cfg.T_greedy
        self.committed = np.stack([task.pin(i, r.dec[i]) for i in range(len(task))]).astype(np.int64)
        ids, vals = r.top_ids.copy(), r.top_vals.copy()
        for i in range(len(task)):
            self.consider(i, X[i], self.committed[i], float(r.q[i]), "A")
        idx, depth = np.arange(len(task)), cfg.T_greedy                     # boards idx's roll on x is at `depth`
        state, r.state = r.state, None        # its one reference: the kept state is freed once its open rows are copied
        for target in cfg.patience:
            still = ~self.done[idx]
            if not still.any():
                break
            idx, state = idx[still], tuple(z[torch.from_numpy(still)] for z in state)    # the open boards' states only
            r = self.roll(X[idx], target - depth, state=state, keep_state=target < cfg.patience[-1])
            self.segs[idx] += target - depth
            depth = target
            state, r.state = r.state, None
            for k, i in enumerate(idx):
                if not self.consider(int(i), X[i], task.pin(int(i), r.dec[k]), float(r.q[k]), "P"):
                    ids[i], vals[i] = r.top_ids[k], r.top_vals[k]
        return {int(i): Node((), self.committed[i], ids[i], vals[i]) for i in np.flatnonzero(~self.done)}

    def _stage_b(self, roots: dict[int, Node]):
        """Stage B from the root nodes of the open boards."""
        task, cfg = self.task, self.cfg
        parents = {i: [root] for i, root in roots.items()}        # open boards only: a board leaves when it commits
        spent = np.zeros(len(task), np.int64)
        for _level in range(cfg.levels):
            kept = []                                               # (board, hints) of each child to roll
            for i, nodes in parents.items():
                if spent[i] < cfg.node_budget:
                    stacks = [node.stack + (hint,) for node in nodes for hint in self._hints(i, node)]
                    kept += [(i, stack) for stack in stacks[: cfg.node_budget - spent[i]]]
            if not kept:
                break
            rows = [self._hinted(i, stack) for i, stack in kept]
            r = self.roll(np.stack(rows), cfg.T)
            children = {}
            for j, ((i, stack), row) in enumerate(zip(kept, rows)):
                self.segs[i] += cfg.T
                spent[i] += 1
                raw, q = r.dec[j], float(r.q[j])
                rank = (sum(int(raw[c]) != tok for c, tok in stack), int(not task.consistent(i, raw)), -q)
                children.setdefault(i, []).append(Node(stack, task.pin(i, raw), r.top_ids[j], r.top_vals[j],
                                                       rank, row, q))
            for i, kids in children.items():
                kids.sort(key=lambda kid: kid.rank)
                if any(self.consider(i, kid.row, kid.answer, kid.q, "B") for kid in kids):
                    del parents[i]
                else:
                    parents[i] = kids[: cfg.beam]
                    self.committed[i] = kids[0].answer

    def _stage_c(self):
        """Stage C, then the q and confirm terminals' commits for the boards still open."""
        task, cfg, X = self.task, self.cfg, self.task.X
        idx = [int(i) for i in np.flatnonzero(~self.done)]
        restated = {i: task.restatements(i, X[i]) for i in idx}
        votes = {i: [] for i in idx}
        for v in range(max((len(rs) for rs in restated.values()), default=0)):
            members = [i for i in idx if v < len(restated[i])]
            r = self.roll(np.stack([restated[i][v][1] for i in members]), cfg.T)
            for j, i in enumerate(members):
                self.segs[i] += cfg.T
                back = task.restate_back(i, restated[i][v][0], r.dec[j])
                if back is None:
                    continue
                answer, q = task.pin(i, back), float(r.q[j])
                votes[i].append((answer, q))
                self.consider(i, X[i], answer, q, "C")
        for i in idx:
            if self.done[i]:
                continue
            if cfg.terminal == "q" and votes[i]:
                key, count = Counter(a.tobytes() for a, _ in votes[i]).most_common(1)[0]
                if count >= cfg.agree:
                    self.commit(i, next(a for a, _ in votes[i] if a.tobytes() == key), "C-agree")
                else:
                    self.commit(i, max(votes[i], key=lambda vq: vq[1])[0], "C-best-q")
            elif cfg.terminal == "confirm":
                if i in self.incumbent:
                    self.committed[i] = self.incumbent[i][0]
                self.done[i] = True
                self.stage[i] = "C-incumbent"


def run_ics(model, task: Task, cfg: ICSConfig, regime: str) -> dict[str, np.ndarray]:
    """ICS on every board: answers (ANSWER_DTYPE) under "ics/<regime>/<scoring>", the same pinned array for both
    scorings (treat it as read-only), plus "ics/<regime>/segs" (int64), every segment rolled for the board, refuted
    branches included, and "ics/<regime>/stage" (str), where it committed: "A", "P", "B" or "C", "C-agree" or "C-best-q"
    (the q terminal's fallbacks), "C-incumbent" (confirm's), or "" when the board stayed open."""
    committed, segs, stage = _Search(model, task, cfg).run()
    out = {f"ics/{regime}/{s}": committed for s in SCORINGS}
    out[f"ics/{regime}/segs"] = segs
    out[f"ics/{regime}/stage"] = stage
    return out
