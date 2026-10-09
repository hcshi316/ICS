import re
import types
from collections import Counter

import numpy as np
import pytest
import torch

from fakes import SOLUTION, ScriptedTRM, maze_pool, solve_maze, sudoku_pool
from ics.methods.ics import ICSConfig, _Search, propose, run_ics
from ics.tasks.maze import Maze
from ics.tasks.sudoku import Sudoku
from ics.trm.model import TRM, TRMConfig
from ics.trm.roll import roll

WRONG = SOLUTION.copy()
WRONG[0] = 3                                  # SOLUTION[0] is 2 (digit 1); 3 duplicates row 0

SMALL = {"T": 2, "T_greedy": 3, "patience": (5,), "levels": 2, "beam": 2, "cells": 4, "alts": 1, "node_budget": 6,
         "batch": 8}


def cfg(terminal, **kw):
    return ICSConfig(terminal=terminal, **{**SMALL, **kw})


def search(model, task, config):
    """The search's outputs (_Search.run) by name."""
    committed, segs, stage = _Search(model, task, config).run()
    return types.SimpleNamespace(committed=committed, segs=segs, stage=stage)


def test_stage_a_and_patience():
    pool = sudoku_pool([(0, 1), (0, 2)])
    late = pool.inputs[1].tobytes()
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION if row.tobytes() != late or t >= 5 else WRONG, 1.0, 5.0))
    out = run_ics(model, Sudoku(pool), cfg("cert"), "cert")
    assert list(out["ics/cert/stage"]) == ["A", "P"]
    np.testing.assert_array_equal(out["ics/cert/segs"], [3, 5])  # P continues board 1's roll from depth 3 to depth 5
    np.testing.assert_array_equal(out["ics/cert/raw"], [SOLUTION, SOLUTION])


def test_stage_b_writes_the_least_confident_alternative():
    pool = sudoku_pool([(0, 30, 60)])
    margins = np.full(81, 5.0)
    margins[0] = 1.0                           # cell 0 is the least confident

    def script(row, t, zl):
        return (SOLUTION if row[0] == SOLUTION[0] else WRONG), -1.0, margins

    out = run_ics(ScriptedTRM(11, script), Sudoku(pool), cfg("cert"), "cert")
    assert list(out["ics/cert/stage"]) == ["B"]
    np.testing.assert_array_equal(out["ics/cert/raw"], [SOLUTION])
    # A and P (the roll on x, to depth 5) + one level of three single-cell hypotheses (3 x T=2)
    np.testing.assert_array_equal(out["ics/cert/segs"], [5 + 3 * 2])
    tight = run_ics(ScriptedTRM(11, script), Sudoku(pool), cfg("cert", node_budget=1), "cert")
    np.testing.assert_array_equal(tight["ics/cert/segs"], [5 + 2])  # the budget admits only the first hypothesis
    np.testing.assert_array_equal(tight["ics/cert/raw"], [SOLUTION])


def test_children_rank_by_hint_respect_then_q():
    # Both cells get the hypothesis token 2 (the best-ranked alternative: lower ids rank higher in ScriptedTRM).
    pool = sudoku_pool([(0, 30)])
    other = WRONG.copy()
    other[30] = 2                              # the cell-30 child keeps its own hint

    def script(row, t, zl):
        if row[0] != 1:                        # hint in cell 0: overwrites its own hint, high q
            out = WRONG.copy(); out[0] = 4
            return out, 2.0, 5.0
        if row[30] != 1:                       # hint in cell 30: respects it, lower q
            return other, 0.5, 5.0
        return WRONG, -1.0, 5.0

    out = run_ics(ScriptedTRM(11, script), Sudoku(pool), cfg("q", patience=()), "model")
    assert list(out["ics/model/stage"]) == ["B"]
    np.testing.assert_array_equal(out["ics/model/raw"][0], other)       # the respecting child ranks first


def test_propose_gives_each_cells_first_alternatives_least_confident_cell_first_cut_at_cells():
    # Blanks 0, 30 and 60 with margins 3, 1 and 2; each cell's ranked tokens start with the decode's own (SOLUTION).
    task = Sudoku(sudoku_pool([(0, 30, 60)]))
    ids, vals = np.zeros((81, 3), np.int64), np.zeros((81, 3), np.float32)
    for cell, ranked, margin in ((0, [2, 9, 5], 3.0), (30, [6, 3, 8], 1.0), (60, [10, 7, 2], 2.0)):
        ids[cell], vals[cell, 0] = ranked, margin
    hints = propose(task, 0, task.X[0], SOLUTION, ids, vals, cells=5, alts=2)
    assert hints == [(30, 3), (30, 8), (60, 7), (60, 2), (0, 9)]


def test_stage_b_stacks_hints_over_levels():
    pool = sudoku_pool([(0, 15)])              # SOLUTION[0] == SOLUTION[15] == 2, each cell's first alternative to 3

    def script(row, t, zl):                    # cells 0 and 15 decode wrong (3) until each carries its hint
        out = SOLUTION.copy()
        for c in (0, 15):
            if row[c] != 2:
                out[c] = 3
        return out, -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool), cfg("cert"))
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed, [SOLUTION])
    # A and P (to depth 5) + level 0: two single hints (2 x 2) + level 1: each kept parent adds the other hint (2 x 2)
    np.testing.assert_array_equal(res.segs, [5 + 2 * 2 + 2 * 2])


def maze_task():
    return Maze(maze_pool([1, 2, 3], (1, 1), (1, 10)))


def orientation_script(correct_q, wrong_q):
    """Solves every restatement except the original orientation (start at (1, 1)), where it answers the empty maze."""
    def script(row, t, zl):
        if row[1 * 30 + 1] == 3:
            return row.copy(), wrong_q, 5.0
        return solve_maze(row), correct_q, 5.0
    return script


def test_stage_c_agreement_under_q():
    task = maze_task()
    res = search(ScriptedTRM(6, orientation_script(-1.0, -1.0)), task, cfg("q", patience=(), levels=0))
    assert list(res.stage) == ["C-agree"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])
    np.testing.assert_array_equal(res.segs, [3 + 8 * 2])


def test_stage_c_best_q_without_agreement():
    # 6 of the 8 restatements solve (q -0.5); the identity and the transpose keep the start at (1, 1) and do not (q -1)
    task = maze_task()
    res = search(ScriptedTRM(6, orientation_script(-0.5, -1.0)), task, cfg("q", patience=(), levels=0, agree=9))
    assert list(res.stage) == ["C-best-q"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])


def test_stage_c_certificate_commits_first_valid_restatement():
    task = maze_task()
    res = search(ScriptedTRM(6, orientation_script(-1.0, -1.0)), task, cfg("cert", patience=(), levels=0))
    assert list(res.stage) == ["C"]


def test_confirm_locks_a_reproduced_incumbent():
    task = maze_task()
    res = search(ScriptedTRM(6, orientation_script(0.9, 0.5)), task, cfg("confirm", patience=(), levels=0))
    assert list(res.stage) == ["C"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])
    # A (3) + its confirmation roll (2) + 8 restatements (16) + the confirmation of the first better one (2)
    np.testing.assert_array_equal(res.segs, [3 + 2 + 16 + 2])


def test_restatement_counts_may_differ_per_board():
    class Uneven(Maze):
        def restatements(self, i, row):
            restatements = super().restatements(i, row)
            return restatements[:3] if i == 0 else restatements

    pool = maze_pool([1, 2, 3], (1, 1), (1, 10))
    both = type(pool)(np.repeat(pool.inputs, 2, 0), np.repeat(pool.labels, 2, 0), np.arange(2), {})
    res = search(ScriptedTRM(6, orientation_script(-1.0, -1.0)), Uneven(both), cfg("q", patience=(), levels=0))
    np.testing.assert_array_equal(res.segs, [3 + 3 * 2, 3 + 8 * 2])


def test_q_terminal_requires_the_givens_kept():
    class Inconsistent(Sudoku):
        def consistent(self, i, y):
            return False

    task = Inconsistent(sudoku_pool([(0,)]), n_restatements=1)
    res = search(ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 1.0, 5.0)), task, cfg("q", patience=(), levels=0))
    assert list(res.stage) == ["C-best-q"]          # q > 0 everywhere, but no pinned decode keeps the givens
    np.testing.assert_array_equal(res.segs, [3 + 2])


def test_run_ics_keys():
    pool = sudoku_pool([(0,)])
    out = run_ics(ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 1.0, 5.0)), Sudoku(pool), cfg("cert"), "cert")
    assert set(out) == {"ics/cert/raw", "ics/cert/pinned", "ics/cert/segs", "ics/cert/stage"}
    assert out["ics/cert/raw"].dtype == np.int16


BAD_DEPTHS = {"patience_not_increasing": (5, 0), "patience_repeated": (5, 5), "patience_at_T_greedy": (3,),
              "patience_below_T_greedy": (2, 5)}             # SMALL's T_greedy is 3
BAD_SETTINGS = {"terminal": ({"terminal": "agree"}, "unknown terminal 'agree'"),
                **{name: ({name: 0}, f"ICSConfig needs {name} >= 1, got {name}=0")
                   for name in ("T", "T_greedy", "beam", "cells", "alts", "node_budget", "agree", "batch")},
                "levels": ({"levels": -1}, "ICSConfig needs levels >= 0, got levels=-1"),
                **{case: ({"patience": depths}, ("ICSConfig needs strictly increasing patience depths above "
                                                  f"T_greedy=3, got patience={depths}"))
                   for case, depths in BAD_DEPTHS.items()}}


@pytest.mark.parametrize("change, error", BAD_SETTINGS.values(), ids=BAD_SETTINGS)
def test_config_rejects_out_of_range_settings(change, error):
    with pytest.raises(ValueError, match=f"^{re.escape(error)}"):
        ICSConfig(**{"terminal": "cert", **SMALL, **change})


def test_config_accepts_the_smallest_settings():
    smallest = ICSConfig("q", T=1, T_greedy=1, patience=(2,), levels=0, beam=1, cells=1, alts=1, node_budget=1, agree=1,
                         batch=1)
    res = search(ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 1.0, 5.0)), Sudoku(sudoku_pool([(0,)])), smallest)
    assert list(res.stage) == ["A"]


def test_config_from_settings_keeps_only_its_fields():
    settings = {"terminal": "q", "T": 3, "patience": [64, 256], "seed": 7, "window": None, "restatements": 32}
    assert ICSConfig.from_settings(settings) == ICSConfig("q", T=3, patience=(64, 256))


MOVED = SOLUTION.copy()
MOVED[1] = SOLUTION[2]                          # changes cell 1, a given on every board below


def test_terminals_see_and_commit_pinned_decodes():
    res = search(ScriptedTRM(11, lambda row, t, zl: (MOVED, 1.0, 5.0)), Sudoku(sudoku_pool([(0,)])), cfg("q"))
    assert list(res.stage) == ["A"]            # the raw decode changes a given; pinned, it keeps them all
    np.testing.assert_array_equal(res.committed, [SOLUTION])


@pytest.mark.parametrize("patience, levels, stage", [((5,), 0, "P"), ((), 1, "B"), ((), 0, "C")])
def test_patience_continues_the_roll_on_x_and_every_other_roll_starts_afresh(patience, levels, stage):
    # Right only at segment 5 of the roll on x (P continues A's 3 segments to depth 5; a fresh roll of the 2 more would
    # end at segment 2) or at segment T = 2 of a hinted or restated roll (continuing A's roll would end at 5). Each way
    # the board costs 3 + 2 segments; P rolling afresh to depth 5 would cost 3 + 5.
    right_at = 5 if stage == "P" else 2
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION if t == right_at else WRONG, -1.0, 5.0))
    res = search(model, Sudoku(sudoku_pool([(0,)]), n_restatements=1), cfg("cert", patience=patience, levels=levels))
    assert list(res.stage) == [stage]
    np.testing.assert_array_equal(res.segs, [3 + 2])


def test_the_root_hypotheses_read_stage_a_decode_and_the_last_patience_margins():
    # Until cell 15 is hinted 2, cell 0 decodes 3, and so does cell 15 after A's 3 segments (right at P's depth 5).
    # The least confident cell is 0 after A and 15 after P. The one hypothesis (cells=1) reads P's margins and A's
    # decode: (15, 2). A's margins would give (0, 2), and P's decode (15, 3).
    pool = sudoku_pool([(0, 15)])

    def script(row, t, zl):
        out, margins = SOLUTION.copy(), np.full(81, 5.0)
        if row[15] != 2:
            out[0] = 3
            if t == 3:
                out[15] = 3
        margins[0 if t == 3 else 15] = 1.0
        return out, -1.0, margins

    res = search(ScriptedTRM(11, script), Sudoku(pool, n_restatements=1), cfg("cert", cells=1, levels=1))
    assert list(res.stage) == ["B"]


def test_children_rank_hint_respect_before_the_givens():
    pool = sudoku_pool([(0, 30)])

    def script(row, t, zl):
        if row[0] != 1:                        # hint in cell 0: respects it, but changes a given
            return MOVED, 1.0, 5.0
        if row[30] != 1:                       # hint in cell 30: overwrites it, keeps the givens
            return WRONG, 1.0, 5.0
        return WRONG, -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool), cfg("q", patience=()))
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed[0], SOLUTION)     # the respecting child, pinned


def test_children_rank_the_givens_before_q():
    pool = sudoku_pool([(0, 30)])
    other = WRONG.copy()
    other[30] = 2

    def script(row, t, zl):
        if row[0] != 1:                        # hint in cell 0: respects it but changes a given, high q
            return MOVED, 2.0, 5.0
        if row[30] != 1:                       # hint in cell 30: respects it and keeps the givens, lower q
            return other, 0.5, 5.0
        return WRONG, -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool), cfg("q", patience=()))
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed[0], other)


def test_gc_counts_every_stacked_hint_the_decode_overwrote():
    # Blanks decode 3 and hints are kept, except that the hint (15, 2) makes the model overwrite cell 0's earlier hint,
    # with the higher q. Level 0 keeps (0, 2); level 1 tries (15, 2) and (15, 4) on top of it.
    pool = sudoku_pool([(0, 15)])

    def script(row, t, zl):
        out = np.where(row == 1, 3, row)
        if row[15] == 2:
            out[0] = 5
        return out, (1.0 if row[15] == 2 else 0.5 if row[15] == 4 else -1.0), 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool), cfg("q", patience=(), cells=2, alts=2, beam=1))
    kept_both = SOLUTION.copy()
    kept_both[15] = 4
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed[0], kept_both)     # not (15, 2)'s higher-q child, which lost (0, 2)


def test_certificate_takes_the_first_valid_child_in_rank_order():
    pool = sudoku_pool([(0, 30)])
    other = WRONG.copy()
    other[30] = 2

    def script(row, t, zl):
        if row[0] != 1:                        # hint in cell 0: right, lower q
            return SOLUTION, -1.0, 5.0
        if row[30] != 1:                       # hint in cell 30: respects it with a higher q, still wrong
            return other, 1.0, 5.0
        return WRONG, -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool, n_restatements=1), cfg("cert", patience=(), levels=1))
    assert list(res.stage) == ["B"]            # the first-ranked child fails the certificate, the second passes


def test_the_next_level_expands_the_kept_childs_own_decode():
    # Bare input: cell 0 decodes 3 and is the least confident; cells 15 and 30 decode right, 30 less surely (margin 3).
    # With cell 0 hinted, cell 15 decodes 3 and becomes the least confident: only that child's own decode and margins
    # lead to the second hint (15, 2).
    pool = sudoku_pool([(0, 15, 30)])

    def script(row, t, zl):
        out, margins = SOLUTION.copy(), np.full(81, 5.0)
        if row[0] == 1:
            out[0], margins[0], margins[30] = 3, 1.0, 3.0
        elif row[15] != 2:
            out[15], margins[15] = 3, 1.0
        out[row != 1] = row[row != 1]          # hints are kept
        return out, -1.0, margins

    res = search(ScriptedTRM(11, script), Sudoku(pool, n_restatements=1),
                 cfg("cert", patience=(), cells=2, alts=2, beam=1))
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed, [SOLUTION])
    # A (3) + level 0: (0, 2) and (0, 4) tie and only the first is kept (2 x 2) + level 1: (15, 2) and (15, 4) (2 x 2)
    np.testing.assert_array_equal(res.segs, [3 + 2 * 2 + 2 * 2])


def test_c_best_q_takes_the_first_of_tied_restatements():
    # As orientation_script(-0.5, -1.0), except that the last restatement, the transpose (start (1, 1), goal (10, 1)),
    # ties the best q with its empty maze.
    task = maze_task()

    def script(row, t, zl):
        if row[1 * 30 + 1] == 3:
            return row.copy(), (-0.5 if row[10 * 30 + 1] == 4 else -1.0), 5.0
        return solve_maze(row), -0.5, 5.0

    res = search(ScriptedTRM(6, script), task, cfg("q", patience=(), levels=0, agree=9))
    assert list(res.stage) == ["C-best-q"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])     # restatement 1's answer, not restatement 7's


@pytest.mark.parametrize("agree, stage", [(6, "C-agree"), (7, "C-best-q")])
def test_c_agreement_needs_at_least_agree_restatements(agree, stage):
    task = maze_task()                         # 6 of the 8 restatements agree on the solution
    res = search(ScriptedTRM(6, orientation_script(-1.0, -1.0)), task, cfg("q", patience=(), levels=0, agree=agree))
    assert list(res.stage) == [stage]


def test_confirm_checks_each_new_incumbent_on_the_next_restatement():
    # Restatements with the start at (1, 1) or (28, 1) answer the empty maze (q 0.5); the others solve it (q 0.9).
    task = maze_task()

    def script(row, t, zl):
        if row[1 * 30 + 1] == 3 or row[28 * 30 + 1] == 3:
            return row.copy(), 0.5, 5.0
        return solve_maze(row), 0.9, 5.0

    res = search(ScriptedTRM(6, script), task, cfg("confirm", patience=(), levels=0))
    # A's empty maze fails its check on restatement 1; restatement 1's solution replaces it and fails its check on
    # restatement 2, so the board commits that incumbent after C
    assert list(res.stage) == ["C-incumbent"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])
    np.testing.assert_array_equal(res.segs, [3 + 2 + 16 + 2])


def test_confirm_re_derives_a_child_from_its_hinted_input():
    task = maze_task()
    margins = np.full(900, 5.0)
    margins[1 * 30 + 5] = 1.0                  # the least confident cell, on the path: the one hypothesis marks it

    def script(row, t, zl):
        if (row == 5).any():                   # the hinted input (PATH), in any restatement: solved
            return solve_maze(row), 0.9, margins
        if row[1 * 30 + 1] == 3 and row[1 * 30 + 10] == 4:     # the bare input itself: the empty maze
            return row.copy(), 0.5, margins
        return np.where(row == 2, 5, row), 0.5, margins        # its other restatements: every empty cell marked

    res = search(ScriptedTRM(6, script), task, cfg("confirm", patience=(), levels=1, cells=1))
    assert list(res.stage) == ["B"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])
    # A (3) + its check (2) + the child (2) + the child's check on restatement 2 of the hinted input (2)
    np.testing.assert_array_equal(res.segs, [3 + 2 + 2 + 2])


def test_each_board_keeps_its_own_segments_and_node_budget():
    # Cells 0 and 15 decode 3 until each carries its hint 2, as in test_stage_b_stacks_hints_over_levels. Board 0 blanks
    # both: level 0 rolls (0, 2) and (15, 2), and at level 1 its budget of 3 admits only ((0, 2), (15, 2)). Board 1
    # blanks cells 0 and 30 (its cell 15 is a given 2): level 0 rolls (0, 2) and (30, 2), and (0, 2) solves it.
    pool = sudoku_pool([(0, 15), (0, 30)])

    def script(row, t, zl):
        out = SOLUTION.copy()
        for c in (0, 15):
            if row[c] != 2:
                out[c] = 3
        return out, -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool), cfg("cert", patience=(), node_budget=3))
    assert list(res.stage) == ["B", "B"]
    np.testing.assert_array_equal(res.committed, [SOLUTION, SOLUTION])
    np.testing.assert_array_equal(res.segs, [3 + 2 * 2 + 1 * 2, 3 + 2 * 2])


def test_each_board_confirms_on_its_own_next_restatement():
    # The scenario of test_confirm_checks_each_new_incumbent_on_the_next_restatement on two copies of the maze. A
    # restatement counter shared by the boards would check board 1's first incumbent on restatement 2, which reproduces
    # it.
    pool = maze_pool([1, 2, 3], (1, 1), (1, 10))
    both = Maze(type(pool)(np.repeat(pool.inputs, 2, 0), np.repeat(pool.labels, 2, 0), np.arange(2), {}))

    def script(row, t, zl):
        if row[1 * 30 + 1] == 3 or row[28 * 30 + 1] == 3:
            return row.copy(), 0.5, 5.0
        return solve_maze(row), 0.9, 5.0

    res = search(ScriptedTRM(6, script), both, cfg("confirm", patience=(), levels=0))
    assert list(res.stage) == ["C-incumbent", "C-incumbent"]
    np.testing.assert_array_equal(res.committed, both.Y)
    np.testing.assert_array_equal(res.segs, [3 + 2 + 16 + 2] * 2)


def test_confirm_under_patience_continues_x_but_re_derives_afresh():
    # The model solves the original maze only at segment 5 of its roll (before, it answers the empty maze, q 0.5) and a
    # restatement only at segment T = 2. A's empty maze becomes the incumbent and fails its check, a fresh 2-segment
    # roll of restatement 1 (it decodes the solution). P continues x's roll by 2 segments to depth 5; the solution beats
    # the incumbent's q and is reproduced by a fresh roll of restatement 2, so the board commits at P with
    # 3 (A) + 2 (A's check) + 2 (P: depth 3 to 5) + 2 (P's check) segments. A P that rolled afresh would roll 5
    # segments, or end unsolved at segment 2 if it rolled only the 2 more; a check that continued x's state would end,
    # unsolved, at segment 7.
    task = maze_task()
    x = task.X[0]

    def script(row, t, zl):
        if np.array_equal(row, x):
            return (solve_maze(row), 0.9, 5.0) if t == 5 else (row.copy(), 0.5, 5.0)
        return (solve_maze(row) if t == 2 else row.copy()), -1.0, 5.0

    res = search(ScriptedTRM(6, script), task, cfg("confirm", patience=(5,), levels=0))
    assert list(res.stage) == ["P"]
    np.testing.assert_array_equal(res.committed[0], task.Y[0])
    np.testing.assert_array_equal(res.segs, [3 + 2 + 2 + 2])


def test_patience_continues_the_open_boards_roll_on_x_to_each_depth():
    # The production depths. Board k decodes right only at segment 20 (Stage A), 64, 256 and never: each board's input
    # sees segments 1, 2, ... once, up to the depth where the board commits; a board still open after P goes on to
    # Stage C, whose one restatement (the input itself) is rolled afresh. batch=1: no pad rows repeat a board's input.
    pool = sudoku_pool([(0,), (0, 1), (0, 2), (0, 3)])
    keys = [x.tobytes() for x in pool.inputs]
    right_at = dict(zip(keys, (20, 64, 256)))
    seen = {key: [] for key in keys}

    def script(row, t, zl):
        seen[row.tobytes()].append(t)
        return (SOLUTION if right_at.get(row.tobytes()) == t else WRONG), -1.0, 5.0

    res = search(ScriptedTRM(11, script), Sudoku(pool, n_restatements=1),
                 cfg("cert", T_greedy=20, patience=(64, 256), levels=0, batch=1))
    assert list(res.stage) == ["A", "P", "P", ""]
    np.testing.assert_array_equal(res.committed[:3], [SOLUTION] * 3)
    np.testing.assert_array_equal(res.segs, [20, 64, 256, 256 + 2])
    assert [seen[key] for key in keys] == [[*range(1, 21)], [*range(1, 65)], [*range(1, 257)], [*range(1, 257), 1, 2]]


def tiny_sudoku_trm():
    """A tiny real TRM of Sudoku's shape, float32 on the CPU. Its q head is drawn at random: TRM's init zeroes the
    weight, which would make q_halt a constant."""
    torch.manual_seed(0)
    model = TRM(TRMConfig(seq_len=81, vocab_size=11, puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=2, L_cycles=2,
                          L_layers=1, hidden_size=16, num_heads=2, expansion=2.0, forward_dtype="float32")).eval()
    torch.nn.init.normal_(model.inner.q_head.weight)
    return model


class Scheduled(_Search):
    """The search with a scripted terminal: board i commits on its accept[i]-th offer (Stage A's is the 1st), else
    never. Records every roll: (rows, T, Rolled)."""

    def __init__(self, model, task, cfg, accept):
        super().__init__(model, task, cfg)
        self.accept, self.offers, self.rolls = accept, Counter(), []

    def roll(self, rows, T, batch=None, state=None, keep_state=False):
        r = super().roll(rows, T, batch, state=state, keep_state=keep_state)
        self.rolls.append((np.array(rows), T, r))
        return r

    def consider(self, i, x_in, answer, q, stage):
        self.offers[i] += 1
        if self.done[i] or self.offers[i] != self.accept.get(i):
            return False
        self.commit(i, answer, stage)
        return True


def test_patience_decodes_equal_fresh_rolls_to_each_depth():
    # A tiny real TRM. Boards 0, 2 and 4 commit at A, at depth 5 and at depth 9, so each depth continues the states of
    # fewer boards. Each patience roll must decode (tokens and q) as the same boards rolled from the initial state to
    # its depth. batch=1 runs every board alone in each roll, so the arithmetic is the same in both and the comparison
    # can be exact.
    model = tiny_sudoku_trm()
    rng = np.random.default_rng(0)
    pool = sudoku_pool([tuple(np.flatnonzero(rng.random(81) < 0.5)) for _ in range(7)])
    run = Scheduled(model, Sudoku(pool, n_restatements=1),
                    cfg("cert", T_greedy=3, patience=(5, 9, 12), levels=0, batch=1), accept={0: 1, 2: 2, 4: 3})
    _, segs, stage = run.run()
    assert list(stage) == ["A", "", "P", "", "P", "", ""]
    np.testing.assert_array_equal(segs, [3, 12 + 2, 5, 12 + 2, 9, 12 + 2, 12 + 2])
    patience = run.rolls[1:4]
    assert [T for _rows, T, _r in patience] == [2, 4, 3]
    for depth, boards, (rows, T, r) in zip((5, 9, 12), ([1, 2, 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 5, 6]), patience):
        np.testing.assert_array_equal(rows, pool.inputs[boards])
        fresh = roll(model, pool.inputs[boards], depth, batch=1)
        for name in ("dec", "q", "top_ids", "top_vals"):
            np.testing.assert_array_equal(getattr(r, name), getattr(fresh, name))
        restart = roll(model, pool.inputs[boards], T, batch=1)     # only the added segments, from the initial state:
        assert not np.array_equal(restart.top_vals, fresh.top_vals)    # the comparison tells the two apart


def test_q_terminal_needs_q_above_zero():
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 0.0, 5.0))
    res = search(model, Sudoku(sudoku_pool([(0,)]), n_restatements=1), cfg("q", patience=(), levels=0))
    assert list(res.stage) == ["C-best-q"]      # q_halt exactly 0 never passes the gate, not even on the solution


def test_an_empty_pool_gives_an_empty_result():
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION, 1.0, 5.0))
    task = Sudoku(sudoku_pool([(0,)]).take(slice(0, 0)))
    res = search(model, task, cfg("cert"))
    assert res.committed.shape == (0, 81) and res.committed.dtype == np.int16
    assert res.segs.shape == res.stage.shape == (0,)
    assert res.segs.dtype == np.int64 and res.stage.dtype.kind == "U"
    assert all(len(v) == 0 for v in run_ics(model, task, cfg("cert"), "cert").values())
