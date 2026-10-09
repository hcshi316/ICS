"""The verifier's selection (ics/verifier/select.py, run_ics_verifier) on Maze and on Line, a task whose second view
reverses a row: each distinct candidate is scored once, by its mean q over its own restatements."""
import numpy as np
import pytest

from fakes import ScriptedTRM, make_pool, maze_pool, solve_maze
from ics.tasks.base import Task
from ics.tasks.maze import PATH, START, WALL, Maze, rule_valid
from ics.verifier.select import run_ics_verifier

GOOD = np.array([2, 3, 4, 5, 2, 3])                 # the answer of every Line board; token 1 is blank
WRONG = np.array([2, 4, 3, 5, 2, 3])                # GOOD with cells 1 and 2 swapped
X0 = np.array([2, 1, 1, 5, 2, 3])                   # Line board 0: cells 1 and 2 blank


class Line(Task):
    """Rows of 6 tokens; restatement t reverses the row when t is odd. A given cell is non-blank; valid = the label."""

    def __init__(self, pool, views=None, lost=()):
        super().__init__(pool)
        self.views, self.lost = views or [2] * len(pool), lost         # restatements per board

    def pin(self, i, y):
        return np.where(self.X[i] != 1, self.X[i], y)

    def valid(self, i, y):
        return bool(np.array_equal(y, self.Y[i]))

    def restatements(self, i, row):
        return [(t, np.array(row)[::-1] if t % 2 else np.array(row)) for t in range(self.views[i])]

    def restate_back(self, i, tag, y):
        if any(np.array_equal(y, z) for z in self.lost):                       # a decode that cannot be mapped back
            return None
        return np.array(y)[::-1] if tag % 2 else np.array(y)


class Bare(Line):
    """A Line whose answers have no restatement of their own: only a board's input restates."""

    def restatements(self, i, row):
        return super().restatements(i, row) if np.array_equal(row, self.X[i]) else []


def line(*inputs, **kw):
    return Line(make_pool(list(inputs), [GOOD] * len(inputs)), **kw)


def open3():
    """Rows 1..3, columns 1..10 open; start (1, 1), goal (1, 10): one shortest path, along row 1."""
    return Maze(maze_pool([1, 2, 3], (1, 1), (1, 10)))


def solver(script):
    """A solver answering script(row) after every segment."""
    return ScriptedTRM(6, lambda row, t, zl: (script(row), 0.0, 1.0))


def judge(q_of):
    """A verifier whose q_halt on a row is q_of(row)."""
    return ScriptedTRM(6, lambda row, t, zl: (row, q_of(row), 1.0))


def known(row):
    return 1.0 if np.array_equal(row, GOOD) or np.array_equal(row, GOOD[::-1]) else -1.0


def maze_q(row):
    """+1 when the overlay's path is a valid answer to the overlay's own maze, else -1."""
    return 1.0 if rule_valid(np.where(row == PATH, 2, row), row) == "OK" else -1.0


def empty_at_start(row):                            # the empty maze where the start sits at (1, 1), else the path
    return row.copy() if row[31] == START else solve_maze(row)


def test_picks_a_verified_restatement():
    task = open3()                                  # restatements 0 and 7 put the start at (1, 1)
    out = run_ics_verifier(solver(empty_at_start), judge(maze_q), task, T=2, verifier_T=1, batch=8)
    assert out["ics/model/restatement"][0] != 0
    assert task.valid(0, out["ics/model/raw"][0])
    np.testing.assert_array_equal(out["ics/model/raw"], out["ics/model/pinned"])
    np.testing.assert_array_equal(out["ics/model/segs"], [8 * 2 + 2 * 8 * 1])   # 8 x T; 2 distinct x 8 views x 1


def test_scores_the_path_on_empty_cells_and_commits_it_pinned():
    # Outside restatements 0 and 7 (start at (1, 1)), the solver solves the maze and also writes path over every wall;
    # in restatements 0 and 7 it answers the empty maze. Only the overlay that keeps the walls verifies.
    task = open3()
    model = solver(lambda row: row.copy() if row[31] == START else np.where(row == WALL, PATH, solve_maze(row)))
    out = run_ics_verifier(model, judge(maze_q), task, T=1, verifier_T=1, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1])
    np.testing.assert_array_equal(out["ics/model/raw"], task.Y)          # the walls restored
    np.testing.assert_array_equal(out["ics/model/pinned"], task.Y)


def test_the_score_averages_the_overlays_8_restatements():
    # The verifier is fooled wherever the start sits at (1, 1): restatement 0's empty maze scores +1 in its views 0 and
    # 7 but (2 - 6) / 8 over all 8, below the +1 of the restatements that solve the maze.
    fooled = judge(lambda row: 1.0 if row[31] == START else maze_q(row))
    out = run_ics_verifier(solver(empty_at_start), fooled, open3(), T=1, verifier_T=1, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1])


def test_solver_and_verifier_roll_T_and_verifier_T_segments():
    # Outside restatements 0 and 7 the solver solves the maze from segment 3 on; the verifier's verdict flips after
    # segment 2.
    task = open3()
    model = ScriptedTRM(6, lambda row, t, zl: (row.copy() if row[31] == START or t < 3 else solve_maze(row), 0.0, 1.0))
    late = ScriptedTRM(6, lambda row, t, zl: (row, maze_q(row) * (1 if t <= 2 else -1), 1.0))
    out = run_ics_verifier(model, late, task, T=3, verifier_T=2, batch=8)
    np.testing.assert_array_equal(out["ics/model/raw"], task.Y)


def test_each_board_commits_its_own_restatement_and_candidate():
    # Board 1 is board 0 mirrored left-right. The solver answers the empty maze wherever the start sits at (1, 1): in
    # restatements 0 and 7 of board 0 but 1 and 2 of board 1, so board 0 commits restatement 1 and board 1
    # restatement 0.
    x0 = maze_pool([1, 2, 3], (1, 1), (1, 10)).inputs[0]
    x1 = np.fliplr(x0.reshape(30, 30)).reshape(-1)
    task = Maze(make_pool([x0, x1], [solve_maze(x0), solve_maze(x1)]))
    out = run_ics_verifier(solver(empty_at_start), judge(maze_q), task, T=1, verifier_T=1, batch=1)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1, 0])
    np.testing.assert_array_equal(out["ics/model/raw"], task.Y)


@pytest.mark.parametrize("T, verifier_T, batch", [(0, 1, 8), (1, 0, 8), (1, 1, 0)],
                         ids=["T_0", "verifier_T_0", "batch_0"])
def test_rejects_out_of_range_settings(T, verifier_T, batch):
    with pytest.raises(ValueError, match="run_ics_verifier needs"):
        run_ics_verifier(solver(solve_maze), judge(maze_q), open3(), T=T, verifier_T=verifier_T, batch=batch)


def test_accepts_the_smallest_settings():
    task = open3()
    out = run_ics_verifier(solver(solve_maze), judge(maze_q), task, T=1, verifier_T=1, batch=1)
    np.testing.assert_array_equal(out["ics/model/raw"], task.Y)
    np.testing.assert_array_equal(out["ics/model/segs"], [8 + 8])   # every restatement the same: 1 distinct candidate


def test_maze_candidates_come_from_its_restatements_in_their_order():
    task = open3()
    seen = []
    run_ics_verifier(solver(lambda row: seen.append(row.copy()) or solve_maze(row)), judge(maze_q), task, T=1,
                     verifier_T=1, batch=1)
    np.testing.assert_array_equal(seen, [row for _, row in task.restatements(0, task.X[0])])


def test_maze_scores_its_two_distinct_candidates_once():
    # Restatements 0 and 7 (start at (1, 1)) decode the empty maze, the six others its one shortest path: two distinct
    # candidates of 8 views each, each scored once. The path is committed, from restatement 1.
    task = open3()
    out = run_ics_verifier(solver(empty_at_start), judge(maze_q), task, T=2, verifier_T=3, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1])
    np.testing.assert_array_equal(out["ics/model/raw"], task.Y)
    np.testing.assert_array_equal(out["ics/model/segs"], [8 * 2 + 2 * 8 * 3])


def test_an_empty_pool_gives_an_empty_result():
    pool = maze_pool([1, 2, 3], (1, 1), (1, 10))
    full = run_ics_verifier(solver(solve_maze), judge(maze_q), Maze(pool), T=1, verifier_T=1, batch=8)
    empty = run_ics_verifier(solver(solve_maze), judge(maze_q), Maze(pool.take(slice(0, 0))), T=1, verifier_T=1,
                             batch=8)
    assert full["ics/model/raw"].dtype == full["ics/model/pinned"].dtype == np.int16
    assert full["ics/model/segs"].dtype == full["ics/model/restatement"].dtype == np.int64
    assert empty.keys() == full.keys()
    for key, arr in full.items():
        assert empty[key].shape == (0,) + arr.shape[1:] and empty[key].dtype == arr.dtype


def test_any_task_selects_among_its_restatements_decodes():
    # The solver answers WRONG on the board as given and GOOD's reversal on the reversed board, which maps back to GOOD.
    task = line(X0)
    out = run_ics_verifier(solver(lambda row: WRONG if row[0] == 2 else GOOD[::-1]), judge(known), task, T=3,
                           verifier_T=2, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1])
    np.testing.assert_array_equal(out["ics/model/raw"], [GOOD])
    assert out["ics/model/raw"] is out["ics/model/pinned"] and out["ics/model/raw"].dtype == np.int16
    np.testing.assert_array_equal(out["ics/model/segs"], [2 * 3 + 2 * 2 * 2])  # 2 decodes x T, 2 x 2 views x 2


def test_a_candidate_scores_the_mean_q_over_its_own_restatements():
    # WRONG scores 2 as it stands but -1 reversed: (2 - 1) / 2 < 1, GOOD's score in both views.
    q = lambda row: 2.0 if np.array_equal(row, WRONG) else known(row)
    out = run_ics_verifier(solver(lambda row: WRONG if row[0] == 2 else GOOD[::-1]), judge(q), line(X0), T=1,
                           verifier_T=1, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [1])


def test_ties_take_the_first_restatement():
    # The decodes are WRONG and another wrong answer: two distinct candidates, both scored, both -1.
    other = np.array([2, 2, 2, 5, 2, 3])
    out = run_ics_verifier(solver(lambda row: WRONG if row[0] == 2 else other[::-1]), judge(known), line(X0), T=1,
                           verifier_T=1, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [0])
    np.testing.assert_array_equal(out["ics/model/raw"], [WRONG])
    np.testing.assert_array_equal(out["ics/model/segs"], [2 * 1 + 2 * 2 * 1])     # 2 distinct candidates x 2 views


def test_candidates_are_pinned_before_they_are_compared():
    # The decodes are GOOD with the given cell 0 overwritten and, from the reversed board, with the given cell 5 too:
    # they differ, but pinned both are GOOD, one candidate scored once.
    bad = GOOD.copy()
    bad[0] = 4
    worse = bad.copy()
    worse[5] = 2
    out = run_ics_verifier(solver(lambda row: bad if row[0] == 2 else worse[::-1]), judge(known), line(X0), T=1,
                           verifier_T=1, batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [0])
    np.testing.assert_array_equal(out["ics/model/raw"], [GOOD])
    np.testing.assert_array_equal(out["ics/model/segs"], [2 * 1 + 1 * 2 * 1])     # 1 distinct candidate x 2 views


def test_unmapped_decodes_are_no_candidates_and_restatement_counts_may_differ():
    # Board 0 has both restatements, but its reversed decode cannot be mapped back; board 1 has only the identity, and
    # its decode cannot be mapped back either: it commits its input, pinned, with restatement -1.
    x1 = np.array([2, 3, 4, 1, 1, 3])
    lost = np.array([5, 5, 5, 5, 5, 5])
    task = line(X0, x1, views=[2, 1], lost=(lost,))
    script = lambda row: WRONG if np.array_equal(row, X0) else lost
    out = run_ics_verifier(solver(script), judge(known), task, T=2, verifier_T=1, batch=1)
    np.testing.assert_array_equal(out["ics/model/restatement"], [0, -1])
    np.testing.assert_array_equal(out["ics/model/raw"], [WRONG, x1])
    np.testing.assert_array_equal(out["ics/model/segs"], [2 * 2 + 2 * 1, 2])


def test_a_candidate_scoring_minus_inf_is_committed_over_an_unmapped_restatement():
    # The board's first decode cannot be mapped back; its second, WRONG, scores -inf: it has no restatement of its own
    # (Bare), or the verifier's q on it is -inf. WRONG is committed, from restatement 1, never the empty slot 0.
    lost = np.array([5, 5, 5, 5, 5, 5])
    script = lambda row: lost if row[0] == 2 else WRONG[::-1]
    for task, verifier, segs in ((Bare(make_pool([X0], [GOOD]), lost=(lost,)), judge(known), 2 * 1),
                                 (line(X0, lost=(lost,)), judge(lambda row: -np.inf), 2 * 1 + 2 * 1)):
        out = run_ics_verifier(solver(script), verifier, task, T=1, verifier_T=1, batch=8)
        np.testing.assert_array_equal(out["ics/model/restatement"], [1])
        np.testing.assert_array_equal(out["ics/model/raw"], [WRONG])
        np.testing.assert_array_equal(out["ics/model/segs"], [segs])


def test_boards_without_restatements_commit_their_inputs():
    # No board of the window restates: each commits its input, pinned, with restatement -1, and nothing is rolled.
    x1 = np.array([2, 3, 4, 1, 1, 3])
    out = run_ics_verifier(solver(lambda row: GOOD), judge(known), line(X0, x1, views=[0, 0]), T=1, verifier_T=1,
                           batch=8)
    np.testing.assert_array_equal(out["ics/model/restatement"], [-1, -1])
    np.testing.assert_array_equal(out["ics/model/raw"], [X0, x1])
    np.testing.assert_array_equal(out["ics/model/segs"], [0, 0])


def test_each_distinct_candidate_is_scored_once():
    # Restatements 0 and 2 are the board, 1 and 3 its reversal: the decodes give WRONG, GOOD, WRONG, GOOD. The two
    # distinct candidates show the verifier their 4 views once each, and GOOD is committed from restatement 1.
    seen = []
    q = lambda row: seen.append(tuple(row)) or known(row)
    script = lambda row: WRONG if row[0] == 2 else GOOD[::-1]
    once = run_ics_verifier(solver(script), judge(q), line(X0, views=[4]), T=1, verifier_T=1, batch=1)
    assert sorted(seen) == sorted(tuple(y[::s]) for y in (WRONG, GOOD) for s in (1, -1, 1, -1))
    np.testing.assert_array_equal(once["ics/model/restatement"], [1])
    np.testing.assert_array_equal(once["ics/model/raw"], [GOOD])
    np.testing.assert_array_equal(once["ics/model/segs"], [4 * 1 + 2 * 4 * 1])     # 2 distinct candidates x 4 views
