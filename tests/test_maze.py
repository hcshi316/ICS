import copy

import numpy as np
import pytest

from fakes import assert_identical, make_pool, maze_pool, solve_maze
from ics.data import load_pool
from ics.tasks import make_task, maze
from ics.tasks.maze import EMPTY, PATH, SIDE, START, WALL, Maze, rule_valid


def random_mazes(n=40):
    """A Maze task over n seeded random rows of tokens 0..5 (restatements do not look at the maze)."""
    X = np.random.default_rng(0).integers(0, PATH + 1, (n, SIDE * SIDE))
    return Maze(make_pool(X, X))


def open3():
    """Rows 1..3, columns 1..10 open; start (1,1), goal (1,10): the unique shortest path runs along row 1."""
    return Maze(maze_pool([1, 2, 3], (1, 1), (1, 10)))


def cell(r, c):
    return r * 30 + c


def test_rule_verdicts():
    t = open3()
    x, sol = t.X[0], t.Y[0]
    assert rule_valid(x, sol) == "OK" and t.valid(0, sol)
    y = sol.copy(); y[cell(3, 5)] = PATH
    assert rule_valid(x, y) == "stray"
    y = sol.copy(); y[cell(1, 5)] = EMPTY
    assert rule_valid(x, y) == "disc"
    y = sol.copy(); y[cell(1, 5)] = EMPTY; y[cell(2, 4)] = y[cell(2, 5)] = y[cell(2, 6)] = PATH
    y[cell(1, 4)] = y[cell(1, 6)] = PATH
    assert rule_valid(x, y) == "longer"
    y = sol.copy(); y[cell(0, 5)] = PATH
    assert rule_valid(x, y) == "given"


def test_any_shortest_path_is_valid():
    t = Maze(maze_pool([1, 2], (1, 1), (2, 10)))       # 2 x 10 open: many shortest paths
    y = t.X[0].copy()
    for c in range(2, 11):
        y[cell(1, c)] = PATH                           # along row 1, then down at column 10
    assert t.valid(0, y) and t.valid(0, t.Y[0])


def test_stray_is_judged_through_path_cells():
    # (2, 5) lies on a shortest start-goal path of the open 2 x 10 room, but not on one through the marked path cells
    t = Maze(maze_pool([1, 2], (1, 1), (2, 10)))
    y = t.X[0].copy()
    for c in range(2, 11):
        y[cell(1, c)] = PATH                           # along row 1, then down at column 10
    y[cell(2, 5)] = PATH
    assert rule_valid(t.X[0], y) == "stray"


def test_union_of_shortest_paths_is_valid():
    t = Maze(maze_pool([1, 2], (1, 1), (2, 10)))       # 2 x 10 open: every open cell lies on a shortest path
    y = t.X[0].copy()
    y[y == EMPTY] = PATH                               # every shortest path marked at once
    assert rule_valid(t.X[0], y) == "OK" and t.valid(0, y)


def test_pin_restores_givens_and_clears_junk():
    t = open3()
    y = t.Y[0].copy()
    y[cell(1, 1)] = PATH                               # start overwritten
    y[cell(3, 3)] = START                              # junk token on an empty cell
    assert not t.valid(0, y) and t.valid(0, t.pin(0, y))
    p = t.pin(0, y)
    assert p[cell(1, 1)] == START and p[cell(3, 3)] == EMPTY
    assert not t.consistent(0, y) and t.consistent(0, p)
    y = t.Y[0].copy(); y[cell(3, 3)] = WALL            # junk off the path: raw-valid
    assert t.valid(0, y) and not t.consistent(0, y)


def test_restatements_round_trip():
    t = open3()
    restatements = t.restatements(0, t.X[0].astype(np.int16))
    assert [tag for tag, _ in restatements] == [(k, m) for k in range(4) for m in (False, True)]
    board = t.X[0].reshape(30, 30)
    for (k4, mir), row in restatements:                # k4 quarter-turns of np.rot90, then mir: a left-right mirror
        view = np.rot90(board, k4)
        np.testing.assert_array_equal(row, (np.fliplr(view) if mir else view).flatten())
        restated_answer = solve_maze(row)
        back = t.restate_back(0, (k4, mir), restated_answer)
        assert row.dtype == back.dtype == np.int16                     # a row's dtype is kept
        np.testing.assert_array_equal(back, t.Y[0])
        assert rule_valid(row, restated_answer) == "OK"


def test_restatements_are_copies():
    t = open3()
    before = t.X[0].copy()
    (_tag, identity) = t.restatements(0, t.X[0])[0]
    identity[0] = 99                                   # writing into a restatement must not reach the pool
    back = t.restate_back(0, (0, False), t.X[0])
    back[1] = 99
    np.testing.assert_array_equal(t.X[0], before)


def test_returned_arrays_are_the_callers():
    # A caller may change every array a call returns; the later calls still return the first's.
    t = random_mazes(n=8)
    for i in range(len(t)):
        want = copy.deepcopy(t.restatements(i, t.X[i]))
        want_back = copy.deepcopy([t.restate_back(i, tag, t.Y[i]) for tag, _ in want])
        for tag, restated in t.restatements(i, t.X[i]):
            restated += 1
            t.restate_back(i, tag, t.Y[i])[:] = 0
        assert_identical(t.restatements(i, t.X[i]), want)
        assert_identical([t.restate_back(i, tag, t.Y[i]) for tag, _ in want], want_back)


def test_views_are_kept_once_for_every_maze():
    # restatement k of a row is row[VIEWS[k]], and BACKS[k] maps an answer to it back: int16, 28,800 bytes in all
    assert maze.VIEWS.dtype == maze.BACKS.dtype == np.int16
    assert maze.VIEWS.shape == maze.BACKS.shape == (8, SIDE * SIDE)
    assert (np.take_along_axis(maze.VIEWS, maze.BACKS, 1) == np.arange(SIDE * SIDE)).all()


def test_hypothesis_cells_and_alternatives():
    t = open3()
    empty = np.flatnonzero(t.X[0] == EMPTY).tolist()
    assert t.editable(0, t.X[0]) == empty
    hinted = t.X[0].copy(); hinted[empty[0]] = PATH
    assert t.editable(0, hinted) == empty[1:]
    # path hypotheses only: PATH for an empty cell, even where the model ranks WALL first, and nothing for a cell the
    # parent's answer already marks as path; WALL is never proposed
    assert t.alternatives(0, empty[0], current=EMPTY, ranked=[1, 5, 2], answer=None) == [PATH]
    assert t.alternatives(0, empty[0], current=EMPTY, ranked=[5, 1, 2], answer=None) == [PATH]
    assert t.alternatives(0, empty[0], current=PATH, ranked=[5, 1, 2], answer=None) == []
    assert isinstance(make_task("maze", t.pool), Maze)


@pytest.mark.data
def test_every_label_is_valid(dataset):
    pool = load_pool(dataset("maze"))
    t = Maze(pool)
    assert t.check("raw", np.arange(len(pool)), pool.labels).all()
