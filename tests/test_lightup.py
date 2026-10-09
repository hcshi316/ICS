import copy

import numpy as np
import pytest

from fakes import assert_identical, digest, make_pool
from ics.data import load_pool
from ics.tasks import base, lightup, make_task
from ics.tasks.base import on_canvas
from ics.tasks.lightup import BULB, EMPTY, W0, WALL, LightUp, akari_valid

DTYPES = (np.int64, np.int32, np.int16, np.uint8)
# (h, w, r0, c0) of synthetic boards: square and not, on the canvas edge, two whose ring the canvas cuts (25 cells
# long), and one whose dims run past the canvas (numpy crops it to 16 rows)
PLACEMENTS = [(1, 1, 1, 1), (2, 4, 1, 1), (4, 2, 1, 1), (3, 3, 4, 7), (5, 8, 1, 1), (7, 11, 2, 0), (24, 24, 1, 1),
              (1, 24, 1, 1), (24, 1, 0, 25), (25, 3, 1, 1), (3, 25, 0, 1), (20, 3, 10, 0)]
SMALL = [(1, 1, 1, 1), (3, 5, 1, 1), (10, 2, 1, 1), (2, 11, 0, 1), (9, 4, 2, 7)]     # on a 12 x 12 canvas
# per canvas side: every tag, restated row and decode mapped back of random_boards, its rows and decodes of each dtype
RESTATEMENTS = {26: "c4b6e687552c501af826ff7c488a63d1d8298921b95ff3a128c836281703c22b",
                12: "dfdf82f6a0f80323bb6672932ff461f222122743300cfe4918ee2bd3cada38aa"}


def random_boards(side=26, placements=PLACEMENTS, copies=3):
    """A Light-Up task on side x side canvases of seeded random tokens 0..8, `copies` boards at each placement, the
    placements taken in turn."""
    dims = np.array([p for _ in range(copies) for p in placements])
    X = np.random.default_rng(0).integers(0, BULB + 1, (len(dims), side * side))
    return LightUp(make_pool(X, X, dims=dims))

# 2x4 board with a wall numbered 1 at (0, 1) and a plain wall at (1, 2). It is not square, and each of the 8 dihedral
# maps changes it, so a restatement mapped back the wrong way cannot pass. Its only solution: bulbs at (0, 2), (1, 0),
# (1, 3).
BOARD = np.array([[EMPTY, W0 + 1, EMPTY, EMPTY],
                  [EMPTY, EMPTY, WALL, EMPTY]])
SOLVED = BOARD.copy()
for r, c in ((0, 2), (1, 0), (1, 3)):
    SOLVED[r, c] = BULB


def task():
    x, y = on_canvas(BOARD, 26, WALL), on_canvas(SOLVED, 26, WALL)
    return LightUp(make_pool([x], [y], dims=np.array([(2, 4, 1, 1)])))


def test_akari_rules():
    assert akari_valid(BOARD, SOLVED)
    y = SOLVED.copy(); y[0, 2] = EMPTY
    assert not akari_valid(BOARD, y)                   # number broken and a cell unlit
    y = SOLVED.copy(); y[0, 3] = BULB
    assert not akari_valid(BOARD, y)                   # two bulbs see each other (every cell lit, the 1 satisfied)
    y = BOARD.copy(); y[0, 3] = y[1, 0] = BULB
    assert not akari_valid(BOARD, y)                   # only the number broken: all lit, no two bulbs see each other
    y = SOLVED.copy(); y[1, 3] = EMPTY
    assert not akari_valid(BOARD, y)                   # only a cell unlit: no two bulbs see each other, the 1 holds


def test_valid_raw_and_pinned():
    t = task()
    assert t.valid(0, t.Y[0])
    y = t.Y[0].copy()
    y[1 * 26 + 1] = WALL                               # junk on an empty cell (board cell (0, 0))
    assert not t.valid(0, y) and t.valid(0, t.pin(0, y))
    y = t.Y[0].copy()
    y[1 * 26 + 2] = EMPTY                              # the numbered wall (board cell (0, 1)) erased
    assert not t.valid(0, y) and t.valid(0, t.pin(0, y))
    assert t.pin(0, y)[1 * 26 + 2] == W0 + 1


def test_restatements_round_trip():
    t = task()
    restatements = t.restatements(0, t.X[0])
    assert [tag[:2] for tag, _ in restatements] == [(k, m) for k in range(4) for m in (False, True)]
    for (k4, mir, dims), row in restatements:
        board, solved = np.rot90(BOARD, k4), np.rot90(SOLVED, k4)      # rotate, then mirror
        if mir:
            board, solved = np.fliplr(board), np.fliplr(solved)
        np.testing.assert_array_equal(row, on_canvas(board, 26, WALL))
        assert dims == ((4, 2, 1, 1) if k4 % 2 else (2, 4, 1, 1))
        restated_answer = on_canvas(solved, 26, WALL)
        np.testing.assert_array_equal(t.restate_back(0, (k4, mir, dims), restated_answer), t.Y[0])


@pytest.mark.parametrize("side, placements", [(26, PLACEMENTS), (12, SMALL)])
def test_the_restatements_are_pinned(side, placements):
    t = random_boards(side, placements)
    rng, arrays = np.random.default_rng(1), []
    for i in range(len(t)):
        row, decode = (rng.integers(0, BULB + 1, side * side).astype(DTYPES[(i + j) % 4]) for j in range(2))
        for tag, restated in t.restatements(i, row):
            arrays += [np.array([*tag[:2], *tag[2]]), restated, t.restate_back(i, tag, decode)]
    assert digest(arrays) == RESTATEMENTS[side]


def test_a_board_too_large_for_a_turned_canvas_raises():
    # a board 26 cells long leaves no room for its ring
    t = random_boards(placements=[(26, 3, 0, 0), (3, 26, 0, 0)], copies=1)
    for i in range(len(t)):
        with pytest.raises(ValueError):
            t.restatements(i, t.X[i])


def test_returned_arrays_are_the_callers():
    # A caller may change every array a call returns; the later calls still return the first's.
    t = random_boards(copies=1)
    decode = np.random.default_rng(1).integers(0, BULB + 1, 26 * 26)
    for i in range(len(t)):
        want = copy.deepcopy(t.restatements(i, t.X[i]))
        want_back = copy.deepcopy([t.restate_back(i, tag, decode) for tag, _ in want])
        for tag, restated in t.restatements(i, t.X[i]):
            restated += 1
            t.restate_back(i, tag, decode)[:] = 0
        assert_identical(t.restatements(i, t.X[i]), want)
        assert_identical([t.restate_back(i, tag, decode) for tag, _ in want], want_back)


def test_maps_are_kept_once_per_placement(monkeypatch):
    built = []
    monkeypatch.setattr(lightup, "canvas_views", lambda side, dims: built.append(dims) or base.canvas_views(side, dims))
    t = random_boards()
    for _ in range(2):
        for i in range(len(t)):
            for tag, restated in t.restatements(i, t.X[i]):
                t.restate_back(i, tag, restated)
    assert sorted(built) == sorted(PLACEMENTS)                          # at the first call per placement only
    for _tags, cells, gather, back in t._maps.values():
        assert cells.dtype == gather.dtype == back.dtype == np.int16
        assert cells.nbytes + gather.nbytes + back.nbytes == 2 * (8 * 26 * 26 + 9 * len(cells))   # 10,816 + 18 n


def test_hypothesis_cells_and_alternatives():
    t = task()
    cells = t.editable(0, t.X[0])
    assert cells == [(r + 1) * 26 + c + 1 for r, c in ((0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3))]
    assert t.alternatives(0, cells[0], current=EMPTY, ranked=[1, 8], answer=None) == [BULB]
    assert t.alternatives(0, cells[0], current=BULB, ranked=[8, 1], answer=None) == []
    assert isinstance(make_task("lightup", t.pool), LightUp)


def test_needs_board_placement():
    x = on_canvas(BOARD, 26, WALL)
    with pytest.raises(ValueError, match="all__dims.npy"):
        LightUp(make_pool([x], [x]))


@pytest.mark.data
def test_labels_valid_and_restatable(dataset):
    pool = load_pool(dataset("lightup"))
    t = LightUp(pool)
    idx = np.arange(len(pool))
    labels = np.where(pool.labels < 0, pool.inputs, pool.labels)
    assert t.check("raw", idx, labels).all()
    for i in range(len(pool)):
        board = t.board(i, labels[i])
        for tag, row in t.restatements(i, labels[i]):  # k4 quarter-turns, then mir: a mirror; at (1, 1) of a canvas
            k4, mir, (h, w, _, _) = tag
            view = np.fliplr(np.rot90(board, k4)) if mir else np.rot90(board, k4)
            np.testing.assert_array_equal(row.reshape(t.side, t.side)[1:h + 1, 1:w + 1], view)
            np.testing.assert_array_equal(t.board(i, t.restate_back(i, tag, row)), board)
