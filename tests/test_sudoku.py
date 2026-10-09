import copy
import random

import numpy as np
import pytest

from fakes import SOLUTION, assert_identical, digest, make_pool, sudoku_pool
from ics.data import load_pool
from ics.tasks import sudoku
from ics.tasks.base import rank_order
from ics.tasks.sudoku import BLANK, Sudoku, make_aug

HOLES = (0, 10, 40, 80)
DTYPES = (np.int64, np.int32, np.int16, np.uint8)
# per restatement count: every tag, restated row and decode mapped back of random_boards (blocks of 64 rows, seed 7),
# its rows of each integer dtype
RESTATEMENTS = {1: "64d3fc132ca7f1078eb89aabcb70cf4fc3b2c8d458237d3f216842b1d0ee1bab",
                2: "938dd9fce47d02b142605eb2a3afa8bd4c0a5c11c9099183a4be5cfdef9f4812",
                5: "c308cad8b3c3934c756490164a2c71d3dee50f3b0e90913bc62bd56ee68a67e4",
                32: "ede90a541e4690e77451f97b37e7c6c1a0a37848d1c28581367b1443b043635b"}


def random_boards(n=300, start=20, block=64, **kw):
    """A Sudoku task over rows start..n-1 of n seeded random rows of tokens 0..10, in blocks of `block` rows."""
    X = np.random.default_rng(0).integers(0, 11, (n, 81))
    return Sudoku(make_pool(X, X).take(slice(start, n)), block=block, **kw)


def task(n_restatements=32, seed=7):
    return Sudoku(sudoku_pool([HOLES]), n_restatements=n_restatements, seed=seed)


def test_valid_raw_and_pinned():
    t = task()
    assert t.valid(0, SOLUTION)
    assert not t.valid(0, t.X[0])                           # blanks are not digits
    y = SOLUTION.copy(); y[1] = SOLUTION[2]                 # duplicate digit in row 0 (cell 1 is a given)
    assert not t.valid(0, y) and t.valid(0, t.pin(0, y))    # pinning restores the given
    y = SOLUTION.copy(); y[0], y[10] = y[10], y[0]          # swap two blank cells: breaks rows
    assert not t.valid(0, y) and not t.valid(0, t.pin(0, y))


def test_pin_and_consistent():
    t = task()
    y = np.full(81, 5)
    p = t.pin(0, y)
    giv = t.X[0] != BLANK
    np.testing.assert_array_equal(p[giv], t.X[0][giv])
    np.testing.assert_array_equal(p[~giv], 5)
    assert t.consistent(0, SOLUTION) and not t.consistent(0, y)


def test_batch_versions_agree():
    t = Sudoku(sudoku_pool([HOLES, (1, 2)]))
    rng = np.random.default_rng(0)
    Y = np.stack([SOLUTION, SOLUTION.copy(), rng.integers(1, 11, 81), SOLUTION])
    Y[1][0] = 3                                  # cell 0 is a given of board 1: raw-invalid, pinned-valid
    idx = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(t.valid_batch(idx, Y), [t.valid(int(i), y) for i, y in zip(idx, Y)])
    np.testing.assert_array_equal(t.check("pinned", idx, Y),
                                  [t.valid(int(i), t.pin(int(i), y)) for i, y in zip(idx, Y)])
    np.testing.assert_array_equal(t.check("raw", idx, Y), [True, False, False, True])
    np.testing.assert_array_equal(t.check("pinned", idx, Y), [True, True, False, True])
    with pytest.raises(ValueError):
        t.check("exact", idx, Y)


def test_restatements_round_trip():
    t = task()
    restatements = t.restatements(0, t.X[0])
    assert len(restatements) == 32 and restatements[0][0] is None
    for tag, row in restatements[1:]:
        pos, inv = tag
        dmap = np.arange(12)
        dmap[inv[2:11]] = np.arange(2, 11)
        restated_solution = np.zeros(81, np.int64)
        restated_solution[pos] = dmap[SOLUTION]
        np.testing.assert_array_equal(t.restate_back(0, tag, restated_solution), SOLUTION)
        given = row != BLANK
        np.testing.assert_array_equal(restated_solution[given], row[given])
        assert Sudoku(make_pool([restated_solution], [restated_solution]), n_restatements=1).valid(0, restated_solution)


def test_restatements_are_seeded_per_block():
    pool = sudoku_pool([HOLES] * 4)                               # four identical inputs, rows 0..3
    restated = lambda t, i: [row for _tag, row in t.restatements(i, t.X[i])]
    same = lambda a, b: all(np.array_equal(x, y) for x, y in zip(a, b))
    whole = Sudoku(pool, block=2)
    assert same(restated(whole, 0), restated(Sudoku(pool, block=2), 0))      # deterministic
    assert not same(restated(whole, 0)[1:], restated(whole, 1)[1:])           # boards of one block differ
    shard = Sudoku(pool.take(slice(2, 4)), block=2)                           # the second block on its own
    assert same(restated(whole, 2), restated(shard, 0)) and same(restated(whole, 3), restated(shard, 1))
    assert not same(restated(whole, 0)[1:], restated(shard, 0)[1:])           # blocks differ


def test_block_generators_keep_their_values():
    # Board i's generator is random.Random(the sub-seed that default_rng([seed, 1, first row of i's block]) deals it).
    # Sudoku results depend on these exact values, whatever builds the seed.
    t = Sudoku(sudoku_pool([HOLES] * 3), seed=5, block=2)
    for i in range(3):
        start = i - i % 2
        sub_seed = np.random.default_rng([5, 1, start]).integers(0, 2 ** 62, size=2)[i - start]
        assert t._rng(i).getstate() == random.Random(int(sub_seed)).getstate()


@pytest.mark.parametrize("n_restatements", [1, 2, 5, 32])
def test_the_restatement_draws_are_pinned(n_restatements):
    t = random_boards(n_restatements=n_restatements)
    decodes, arrays = np.random.default_rng(1).integers(0, 11, (len(t), 81)), []
    for i in range(len(t)):
        for tag, restated in t.restatements(i, t.X[i].astype(DTYPES[i % 4])):
            arrays += [*(tag or ()), restated, t.restate_back(i, tag, decodes[i])]
    assert digest(arrays) == RESTATEMENTS[n_restatements]


def test_returned_arrays_are_the_callers():
    # A caller may change every array a call returns, tags included; the next call still returns the first's.
    t = random_boards(n=40)
    for i in range(len(t)):
        want = copy.deepcopy(t.restatements(i, t.X[i]))
        for tag, restated in t.restatements(i, t.X[i]):
            restated += 1
            for a in tag or ():
                a[:] = 0
        assert_identical(t.restatements(i, t.X[i]), want)


def test_a_boards_tables_are_drawn_once_and_kept_compact(monkeypatch):
    drawn = []
    monkeypatch.setattr(sudoku, "make_aug", lambda rng: drawn.append(rng) or make_aug(rng))
    t = random_boards(n=60)
    for _ in range(3):
        for i in range(len(t)):
            t.restatements(i, t.X[i])
    assert len(drawn) == 31 * len(t)                                 # at the first call per board only
    assert sorted(t._tables) == list(range(len(t)))
    for tables in t._tables.values():
        assert all(a.dtype == np.int8 for a in tables)
        assert sum(a.nbytes for a in tables) == 31 * (81 + 12 + 12)  # pos, digit map, inverse: 3,255 bytes a board


def test_hypothesis_cells_and_alternatives():
    t = task()
    assert t.editable(0, t.X[0]) == list(HOLES)
    hinted = t.X[0].copy(); hinted[10] = 3
    assert t.editable(0, hinted) == [0, 40, 80]
    assert t.alternatives(0, 0, current=4, ranked=[4, 9, 2], answer=None) == [9, 2, 3, 5, 6, 7, 8, 10]
    assert rank_order([5, 3, 8], [8, 1]) == [8, 5, 3]
    assert rank_order([2, 3, 4], [4, 4, 3]) == [4, 3, 2]          # a repeated ranked token keeps unranked ones last


@pytest.mark.data
def test_every_label_is_valid(dataset):
    pool = load_pool(dataset("sudoku"), limit=4096)
    t = Sudoku(pool)
    assert t.check("raw", np.arange(len(pool)), pool.labels).all()
