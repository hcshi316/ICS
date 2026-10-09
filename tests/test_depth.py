import numpy as np
import pytest

from fakes import SOLUTION, ScriptedTRM, sudoku_pool
from ics.methods.trm import run_trm
from ics.tasks.base import SCORINGS
from ics.tasks.sudoku import Sudoku

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]                  # cell 0 (a blank) duplicates row 0: invalid raw and pinned


def test_depth_rows():
    pool = sudoku_pool([(0, 20), (0, 30)])
    task = Sudoku(pool)
    a = pool.inputs[0].tobytes()

    def script(row, t, zl):
        if row.tobytes() == a:          # wrong until t=30; q fires wrongly at t=10 and peaks at t=40
            ans = SOLUTION if t >= 30 else WRONG
            q = {10: 0.5, 40: 2.0}.get(t, -1.0)
        else:                           # always right, q never fires
            ans, q = SOLUTION, -1.0
        return ans, q, 1.0

    out = run_trm(ScriptedTRM(11, script), task, T=50, batch=8, standard_at=16)
    for s in ("raw", "pinned"):
        np.testing.assert_array_equal(out[f"standard/model/{s}"], [WRONG, SOLUTION])
        np.testing.assert_array_equal(out[f"standard/cert/{s}"], [WRONG, SOLUTION])
        np.testing.assert_array_equal(out[f"greedy_depth/model/{s}"], [WRONG, SOLUTION])
        np.testing.assert_array_equal(out[f"depth_scaling/model/{s}"], [SOLUTION, SOLUTION])
        np.testing.assert_array_equal(out[f"greedy_depth/cert/{s}"], [SOLUTION, SOLUTION])
        np.testing.assert_array_equal(out[f"depth_scaling/cert/{s}"], [SOLUTION, SOLUTION])
        np.testing.assert_array_equal(out[f"greedy_depth/cert/{s}/segment"], [30, 1])
        assert out[f"depth_scaling/cert/{s}/segment"] is out[f"greedy_depth/cert/{s}/segment"]
    np.testing.assert_array_equal(out["greedy_depth/model/segment"], [10, 50])
    np.testing.assert_array_equal(out["depth_scaling/model/segment"], [40, 1])
    assert out["standard/model/raw"].dtype == np.int16
    rows = ("standard", "greedy_depth", "depth_scaling")
    assert set(out) == ({f"{row}/{regime}/{s}" for row in rows for regime in ("model", "cert") for s in SCORINGS}
                        | {f"{row}/model/segment" for row in rows[1:]}
                        | {f"{row}/cert/{s}/segment" for row in rows[1:] for s in SCORINGS})
    one_per_chunk = run_trm(ScriptedTRM(11, script), task, T=50, batch=1, standard_at=16)
    assert one_per_chunk.keys() == out.keys()
    for k in out:                                   # chunking the pool does not change any result
        np.testing.assert_array_equal(one_per_chunk[k], out[k])


def test_pinned_certificate_can_accept_earlier():
    pool = sudoku_pool([(0,)])
    task = Sudoku(pool)
    broken_given = SOLUTION.copy()
    broken_given[1] = SOLUTION[5]       # a given changed: raw-invalid, pinned-valid

    def script(row, t, zl):
        return (broken_given if t < 5 else SOLUTION), -1.0, 1.0

    out = run_trm(ScriptedTRM(11, script), task, T=8, batch=8, standard_at=2)
    np.testing.assert_array_equal(out["greedy_depth/cert/raw/segment"], [5])
    np.testing.assert_array_equal(out["greedy_depth/cert/pinned/segment"], [1])


def test_never_valid_falls_back_to_last_decode():
    task = Sudoku(sudoku_pool([(0,)]))
    out = run_trm(ScriptedTRM(11, lambda row, t, zl: (WRONG, -1.0, 1.0)), task, T=4, batch=8, standard_at=2)
    np.testing.assert_array_equal(out["greedy_depth/cert/raw/segment"], [-1])
    np.testing.assert_array_equal(out["depth_scaling/cert/raw"], [WRONG])


def test_each_row_commits_the_decode_of_its_segment():
    task = Sudoku(sudoku_pool([(0,)]))

    def at(t):                          # segment t's decode: token t + 2 in cell 0 (the solution's is 2), never valid
        y = SOLUTION.copy()
        y[0] = t + 2
        return y

    q = {2: 0.0, 3: 0.5, 5: 2.0}
    out = run_trm(ScriptedTRM(11, lambda row, t, zl: (at(t), q.get(t, -1.0), 1.0)), task, T=7, batch=8, standard_at=4)
    np.testing.assert_array_equal(out["standard/model/raw"], [at(4)])
    np.testing.assert_array_equal(out["greedy_depth/model/raw"], [at(3)])     # q = 0 at t=2 does not fire
    np.testing.assert_array_equal(out["depth_scaling/model/raw"], [at(5)])
    np.testing.assert_array_equal(out["depth_scaling/cert/raw"], [at(7)])     # never valid: segment T's decode


def test_a_later_chunk_is_certified_against_its_own_boards():
    task = Sudoku(sudoku_pool([(30,), (20,)]))
    off = SOLUTION.copy()
    off[20] = SOLUTION[21]              # raw-invalid; pinned-valid only on board 0, where cell 20 is a given
    out = run_trm(ScriptedTRM(11, lambda row, t, zl: (off, -1.0, 1.0)), task, T=2, batch=1, standard_at=1)
    np.testing.assert_array_equal(out["greedy_depth/cert/pinned/segment"], [1, -1])
    np.testing.assert_array_equal(out["greedy_depth/cert/pinned"], [off, off])  # board 1 falls back to its own decode


@pytest.mark.parametrize("T, standard_at, batch", [(1, 2, 8), (2, 0, 8), (2, 1, 0)],
                         ids=["T_below_standard_at", "standard_at_0", "batch_0"])
def test_rejects_out_of_range_settings(T, standard_at, batch):
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION, -1.0, 1.0))
    with pytest.raises(ValueError, match="run_trm needs"):
        run_trm(model, Sudoku(sudoku_pool([(0,)])), T=T, batch=batch, standard_at=standard_at)


def test_standard_at_may_be_the_last_segment():
    model = ScriptedTRM(11, lambda row, t, zl: (SOLUTION if t == 2 else WRONG, -1.0, 1.0))
    out = run_trm(model, Sudoku(sudoku_pool([(0,)])), T=2, batch=8, standard_at=2)
    np.testing.assert_array_equal(out["standard/model/raw"], [SOLUTION])        # segment 2's decode
