import numpy as np
import torch

from fakes import SOLUTION, ScriptedRestarts, sudoku_pool
from ics.tasks.sudoku import Sudoku
from ics_baselines.eqr.predict import restarts, run_eqr

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]
BROKEN = SOLUTION.copy()
BROKEN[1] = SOLUTION[5]         # raw-invalid; pinned-valid where cell 1 is a given (boards 0 and 2)

# board (its blank cells) -> per restart r: (decode, final q_halt)
SCRIPT = {(0,): [(WRONG, 0.2), (BROKEN, 0.9), (SOLUTION, 0.9)],
          (0, 1): [(WRONG, -1.0), (WRONG, -1.0), (BROKEN, -1.0)],
          (0, 2): [(SOLUTION, 0.1), (SOLUTION, 0.5), (WRONG, 0.3)]}


def scripted():
    return ScriptedRestarts(11, 3, lambda row, r: SCRIPT[tuple(np.flatnonzero(row == 1))][r])


def pool3():
    return sudoku_pool([(0,), (0, 1), (0, 2)])


def test_rows_and_selection_rules():
    out = run_eqr(scripted(), Sudoku(pool3()), R=3, steps=2, batch=8)
    np.testing.assert_array_equal(out["eqr/model/restart"], [1, 0, 1])         # largest final q; the first on ties
    np.testing.assert_array_equal(out["eqr/model/raw"], [BROKEN, WRONG, SOLUTION])
    np.testing.assert_array_equal(out["eqr/cert/raw/restart"], [2, -1, 0])
    np.testing.assert_array_equal(out["eqr/cert/raw"], [SOLUTION, WRONG, SOLUTION])
    np.testing.assert_array_equal(out["eqr/cert/pinned/restart"], [1, -1, 0])
    np.testing.assert_array_equal(out["eqr/cert/pinned"], [BROKEN, WRONG, SOLUTION])
    assert set(out) == {"eqr/model/raw", "eqr/model/pinned", "eqr/model/restart", "eqr/cert/raw", "eqr/cert/pinned",
                        "eqr/cert/raw/restart", "eqr/cert/pinned/restart"}


def test_restarts_are_board_major_and_share_the_block_generator():
    model = scripted()
    out = run_eqr(model, Sudoku(pool3()), R=3, steps=2, batch=8)
    assert len(model.gens) == 3 and isinstance(model.gens[0], torch.Generator)  # the initial state, then 2 steps:
    assert all(g is model.gens[0] for g in model.gens)                          # all on the block's one generator
    assert model.rows == [8 * 3, 8 * 3]                        # 3 boards padded to 8, each repeated R = 3 times
    assert out["eqr/model/raw"].shape == (3, 81)               # the pad rows are dropped


def test_restarts_returns_every_candidate():
    x = torch.as_tensor(pool3().inputs, dtype=torch.int32)
    decs, qs = restarts(scripted(), x, 3, 1, None)
    assert decs.shape == (3, 3, 81) and decs.dtype == np.int16 and qs.shape == (3, 3) and qs.dtype == np.float32
    np.testing.assert_array_equal(decs[:, 0], [WRONG, BROKEN, SOLUTION])      # board 0's restarts 0, 1, 2
    np.testing.assert_allclose(qs[:, 2], [0.1, 0.5, 0.3], rtol=1e-6)
