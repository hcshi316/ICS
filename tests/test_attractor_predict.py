import numpy as np
import torch

from fakes import SOLUTION, ScriptedAttractor, sudoku_pool
from ics.seeding import ATTRACTOR_RESTARTS, block_seed, torch_generator
from ics.tasks.sudoku import Sudoku
from ics_baselines.attractor.predict import restarts, run_attractor

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]
BROKEN = SOLUTION.copy()
BROKEN[1] = SOLUTION[5]         # raw-invalid; pinned-valid where cell 1 is a given (boards 0 and 2)

# board (its blank cells) -> per restart k: (decode, final q_halt)
SCRIPT = {(0,): [(WRONG, 0.2), (BROKEN, 0.9), (SOLUTION, 0.9)],
          (0, 1): [(WRONG, -1.0), (WRONG, -1.0), (BROKEN, -1.0)],
          (0, 2): [(SOLUTION, 0.1), (SOLUTION, 0.5), (WRONG, 0.3)]}


def scripted():
    return ScriptedAttractor(11, 3, lambda row, k: SCRIPT[tuple(np.flatnonzero(row == 1))][k])


def pool3():
    return sudoku_pool([(0,), (0, 1), (0, 2)])


def test_rows_and_selection_rules():
    out = run_attractor(scripted(), Sudoku(pool3()), R=3, sigma=0.5, segments=2, batch=8)
    np.testing.assert_array_equal(out["attractor/model/restart"], [1, 0, 1])   # largest final q; the first on ties
    np.testing.assert_array_equal(out["attractor/model/raw"], [BROKEN, WRONG, SOLUTION])
    np.testing.assert_array_equal(out["attractor/cert/raw/restart"], [2, -1, 0])
    np.testing.assert_array_equal(out["attractor/cert/raw"], [SOLUTION, WRONG, SOLUTION])
    np.testing.assert_array_equal(out["attractor/cert/pinned/restart"], [1, -1, 0])
    np.testing.assert_array_equal(out["attractor/cert/pinned"], [BROKEN, WRONG, SOLUTION])
    assert len(out) == 7


def test_restart_zero_is_clean_and_restart_k_draws_its_perturbation_from_the_block_generator():
    model = scripted()
    run_attractor(model, Sudoku(pool3()), R=3, sigma=0.5, segments=1, batch=8)
    replay = torch_generator(block_seed(0, ATTRACTOR_RESTARTS, 0), "cpu")
    assert model.starts[0] == (3, None, None)
    for k in (1, 2):
        dH = 0.5 * torch.randn(4, generator=replay)
        dL = 0.5 * torch.randn(4, generator=replay)
        B, gH, gL = model.starts[k]
        assert B == 3 and torch.equal(gH, dH) and torch.equal(gL, dL)


def test_a_block_holds_exactly_its_boards():
    model = scripted()
    out = run_attractor(model, Sudoku(pool3()), R=3, sigma=0.5, segments=2, batch=8)
    assert model.rows == [3] * (3 * 2)                   # never padded: 3 restarts x 2 segments, all on the 3 boards
    assert out["attractor/model/raw"].shape == (3, 81)


def test_sigma_zero_draws_nothing():
    model = scripted()
    run_attractor(model, Sudoku(pool3()), R=3, sigma=0.0, segments=1, batch=8)
    assert [(dH, dL) for _, dH, dL in model.starts] == [(None, None)] * 3


def test_restarts_returns_every_candidate():
    x = torch.as_tensor(pool3().inputs, dtype=torch.int32)
    decs, qs = restarts(scripted(), x, 3, 0.5, 1, torch.Generator())
    assert decs.shape == (3, 3, 81) and decs.dtype == np.int16 and qs.shape == (3, 3) and qs.dtype == np.float32
    np.testing.assert_array_equal(decs[:, 0], [WRONG, BROKEN, SOLUTION])
