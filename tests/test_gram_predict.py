import numpy as np
import torch

from fakes import SOLUTION, ScriptedSampler, sudoku_pool
from ics.tasks.sudoku import Sudoku
from ics_baselines.gram.predict import run_gram, samples

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]
BROKEN = SOLUTION.copy()
BROKEN[1] = SOLUTION[5]         # raw-invalid; pinned-valid where cell 1 is a given (boards 0 and 2)

# board (its blank cells) -> per sample i: (decode, LPRM value logit)
SCRIPT = {(0,): [(WRONG, 0.1), (BROKEN, 0.9), (SOLUTION, 0.9), (SOLUTION, 0.2)],
          (0, 1): [(WRONG, 0.0), (WRONG, 0.0), (BROKEN, 0.0), (BROKEN, 0.0)],
          (0, 2): [(SOLUTION, 0.5), (WRONG, 0.7), (WRONG, 0.1), (SOLUTION, 0.7)]}


def scripted():
    return ScriptedSampler(11, lambda row, i, gen: SCRIPT[tuple(np.flatnonzero(row == 1))][i], N=4)


def constant():
    return ScriptedSampler(11, lambda row, i, gen: (SOLUTION, 0.0))


def pool3():
    return sudoku_pool([(0,), (0, 1), (0, 2)])


def test_rows_and_selection_rules():
    out = run_gram(scripted(), Sudoku(pool3()), N=4, D=2, batch=8)
    np.testing.assert_array_equal(out["gram_lprm/model/sample"], [1, 0, 1])       # largest value, the first on ties
    np.testing.assert_array_equal(out["gram_lprm/model/raw"], [BROKEN, WRONG, WRONG])
    np.testing.assert_array_equal(out["gram_majority/model/sample"], [2, 0, 0])   # most frequent; ties to the first
    np.testing.assert_array_equal(out["gram_majority/model/pinned"], [SOLUTION, WRONG, SOLUTION])
    for row in ("gram_lprm", "gram_majority"):                                    # both rows share the certificate
        np.testing.assert_array_equal(out[f"{row}/cert/raw/sample"], [2, -1, 0])
        np.testing.assert_array_equal(out[f"{row}/cert/raw"], [SOLUTION, WRONG, SOLUTION])
        np.testing.assert_array_equal(out[f"{row}/cert/pinned/sample"], [1, -1, 0])
        np.testing.assert_array_equal(out[f"{row}/cert/pinned"], [BROKEN, WRONG, SOLUTION])
    assert out["gram_lprm/model/raw"].dtype == np.int16 and len(out) == 14


def test_samples_returns_every_decode_and_the_sigmoid_of_v():
    x = torch.as_tensor(pool3().inputs, dtype=torch.int32)
    decs, values = samples(scripted(), x, 4, 1, lambda i: i)
    assert decs.shape == (4, 3, 81) and decs.dtype == np.int16 and values.dtype == np.float32
    np.testing.assert_array_equal(decs[1, 0], BROKEN)
    np.testing.assert_allclose(values[:, 0], 1 / (1 + np.exp(-np.array([0.1, 0.9, 0.9, 0.2]))), rtol=1e-6)


def test_each_sample_runs_D_steps_on_its_own_generator():
    model = constant()
    run_gram(model, Sudoku(pool3()), N=3, D=2, batch=8)
    gens = [g for _, g in model.calls]
    assert all(gens[j] is gens[j - j % 2] for j in range(6))                 # a sample's 2 steps share a generator
    assert len({g.get_state().numpy().tobytes() for g in gens[::2]}) == 3     # and the 3 samples' generators differ


def test_a_short_block_is_padded_like_a_trm_roll_and_the_pad_rows_are_dropped():
    model = constant()
    out = run_gram(model, Sudoku(pool3()), N=2, D=3, batch=16)
    assert [b for b, _ in model.calls] == [8] * 6                  # 3 boards padded to 8 rows; 2 samples x 3 steps
    assert out["gram_lprm/model/raw"].shape == (3, 81) and out["gram_lprm/model/sample"].shape == (3,)
