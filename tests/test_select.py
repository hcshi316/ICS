import numpy as np

from fakes import SOLUTION, sudoku_pool
from ics.select import best, first_valid, join, majority, pick, take
from ics.tasks.sudoku import Sudoku

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]                  # a repeated digit in cell 0, which is blank on every board: invalid either way
BROKEN = SOLUTION.copy()
BROKEN[1] = SOLUTION[5]                 # raw-invalid; pinned-valid where cell 1 is a given (boards 0 and 2)


def task3():
    return Sudoku(sudoku_pool([(0,), (0, 1), (0, 2)]))


# candidate k (rows) of boards 0, 1, 2 (columns)
CANDS = np.stack([np.stack([WRONG, WRONG, BROKEN]),
                  np.stack([BROKEN, BROKEN, SOLUTION]),
                  np.stack([SOLUTION, WRONG, SOLUTION])]).astype(np.int16)


def test_first_valid_is_per_scoring_and_minus_one_when_none_is():
    idx = np.arange(3)
    np.testing.assert_array_equal(first_valid(task3(), idx, CANDS, "raw"), [2, -1, 1])
    np.testing.assert_array_equal(first_valid(task3(), idx, CANDS, "pinned"), [1, -1, 0])


def test_first_valid_checks_each_candidate_against_the_board_idx_names():
    cands = CANDS[:, 1:]                                          # the candidates of boards 1 and 2
    np.testing.assert_array_equal(first_valid(task3(), np.array([1, 2]), cands, "pinned"), [-1, 0])
    np.testing.assert_array_equal(first_valid(task3(), np.arange(2), cands, "pinned"), [1, 1])   # boards 0 and 1


def test_best_takes_the_first_of_equal_scores():
    scores = np.array([[0.5, 2.0, -1.0], [0.7, 2.0, -1.0], [0.7, 1.0, -1.0]], np.float32)     # [K=3, n=3]
    np.testing.assert_array_equal(best(scores), [1, 0, 0])


def test_majority_takes_the_most_frequent_answer_at_its_first_occurrence():
    A, B, C = (np.full(4, v, np.int16) for v in (1, 2, 3))
    boards = [[A, B, B, A, C],          # A and B twice: A occurs first -> sample 0
              [B, A, A, B, C],          # B and A twice: B occurs first -> sample 0
              [C, A, A, B, B],          # A and B twice: A occurs first, at sample 1
              [C, B, A, A, A]]          # A three times: its first occurrence, sample 2
    cands = np.stack([np.stack(b) for b in boards], axis=1)       # [K=5, n=4, 4]
    np.testing.assert_array_equal(majority(cands), [0, 0, 1, 2])


def test_take_falls_back_to_candidate_zero():
    cands = np.arange(6).reshape(2, 3, 1)
    np.testing.assert_array_equal(take(cands, np.array([1, -1, 0])), [[3], [1], [2]])


def test_pick_writes_every_row_both_regimes_and_both_scorings():
    out = pick(task3(), np.arange(3), CANDS, {"a": np.array([2, 0, 1]), "b": np.zeros(3, np.int64)}, "sample")
    parts = ("model/raw", "model/pinned", "model/sample", "cert/raw", "cert/pinned", "cert/raw/sample",
             "cert/pinned/sample")
    assert set(out) == {f"{r}/{p}" for r in "ab" for p in parts}
    np.testing.assert_array_equal(out["a/model/raw"], take(CANDS, np.array([2, 0, 1])))
    np.testing.assert_array_equal(out["a/model/pinned"], out["a/model/raw"])       # one commit, scored both ways
    np.testing.assert_array_equal(out["a/model/sample"], [2, 0, 1])
    for row in "ab":                                                                  # every row shares the certificate
        np.testing.assert_array_equal(out[f"{row}/cert/raw/sample"], [2, -1, 1])
        np.testing.assert_array_equal(out[f"{row}/cert/pinned"], [BROKEN, WRONG, BROKEN])
    assert out["a/cert/raw"].dtype == np.int16 and out["a/model/sample"].dtype == np.int64


def test_join_concatenates_blocks_along_the_boards():
    parts = [{"x/model/raw": np.zeros((2, 3), np.int16)}, {"x/model/raw": np.ones((1, 3), np.int16)}]
    np.testing.assert_array_equal(join(parts)["x/model/raw"], [[0, 0, 0], [0, 0, 0], [1, 1, 1]])
