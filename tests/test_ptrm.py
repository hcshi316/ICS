import numpy as np
import torch

from fakes import SOLUTION, ScriptedTRM, sudoku_pool
from ics.methods.ptrm import BlockNoise, run_ptrm
from ics.seeding import PTRM_NOISE, block_seed
from ics.tasks.sudoku import Sudoku

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]


def noisy_model():
    """Right iff the accumulated noise z_L > 0.5; q = z_L."""
    return ScriptedTRM(11, lambda row, t, zl: (SOLUTION if zl > 0.5 else WRONG, zl, 1.0))


def pool4():
    return sudoku_pool([(0,), (0, 1), (0, 2), (0, 3)])


def block_noise(sigma, seed):
    return BlockNoise(sigma, block_seed(seed, PTRM_NOISE, 1, 0), "cpu")


def test_block_noise_seeds_its_generator_with_the_first_word_of_its_seed_sequence():
    seeds = block_seed(7, PTRM_NOISE, 3, 512)
    assert BlockNoise(0.3, seeds, "cpu").gen.initial_seed() == int(seeds.generate_state(1, np.uint64)[0]) & (2**63 - 1)


def test_block_noise_skips_first_segment_and_is_seeded():
    z = torch.zeros(2, 3)
    a, b, c = block_noise(0.5, 7), block_noise(0.5, 7), block_noise(0.5, 8)
    assert torch.equal(a(1, z), z)                  # the first segment starts clean
    noisy = a(2, z)
    torch.testing.assert_close(noisy, b(2, z))      # the same seed gives the same noise
    assert not torch.equal(noisy, z) and not torch.equal(noisy, c(2, z))
    assert a(3, z.bfloat16()).dtype == torch.bfloat16


def test_selection_rules():
    task = Sudoku(pool4())
    out = run_ptrm(noisy_model(), task, K=32, D=4, sigma=1.0, batch=2, seed=0)
    for s in ("raw", "pinned"):
        acc = task.check(s, np.arange(4), out[f"ptrm/cert/{s}"])
        found = out[f"ptrm/cert/{s}/rollout"] >= 0
        np.testing.assert_array_equal(acc, found)
    # each noisy rollout ends with z_L ~ N(0, 3), right with p ~ 0.39, so 31 of them miss a board with p ~ 3e-7
    assert (out["ptrm/cert/raw/rollout"] > 0).all()                 # found by noise; the clean rollout 0 never is
    np.testing.assert_array_equal(out["ptrm/model/raw"], [SOLUTION] * 4)   # the largest final q is a right rollout
    np.testing.assert_array_equal(out["ptrm/model/raw"], out["ptrm/model/pinned"])
    assert set(out) == {"ptrm/model/raw", "ptrm/model/pinned", "ptrm/model/rollout",
                        "ptrm/cert/raw", "ptrm/cert/pinned", "ptrm/cert/raw/rollout", "ptrm/cert/pinned/rollout"}


def test_clean_rollout_only_without_noise():
    task = Sudoku(pool4())
    out = run_ptrm(noisy_model(), task, K=3, D=4, sigma=0.0, batch=4, seed=0)
    np.testing.assert_array_equal(out["ptrm/cert/raw/rollout"], [-1] * 4)
    np.testing.assert_array_equal(out["ptrm/cert/raw"], [WRONG] * 4)
    np.testing.assert_array_equal(out["ptrm/model/rollout"], [0] * 4)


def test_accepts_the_smallest_settings():
    out = run_ptrm(noisy_model(), Sudoku(pool4()), K=1, D=1, sigma=1.0, batch=1, seed=0)
    np.testing.assert_array_equal(out["ptrm/model/raw"], [WRONG] * 4)      # the clean rollout's first segment


def test_block_noise_adds_sigma_times_its_generators_normals():
    # 4 x 9 = 36 values. A CPU bfloat16 draw differs from a float32 draw rounded to bfloat16 only at 16 or more values
    # that are not a multiple of 16
    z = torch.ones(4, 9)
    noise = block_noise(0.5, 7)
    gen = torch.Generator().set_state(noise.gen.get_state())
    eps2, eps3 = torch.randn(4, 9, generator=gen), torch.randn(4, 9, generator=gen)    # float32, one draw per segment
    torch.testing.assert_close(noise(2, z), z + 0.5 * eps2)
    torch.testing.assert_close(noise(3, z.bfloat16()), z.bfloat16() + (0.5 * eps3).bfloat16())


def test_certificate_takes_the_first_accepted_rollout_per_scoring():
    broken_given = SOLUTION.copy()
    broken_given[1] = SOLUTION[5]       # raw-invalid; pinned-valid where cell 1 is a given (every board but board 1)
    # z_L stays 0 only in the clean rollout 0, which decodes WRONG; the noisy rollouts decode broken_given
    model = ScriptedTRM(11, lambda row, t, zl: (WRONG if zl == 0 else broken_given, 0.0, 1.0))
    out = run_ptrm(model, Sudoku(pool4()), K=3, D=2, sigma=1.0, batch=4, seed=0)
    np.testing.assert_array_equal(out["ptrm/cert/raw/rollout"], [-1] * 4)
    np.testing.assert_array_equal(out["ptrm/cert/raw"], [WRONG] * 4)        # none accepted: rollout 0's decode
    np.testing.assert_array_equal(out["ptrm/cert/pinned/rollout"], [1, -1, 1, 1])
    np.testing.assert_array_equal(out["ptrm/cert/pinned"], [broken_given, WRONG, broken_given, broken_given])
    assert out["ptrm/cert/pinned"].dtype == out["ptrm/model/pinned"].dtype == np.int16


def test_each_rollout_chunk_and_seed_draws_its_own_noise():
    task = Sudoku(sudoku_pool([(0,)] * 8))                  # 8 identical boards, one chunk each
    picks = [run_ptrm(noisy_model(), task, K=8, D=2, sigma=1.0, batch=1, seed=s)["ptrm/model/rollout"] for s in (0, 1)]
    # a pick is the argmax of 8 final q's; with independent noise, each assertion below fails with p < 2e-6
    assert picks[0].max() > 1                               # the rollouts do not share one noise stream (ties go first)
    assert len(set(picks[0])) > 1                           # nor do the chunks
    assert not np.array_equal(picks[0], picks[1])           # the seed changes the noise
