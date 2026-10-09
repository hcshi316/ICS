"""The block loop of the sampling methods (ics/select.py, blocks), for PTRM, GRAM, EqR and the Attractor alike: a block
of `batch` boards draws from generators seeded by its first row in the test split, so shards of whole blocks reproduce
the full run; a setting below 1 is refused, and an empty task gives empty answers under every key."""
import numpy as np
import pytest
import torch

from fakes import SOLUTION, ScriptedAttractor, ScriptedRestarts, ScriptedSampler, ScriptedTRM, sudoku_pool
from ics.methods.ptrm import run_ptrm
from ics.seeding import ATTRACTOR_RESTARTS, EQR_RESTARTS, GRAM_NOISE, PTRM_NOISE, block_seed, torch_generator
from ics.tasks.sudoku import Sudoku
from ics_baselines.attractor.predict import run_attractor
from ics_baselines.eqr.predict import run_eqr
from ics_baselines.gram.predict import run_gram

WRONG = SOLUTION.copy()
WRONG[0] = SOLUTION[1]


def answer(u: float) -> np.ndarray:
    return SOLUTION if u > 0.5 else WRONG


class Restarted(ScriptedRestarts):
    """EqR's interface: every row draws u ~ U(0, 1) from its block's generator for its initial state and decodes
    answer(u), with q_halt u."""

    def __init__(self):
        super().__init__(11, 1, None)

    def initial_state(self, batch_size, gen):
        u = torch.rand(batch_size, 1, generator=gen)
        return u, u.clone()

    def step(self, z_H, z_L, inputs, gen):
        answers = np.stack([answer(float(u)) for u in z_H[:, 0]])
        return z_H, z_L, torch.nn.functional.one_hot(torch.as_tensor(answers), 11).float(), z_H[:, 0].clone()


class Perturbed(ScriptedAttractor):
    """The Attractor's interface: a restart decodes answer(0.5 + dH[0]), restart 0 (unperturbed) WRONG, with q_halt
    dH[0]."""

    def __init__(self):
        super().__init__(11, 3, None)

    def segment(self, z_H, z_L, inputs, puzzle_identifiers):
        dH = self.starts[-1][1]
        u = 0.0 if dH is None else float(dH[0])
        answers = np.stack([answer(0.5 + u)] * inputs.shape[0])
        logits = torch.nn.functional.one_hot(torch.as_tensor(answers), 11).float()
        return z_H, z_L, logits, torch.full((len(answers),), u)


def sampled():
    """GRAM's interface: a row of a sample decodes answer(u), with LPRM logit u, u drawn from the sample's generator."""
    def script(row, i, gen):
        u = float(torch.rand(1, generator=gen))
        return answer(u), u
    return ScriptedSampler(11, script)


def noised():
    """TRM's interface under PTRM: a row decodes answer(0.5 + z_L), z_L the noise added so far."""
    return ScriptedTRM(11, lambda row, t, zl: (answer(0.5 + zl), zl, 1.0))


# name -> (the protocol, with a model whose answers follow its block's draws; the module that seeds its generators;
# its purpose id; the seeding keys of a block before its first row, at the default counts)
PROTOCOLS = {
    "ptrm": (lambda task, **kw: run_ptrm(noised(), task, **{"K": 3, "D": 2, **kw}), "ics.methods.ptrm", PTRM_NOISE,
             [(1,), (2,)]),
    "gram": (lambda task, **kw: run_gram(sampled(), task, **{"N": 3, "D": 1, **kw}), "ics_baselines.gram.predict",
             GRAM_NOISE, [(0,), (1,), (2,)]),
    "eqr": (lambda task, **kw: run_eqr(Restarted(), task, **{"R": 3, "steps": 1, **kw}), "ics_baselines.eqr.predict",
            EQR_RESTARTS, [()]),
    "attractor": (lambda task, **kw: run_attractor(Perturbed(), task, **{"R": 6, "segments": 1, **kw}),
                  "ics_baselines.attractor.predict", ATTRACTOR_RESTARTS, [()]),
}
SETTINGS = {"ptrm": ("K", "D", "batch"), "gram": ("N", "D", "batch"), "eqr": ("R", "steps", "batch"),
            "attractor": ("R", "segments", "batch")}


@pytest.mark.parametrize("name", PROTOCOLS)
def test_each_block_seeds_its_generators_from_its_first_row(name, monkeypatch):
    run, module, purpose, keys = PROTOCOLS[name]
    seen = []
    monkeypatch.setattr(f"{module}.torch_generator",
                        lambda seeds, device: seen.append(seeds.entropy) or torch_generator(seeds, device))
    pool = sudoku_pool([(0,)] * 6).take(slice(2, 6))            # rows 2..5 of the split: blocks start at rows 2 and 4
    run(Sudoku(pool), batch=2, seed=5)
    assert seen == [block_seed(5, purpose, *key, row).entropy for row in (2, 4) for key in keys]


@pytest.mark.parametrize("n", [4, 5], ids=["whole_blocks", "short_last_block"])
@pytest.mark.parametrize("name", PROTOCOLS)
def test_whole_block_shards_reproduce_the_full_run(name, n):
    run = PROTOCOLS[name][0]
    pool = sudoku_pool([(0,), (0, 1), (0, 2), (0, 3), (0, 4)][:n])
    whole = run(Sudoku(pool), batch=2, seed=3)
    shards = [run(Sudoku(pool.take(slice(a, b))), batch=2, seed=3) for a, b in ((0, 2), (2, n))]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([s[key] for s in shards]))


@pytest.mark.parametrize("name, setting", [(n, s) for n, settings in SETTINGS.items() for s in settings])
def test_a_setting_below_one_is_refused(name, setting):
    with pytest.raises(ValueError, match=rf"^run_{name} needs .*\b{setting}=0\b"):
        PROTOCOLS[name][0](Sudoku(sudoku_pool([(0,)])), **{setting: 0})


@pytest.mark.parametrize("name", PROTOCOLS)
def test_an_empty_task_gives_empty_answers_under_every_key(name):
    run, pool = PROTOCOLS[name][0], sudoku_pool([(0,), (0, 1)])
    full, empty = run(Sudoku(pool)), run(Sudoku(pool.take(slice(0, 0))))
    assert empty.keys() == full.keys()
    for key, a in full.items():
        assert empty[key].shape == (0, *a.shape[1:]) and empty[key].dtype == a.dtype, key
