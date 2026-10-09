import numpy as np
import torch

from ics.seeding import (
    ATTRACTOR_RESTARTS,
    EQR_RESTARTS,
    GRAM_NOISE,
    PTRM_NOISE,
    SUDOKU_RESTATEMENTS,
    block_seed,
    torch_generator,
)


def test_purpose_ids_are_fixed_and_distinct():
    assert [SUDOKU_RESTATEMENTS, PTRM_NOISE, GRAM_NOISE, EQR_RESTARTS, ATTRACTOR_RESTARTS] == [1, 2, 3, 4, 5]


def test_torch_generator_is_seeded_with_the_first_word_of_its_seed_sequence():
    seeds = block_seed(7, GRAM_NOISE, 3, 512)
    gen = torch_generator(seeds, "cpu")
    assert gen.initial_seed() == int(seeds.generate_state(1, np.uint64)[0]) & (2 ** 63 - 1)
    assert torch.equal(torch.randn(4, generator=gen), torch.randn(4, generator=torch_generator(seeds, "cpu")))
