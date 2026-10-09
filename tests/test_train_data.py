import numpy as np
import pytest

from fakes import write_trm_split
from ics.data import IGNORE, load_train, train_batches


def official_block(starts, seed, epochs, batch, block):
    """A block as TinyRecursiveModels' PuzzleDataset draws it: every permutation, then one scalar pick per group in
    order; the incomplete last batch is dropped."""
    rng = np.random.Generator(np.random.Philox(seed + block))
    n = len(starts) - 1
    order = np.concatenate([rng.permutation(n) for _ in range(epochs)])
    picks = np.array([rng.integers(starts[g], starts[g + 1]) for g in order])
    k = len(order) // batch
    return picks[:k * batch].reshape(k, batch)


@pytest.mark.parametrize("sizes", [[3, 1, 4, 2, 5], [1] * 7])
def test_the_stream_is_the_official_order(sizes):
    starts = np.concatenate([[0], np.cumsum(sizes)])
    stream = train_batches(starts, seed=11, epochs_per_block=3, global_batch=4)
    for block in (1, 2, 3):
        for index, rows in enumerate(official_block(starts, 11, 3, 4, block)):
            b, i, got = next(stream)
            assert (b, i) == (block, index)
            np.testing.assert_array_equal(got, rows)


def test_a_block_draws_every_group_epochs_per_block_times_but_the_dropped_tail():
    # 10 groups x 5 epochs = 50 draws, 12 batches
    stream = train_batches(np.arange(11), seed=0, epochs_per_block=5, global_batch=4)
    counts = np.bincount(np.concatenate([next(stream)[2] for _ in range(12)]), minlength=10)
    assert counts.sum() == 48 and set(counts) <= {4, 5}
    assert next(stream)[:2] == (2, 0)


def test_a_stream_started_at_a_position_continues_the_stream():
    starts = np.concatenate([[0], np.cumsum([2, 3, 1, 2])])         # 4 groups, 3 epochs, batch 2: 6 per block
    full = train_batches(starts, 5, 3, 2)
    batches = [next(full) for _ in range(20)]
    for k in (0, 4, 5, 6, 13):
        resumed = train_batches(starts, 5, 3, 2, start=batches[k][:2])
        for block, index, rows in batches[k:]:
            got = next(resumed)
            assert got[:2] == (block, index) and np.array_equal(got[2], rows)
    assert next(train_batches(starts, 5, 3, 2, start=(1, 6)))[:2] == (2, 0)


def test_the_seed_changes_the_order():
    a, b, c = (train_batches(np.arange(101), seed, 2, 8) for seed in (0, 0, 1))
    first = [next(a)[2] for _ in range(5)]
    assert all(np.array_equal(x, next(b)[2]) for x in first)
    assert not all(np.array_equal(x, next(c)[2]) for x in first)


def test_a_block_without_a_full_batch_is_refused():
    with pytest.raises(ValueError, match="holds no batch"):
        next(train_batches(np.arange(4), 0, 1, 4))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_load_train_reads_either_index_dtype_and_collates_like_the_official_loader(tmp_path, dtype):
    X = np.arange(30).reshape(6, 5) % 7 + 1
    Y = X.copy()
    Y[0, 1] = 0                                                      # ignore_label_id -> IGNORE
    write_trm_split(tmp_path, "train", X, Y, group_sizes=[2, 1, 3], index_dtype=dtype)
    split = load_train(tmp_path)
    assert split.group_starts.dtype == np.int64
    np.testing.assert_array_equal(split.group_starts, [0, 2, 3, 6])
    batch = split.rows(np.array([3, 0]))
    assert all(v.dtype == np.int32 for v in batch.values())
    np.testing.assert_array_equal(batch["inputs"], X[[3, 0]])
    np.testing.assert_array_equal(batch["labels"], np.where(Y == 0, IGNORE, Y)[[3, 0]])
    np.testing.assert_array_equal(batch["puzzle_identifiers"], [0, 0])


def test_load_train_refuses_puzzles_of_several_examples(tmp_path):
    write_trm_split(tmp_path, "train", np.ones((4, 3)), np.ones((4, 3)))
    np.save(tmp_path / "train" / "all__puzzle_indices.npy", np.array([0, 2, 3, 4], np.int32))
    with pytest.raises(ValueError, match="exactly one example"):
        load_train(tmp_path)
