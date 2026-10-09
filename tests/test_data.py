import json

import numpy as np
import pytest

from fakes import digest, write_trm_split
from ics.data import IGNORE, dataset_dims, decode_labels, encode_labels, label_key, load_pool, load_train


def write_split(root, split, X, Y, dims=None, golden=None, vocab=None):
    d = root / split
    d.mkdir(parents=True)
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 11, "seq_len": X.shape[1],
            "num_puzzle_identifiers": 1, "total_groups": len(X), "mean_puzzle_examples": 1.0,
            "total_puzzles": len(X), "sets": ["all"]}
    (d / "dataset.json").write_text(json.dumps(meta))
    np.save(d / "all__inputs.npy", X.astype(np.uint8))
    np.save(d / "all__labels.npy", Y.astype(np.uint8))
    if dims is not None:
        np.save(d / "all__dims.npy", dims)
    if golden is not None:
        np.save(d / "all__is_golden.npy", golden)
    if vocab is not None:
        (root / "vocab.json").write_text(json.dumps(vocab))


def test_load_pool_maps_ignore_and_slices(tmp_path):
    X = np.arange(12).reshape(4, 3) % 5 + 1
    Y = X.copy()
    Y[1, 2] = 0                                   # ignore_label_id -> IGNORE
    write_split(tmp_path, "test", X, Y, dims=np.ones((4, 4), int), golden=np.array([0, 0, 1, 1]),
                vocab={"('nu', 3)": 5})
    pool = load_pool(tmp_path, "test", start=1, limit=2)
    assert len(pool) == 2
    assert pool.inputs.dtype == np.int64 and pool.labels.dtype == np.int64
    assert type(pool.inputs) is np.ndarray and type(pool.labels) is np.ndarray
    np.testing.assert_array_equal(pool.inputs, X[1:3])
    assert pool.labels[0, 2] == IGNORE
    np.testing.assert_array_equal(pool.index, [1, 2])
    np.testing.assert_array_equal(pool.is_golden, [0, 1])
    assert pool.dims.shape == (2, 4)
    assert pool.vocab == {"('nu', 3)": 5}
    assert pool.meta["seq_len"] == 3


def test_window_keeps_global_index(tmp_path):
    X = np.ones((5, 2), int)
    write_split(tmp_path, "test", X, X)
    pool = load_pool(tmp_path)
    w = pool.take(slice(3, 10))
    assert len(w) == 2
    np.testing.assert_array_equal(w.index, [3, 4])
    assert w.dims is None and w.is_golden is None


def test_window_slices_board_metadata(tmp_path):
    X = np.ones((4, 2), int)
    dims = np.arange(16).reshape(4, 4)
    write_split(tmp_path, "test", X, X, dims=dims, golden=np.array([0, 1, 0, 1]), vocab={"('ta', '1')": 5})
    pool = load_pool(tmp_path)
    w = pool.take(slice(1, 3))
    np.testing.assert_array_equal(w.dims, dims[1:3])
    np.testing.assert_array_equal(w.is_golden, [1, 0])
    assert w.vocab == pool.vocab and w.meta is pool.meta


def test_rejects_negative_ranges(tmp_path):
    X = np.ones((2, 2), int)
    write_split(tmp_path, "test", X, X)
    with pytest.raises(ValueError):
        load_pool(tmp_path, start=-1)
    with pytest.raises(ValueError):
        load_pool(tmp_path, limit=-1)


def test_dataset_dims_reads_a_splits_metadata_by_default_the_test_splits(tmp_path):
    for split, seq_len in (("test", 81), ("train", 900)):
        (tmp_path / split).mkdir()
        (tmp_path / split / "dataset.json").write_text(json.dumps(
            {"seq_len": seq_len, "vocab_size": 11, "num_puzzle_identifiers": 1, "pad_id": 0}))
    assert dataset_dims(tmp_path) == {"seq_len": 81, "vocab_size": 11, "num_puzzle_identifiers": 1}
    assert dataset_dims(tmp_path, "train") == {"seq_len": 900, "vocab_size": 11, "num_puzzle_identifiers": 1}


def test_take_picks_boards_by_index(tmp_path):
    X = np.arange(8).reshape(4, 2) + 1
    write_split(tmp_path, "test", X, X, dims=np.arange(16).reshape(4, 4), golden=np.array([0, 1, 0, 1]))
    sub = load_pool(tmp_path).take(np.array([3, 1]))
    np.testing.assert_array_equal(sub.inputs, X[[3, 1]])
    np.testing.assert_array_equal(sub.index, [3, 1])
    np.testing.assert_array_equal(sub.is_golden, [1, 1])
    np.testing.assert_array_equal(sub.dims, np.arange(16).reshape(4, 4)[[3, 1]])


def test_load_pool_reads_the_rows_it_is_given_as_take_picks_them(tmp_path):
    X = np.arange(12).reshape(4, 3) % 5 + 1
    Y = X.copy()
    Y[3, 0] = 0                                   # ignore_label_id -> IGNORE
    write_split(tmp_path, "test", X, Y, dims=np.arange(16).reshape(4, 4), golden=np.array([0, 1, 0, 1]),
                vocab={"('nu', 3)": 5})
    pool, taken = load_pool(tmp_path, rows=np.array([3, 0])), load_pool(tmp_path).take(np.array([3, 0]))
    for name in ("inputs", "labels", "index", "dims", "is_golden"):
        np.testing.assert_array_equal(getattr(pool, name), getattr(taken, name))
    assert pool.labels[0, 0] == IGNORE and pool.inputs.dtype == np.int64
    assert pool.vocab == taken.vocab and pool.meta == taken.meta


@pytest.mark.parametrize("dtype, vocab", [(np.uint8, 9), (np.int32, 1000)])
def test_encoded_labels_decode_to_the_labels_in_their_dtype(dtype, vocab):
    labels = np.random.default_rng(0).integers(0, vocab, (8, 676)).astype(dtype)
    stored = encode_labels(labels, vocab)
    assert stored.dtype == dtype and stored.min() >= 0 and stored.max() < vocab
    assert (stored != labels).mean() > 0.5                          # most cells hold another token
    decoded = decode_labels(stored, vocab)
    assert decoded.dtype == dtype and np.array_equal(decoded, labels)


def test_the_label_key_is_fixed():
    # the key encoded datasets were written with, an offset per position whatever the length: a change would decode
    # them wrongly
    assert label_key(8).tolist() == [62501, 33003, 12172, 5192, 32511, 50057, 43723, 7813]
    assert np.array_equal(label_key(81), label_key(676)[:81])
    assert digest([label_key(676)]) == "2b4f0d3633f5f0ccceecdcc0a7722a5fea9f7dd1dbe5bdfe8f80f159b1e87f99"


@pytest.mark.parametrize("encoded", [False, True], ids=["plain", "encoded"])
def test_the_loaders_give_a_splits_labels_stored_plain_or_encoded(tmp_path, encoded):
    X = np.arange(12).reshape(4, 3) % 5 + 1
    Y = X.copy()
    Y[1, 2] = 0                                   # ignore_label_id -> IGNORE
    write_trm_split(tmp_path, "train", X, Y, encoded=encoded)        # encoded: dataset.json declares it
    assert np.array_equal(np.load(tmp_path / "train" / "all__labels.npy"), Y) != encoded
    want = np.where(Y == 0, IGNORE, Y)
    np.testing.assert_array_equal(load_pool(tmp_path, "train").labels, want)
    np.testing.assert_array_equal(load_train(tmp_path).rows(np.arange(4))["labels"], want)
