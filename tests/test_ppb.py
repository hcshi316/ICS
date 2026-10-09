import copy

import numpy as np
import pytest

from fakes import assert_identical, digest, make_pool
from ics.data import load_pool
from ics.tasks import base, make_task, ppb
from ics.tasks.ppb import KINDS, PPB, SIDE
from ics.tasks.ppb_rules import (
    SHADE,
    WHITE,
    borders_to_rooms,
    heyawake_valid,
    nurikabe_valid,
    rooms_to_borders,
    tapa_valid,
)

DTYPES = (np.int64, np.int32, np.int16, np.uint8)
# (h, w, r0, c0) of synthetic boards: square and not, on the canvas edge, one whose dims run past the canvas (numpy
# crops it to 16 rows), and three too large for a turned canvas (no restatement)
PLACEMENTS = [(1, 1, 1, 1), (2, 3, 1, 1), (3, 2, 1, 1), (4, 4, 3, 5), (5, 7, 1, 1), (24, 24, 1, 1), (1, 24, 1, 1),
              (24, 2, 0, 24), (20, 4, 10, 3), (25, 3, 1, 1), (3, 25, 0, 1), (26, 2, 0, 0)]
CLUE_KEYS = {"nurikabe": [("nu", n) for n in (-1, 1, 2, 3)], "tapa": [("ta", s) for s in ("-", "1", "1,1", "2,-")]}
# per kind: every tag, restated row and decode mapped back of random_boards, of its rows and random rows of each dtype
# (a board too large for a turned canvas has no restatement; a Heyawake view or decode without a token is dropped).
# Nurikabe's and Tapa's random boards differ only in their clues' keys, so their values agree.
RESTATEMENTS = {"nurikabe": "1b55cf860239e88fbd8ab03664188539db825d628594128e6a4f90b9ea85ff37",
                "tapa": "1b55cf860239e88fbd8ab03664188539db825d628594128e6a4f90b9ea85ff37",
                "heyawake": "851d0bed76df4b7cea8a9d941b30aa453b4b5596119c2d2a95aa5ce0107ffbc8"}


def cropped(h, w, r0, c0):
    """The shape of the board at dims (h, w, r0, c0) as numpy crops it from the canvas."""
    return len(range(26)[r0:r0 + h]), len(range(26)[c0:c0 + w])


def random_boards(kind, copies=2):
    """A PPB task of `kind` on canvases of seeded random tokens, `copies` boards at each placement, the placements taken
    in turn, and its vocabulary size. Nurikabe and Tapa boards: undecided, shaded and unshaded cells, a fifth of them
    clues. Heyawake boards: random rooms (half with the borders the builder writes, half with raw border bits, which
    the rooms may not follow) and clues 0..3; the vocabulary holds each input cell's token but lacks a fifth of the
    other keys."""
    rng = np.random.default_rng(0)
    dims = [p for _ in range(copies) for p in PLACEMENTS]
    keys = CLUE_KEYS.get(kind)
    if kind == "heyawake":
        boards = []
        for d in dims:
            h, w = cropped(*d)
            rb, db = ((rng.random((h, w)) < 0.3).astype(int).tolist() for _ in range(2))
            if rng.random() < 0.5:
                rb, db = rooms_to_borders(borders_to_rooms(rb, db))
            clues = [[int(rng.integers(4)) if rng.random() < 0.15 else None for _ in range(w)] for _ in range(h)]
            boards.append([[("hy", rb[r][c], db[r][c], clues[r][c], 0) for c in range(w)] for r in range(h)])
        used = {key for board in boards for line in board for key in line}
        keys = [k for k in (("hy", rb, db, cl, s) for rb in (0, 1) for db in (0, 1) for cl in (None, 0, 1, 2, 3)
                            for s in (0, 1, 2)) if k in used or rng.random() < 0.8]
    tok = dict(zip(keys, (5 + rng.permutation(len(keys))).tolist()))
    X = rng.integers(0, 5 + len(keys), (len(dims), 26 * 26))
    for j, (x, (h, w, r0, c0)) in enumerate(zip(X, dims)):
        shape = cropped(h, w, r0, c0)
        if kind == "heyawake":
            board = [[tok[key] for key in line] for line in boards[j]]
        else:
            board = np.where(rng.random(shape) < 0.2, rng.choice(list(tok.values()), shape), rng.integers(2, 5, shape))
        x.reshape(26, 26)[r0:r0 + h, c0:c0 + w] = board
    pool = make_pool(X, X, dims=np.array(dims), vocab={str(k): v for k, v in tok.items()})
    return PPB(pool, kind), 5 + len(keys)


def random_rows(t, i, vocab_size, rng, n):
    """n rows of random tokens; on board i's cells, Heyawake's are random cell tokens of its vocabulary."""
    rows = rng.integers(0, vocab_size, (n, 26 * 26))
    if t.kind == "heyawake":
        h, w, r0, c0 = t.dims[i]
        for row in rows:
            board = row.reshape(26, 26)[r0:r0 + h, c0:c0 + w]
            board[...] = rng.choice(list(t.rev), board.shape)
    return rows


def grid(rows):
    return [list(r) for r in rows]


def assert_heyawake_restatement(t, i, row, tag, restated):
    """The restatement `restated` (tagged `tag`) of Heyawake row (board i) shows the board's image, found through an
    index grid: the rooms turned with their borders recomputed, and the clues and cell states turned."""
    info = t.info[i]
    src = np.rot90(np.arange(info["h"] * info["w"]).reshape(info["h"], info["w"]), tag[0])
    src = np.fliplr(src) if tag[1] else src         # src[r, c]: the board cell shown at cell (r, c) of the restatement
    rb, db = rooms_to_borders(np.array(info["rooms"]).reshape(-1)[src].tolist())
    clues = np.array(info["clues"], dtype=object).reshape(-1)[src]
    states = np.array([t.rev[int(v)][4] for v in t._crop(row, i).reshape(-1)])[src]
    hh, ww = src.shape
    shown = [[t.rev[int(v)] for v in line] for line in restated.reshape(26, 26)[1:hh + 1, 1:ww + 1]]
    assert shown == [[("hy", rb[r][c], db[r][c], clues[r, c], int(states[r, c])) for c in range(ww)] for r in range(hh)]


def test_nurikabe():
    clues = grid([[None] * 3, [None, 1, None], [None] * 3])
    state = grid([[SHADE] * 3, [SHADE, WHITE, SHADE], [SHADE] * 3])
    assert nurikabe_valid(clues, state) == "OK"
    state[0][0] = WHITE
    assert nurikabe_valid(clues, state) == "island-clues!=1"


def test_tapa():
    clues = grid([[None] * 3, [None, [8], None], [None] * 3])
    state = grid([[SHADE] * 3, [SHADE, WHITE, SHADE], [SHADE] * 3])
    assert tapa_valid(clues, state) == "OK"
    state[0][0] = WHITE
    assert tapa_valid(clues, state) == "ring-mismatch"


def test_heyawake():
    rooms = grid([[0, 0], [0, 0]])
    clues = grid([[1, None], [None, None]])
    assert heyawake_valid(rooms, clues, grid([[SHADE, WHITE], [WHITE, WHITE]])) == "OK"
    assert heyawake_valid(rooms, clues, grid([[SHADE, SHADE], [WHITE, WHITE]])) == "adjacent-shade"
    assert heyawake_valid(rooms, clues, grid([[WHITE] * 2] * 2)) == "room-count"


S, W = SHADE, WHITE
VERDICTS = [    # one grid per verdict: it passes every earlier rule of its checker and fails the named one
    (nurikabe_valid, ([[1, None]], [[W, S]]), "OK"),
    (nurikabe_valid, ([[1, None]], [[S, W]]), "clue-shaded"),
    (nurikabe_valid, ([[None, None, 2], [None, None, None]], [[S, S, W], [S, S, W]]), "2x2-sea"),
    (nurikabe_valid, ([[None, 1, None]], [[S, W, S]]), "sea-split"),
    (nurikabe_valid, ([[1, None, None]], [[W, S, W]]), "island-clues!=1"),
    (nurikabe_valid, ([[2, None]], [[W, S]]), "island-size"),
    (tapa_valid, ([[[1], None]], [[W, S]]), "OK"),
    (tapa_valid, ([[[1], None]], [[S, W]]), "clue-shaded"),
    (tapa_valid, ([[None, None, [4]], [None, None, None]], [[S, S, W], [S, S, W]]), "2x2"),
    (tapa_valid, ([[None, [1, 1], None]], [[S, W, S]]), "split"),
    (tapa_valid, ([[[2], None]], [[W, S]]), "ring-mismatch"),
    (heyawake_valid, ([[0, 0]], [[1, None]], [[S, W]]), "OK"),
    (heyawake_valid, ([[0, 0]], [[None, None]], [[S, S]]), "adjacent-shade"),
    (heyawake_valid, ([[0, 0, 0]], [[None] * 3], [[W, S, W]]), "white-split"),
    (heyawake_valid, ([[0, 0]], [[0, 0]], [[W, W]]), "two-clues-one-room"),
    (heyawake_valid, ([[0, 0]], [[1, None]], [[W, W]]), "room-count"),
    (heyawake_valid, ([[0, 1, 2]], [[None] * 3], [[W] * 3]), "3-room-line"),
    (heyawake_valid, ([[0], [1], [2]], [[None]] * 3, [[W]] * 3), "3-room-line"),     # the same rule down a column
]


@pytest.mark.parametrize("check, args, verdict", VERDICTS, ids=[f"{c.__name__}-{v}" for c, _, v in VERDICTS])
def test_each_rule_has_its_verdict(check, args, verdict):
    assert check(*args) == verdict


def test_rooms_borders_round_trip():
    rooms = [[0, 0, 1], [2, 2, 1]]                  # ids in scan order, as borders_to_rooms numbers them
    assert rooms_to_borders(rooms) == ([[0, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, 0]])
    assert borders_to_rooms(*rooms_to_borders(rooms)) == rooms


def test_alternatives_flip_the_parent_state():
    vocab = {"('nu', 1)": 5}
    x = np.zeros((26, 26), np.int64)
    x[:5, :5] = 1                                   # wall ring
    x[1:4, 1:4] = 2                                 # undecided board cells
    x[2, 2] = 5                                     # the clue
    t = PPB(make_pool([x.reshape(-1)], [x.reshape(-1)], dims=np.array([[3, 3, 1, 1]]), vocab=vocab), "nurikabe")
    answer = x.reshape(-1).copy()
    answer[1 * 26 + 1] = 3                          # board cell (0, 0) shaded
    assert t.alternatives(0, 1 * 26 + 1, current=3, ranked=[3, 4], answer=answer) == [4]
    assert t.alternatives(0, 1 * 26 + 2, current=4, ranked=[4, 3], answer=answer) == [3]
    assert t.alternatives(0, 1 * 26 + 1, current=4, ranked=[4, 3], answer=answer) == [4]   # the answer's state decides
    assert t.editable(0, x.reshape(-1)) == [27, 28, 29, 53, 55, 79, 80, 81]   # board cells except the clue (54)
    assert 27 not in t.editable(0, answer)                                        # a cell already set is not editable


def test_valid_raw_and_pinned():
    x = np.zeros((26, 26), np.int64)
    x[:5, :5] = 1                                   # wall ring
    x[1:4, 1:4] = 2                                 # undecided board cells
    x[2, 2] = 5                                     # the clue 2 at board cell (1, 1)
    y = x.copy()
    y[1:4, 1:4] = 3
    y[2, 2], y[2, 3] = 5, 4                         # solved: the island {(1, 1), (1, 2)}, the sea around it
    pool = make_pool([x.reshape(-1)], [y.reshape(-1)], dims=np.array([[3, 3, 1, 1]]), vocab={"('nu', 2)": 5})
    t = PPB(pool, "nurikabe")
    y = y.reshape(-1)
    assert t.valid(0, y) and t.consistent(0, y)
    junk = y.copy()
    junk[2 * 26 + 3] = 2                            # the island cell left undecided
    assert not t.valid(0, junk) and t.valid(0, t.pin(0, junk)) and t.consistent(0, junk)
    erased = y.copy()
    erased[2 * 26 + 2] = 4                          # the clue overwritten
    assert not t.valid(0, erased) and t.valid(0, t.pin(0, erased)) and not t.consistent(0, erased)
    outside = y.copy()
    outside[0] = outside[5 * 26 + 5] = 5            # junk on the wall ring and on the pad
    assert t.valid(0, outside)                      # only board cells are the answer
    np.testing.assert_array_equal(t.pin(0, outside), y)


def test_heyawake_restatements_are_images_of_the_structure():
    keys = [("hy", rb, db, cl, s) for rb in (0, 1) for db in (0, 1) for cl in (None, 1) for s in (0, 1, 2)]
    keys.remove(("hy", 0, 1, None, 1))              # no token for a shaded cell above a room border
    tok = {k: 5 + j for j, k in enumerate(keys)}
    x = np.zeros((26, 26), np.int64)
    x[:3, :5] = 1                                   # wall ring
    x[1, 1:4] = tok[("hy", 0, 0, 1, 0)], tok[("hy", 1, 0, None, 0)], tok[("hy", 0, 0, None, 0)]   # rooms [[0, 0, 1]]
    vocab = {str(k): v for k, v in tok.items()}
    t = PPB(make_pool([x.reshape(-1)], [x.reshape(-1)], dims=np.array([[1, 3, 1, 1]]), vocab=vocab), "heyawake")
    row = x.reshape(-1).copy()
    row[1 * 26 + 3] = tok[("hy", 0, 0, None, 1)]    # board cell (0, 2), room 1, shaded; clue 1 sits at (0, 0) in room 0
    restatements = t.restatements(0, row)
    # both k4 = 1 restatements put the shaded cell above a room border, which has no token, so they are dropped
    assert [tag for tag, _ in restatements] == [(0, False), (0, True), (2, False), (2, True), (3, False), (3, True)]
    for tag, restated in restatements:
        assert_heyawake_restatement(t, 0, row, tag, restated)
        np.testing.assert_array_equal(t._crop(t.restate_back(0, tag, restated), 0), t._crop(t.pin(0, row), 0))


def test_heyawake_pin_keeps_a_shaded_clue_0_cell():
    tok = {("hy", 1, 1, 0, 0): 5, ("hy", 1, 1, 0, 2): 6,          # the clue cell of a clue-0 room: no shaded token
           ("hy", 0, 0, None, 0): 7, ("hy", 0, 0, None, 1): 8, ("hy", 0, 0, None, 2): 9}
    x = np.zeros((26, 26), np.int64)
    x[:4, :4] = 1                                   # wall ring
    x[1:3, 1:3] = [[5, 7], [7, 7]]                  # 2x2 board: room {(0, 0)} with clue 0, and a room of 3 cells
    vocab = {str(k): v for k, v in tok.items()}
    t = PPB(make_pool([x.reshape(-1)], [x.reshape(-1)], dims=np.array([[2, 2, 1, 1]]), vocab=vocab), "heyawake")
    y = x.copy()
    y[1:3, 1:3] = [[8, 9], [9, 9]]                  # the clue-0 cell shaded through another cell's token, the rest not
    y = y.reshape(-1)
    p = t.pin(0, y)
    assert p[1 * 26 + 1] == 8                       # no token shades that cell: the answer's own token stays
    np.testing.assert_array_equal(t.pin(0, p), p)   # pin is idempotent
    info = t.info[0]
    assert heyawake_valid(info["rooms"], info["clues"], t._state(0, p)) == "room-count"   # the rules see the shading
    assert not t.valid(0, p) and not t.valid(0, t.pin(0, y))
    assert not t.consistent(0, y) and not t.consistent(0, p)     # the kept token has another cell's structure
    assert t.restate_back(0, (0, False), y) is None              # a decode that shades the cell does not map back
    white = y.copy()
    white[1 * 26 + 1] = 9                           # the clue-0 cell unshaded, through another cell's token
    assert not t.consistent(0, white) and t.consistent(0, t.pin(0, white)) and t.valid(0, t.pin(0, white))


def test_restatements_are_the_dihedral_images():
    b = np.array([[5, 2, 2], [2, 2, 3]])           # 2x3, no symmetry: a clue, undecided cells, one shaded cell
    x = np.zeros((26, 26), np.int64)
    x[:4, :5] = 1                                   # wall ring
    x[1:3, 1:4] = b
    t = PPB(make_pool([x.reshape(-1)], [x.reshape(-1)], dims=np.array([[2, 3, 1, 1]]), vocab={"('nu', 2)": 5}),
            "nurikabe")
    restatements = t.restatements(0, x.reshape(-1))
    assert [tag for tag, _ in restatements] == [(k4, mir) for k4 in range(4) for mir in (False, True)]
    for (k4, mir), row in restatements:
        v = np.rot90(b, k4)
        if mir:
            v = np.fliplr(v)
        h, w = v.shape
        canvas = row.reshape(26, 26)
        np.testing.assert_array_equal(canvas[1:h + 1, 1:w + 1], v)
        assert (canvas[0, :w + 2] == 1).all() and (canvas[h + 1, :w + 2] == 1).all()
        back = t.restate_back(0, (k4, mir), row)
        np.testing.assert_array_equal(t._crop(back, 0), t._crop(t.pin(0, x.reshape(-1)), 0))


@pytest.mark.parametrize("kind", KINDS)
def test_the_restatements_are_pinned(kind):
    t, vocab_size = random_boards(kind)
    rng, arrays = np.random.default_rng(1), []
    for i in range(len(t)):
        for j, row in enumerate([t.X[i], *random_rows(t, i, vocab_size, rng, 3)]):
            for tag, restated in t.restatements(i, row.astype(DTYPES[(i + j) % 4])):
                decode = rng.integers(0, vocab_size, 26 * 26).astype(DTYPES[(i + j + 1) % 4])
                back = t.restate_back(i, tag, decode)
                arrays += [np.array(tag), restated, *([] if back is None else [back])]
    assert digest(arrays) == RESTATEMENTS[kind]


def test_a_heyawake_token_without_a_state_raises():
    # a Heyawake cell's state comes from its token's key: KeyError for a token outside the vocabulary
    t, vocab_size = random_boards("heyawake", copies=1)
    for i in range(len(t)):
        h, w, r0, c0 = t.dims[i]
        for junk in (2, vocab_size, -1):
            row = t.X[i].copy()
            row.reshape(26, 26)[r0:r0 + h, c0:c0 + w][-1, -1] = junk
            with pytest.raises(KeyError):
                t.restatements(i, row)


@pytest.mark.parametrize("kind", KINDS)
def test_returned_arrays_are_the_callers(kind):
    # A caller may change every array a call returns; the later calls still return the first's.
    t, vocab_size = random_boards(kind, copies=1)
    decode = np.random.default_rng(1).integers(0, vocab_size, 26 * 26)
    for i in range(len(t)):
        want = copy.deepcopy(t.restatements(i, t.X[i]))
        want_back = copy.deepcopy([t.restate_back(i, tag, decode) for tag, _ in want])
        for tag, restated in t.restatements(i, t.X[i]):
            restated += 1
            back = t.restate_back(i, tag, decode)
            if back is not None:
                back[:] = 0
        assert_identical(t.restatements(i, t.X[i]), want)
        assert_identical([t.restate_back(i, tag, decode) for tag, _ in want], want_back)


@pytest.mark.parametrize("kind", KINDS)
def test_maps_are_kept_once_per_placement(kind, monkeypatch):
    built = []
    monkeypatch.setattr(ppb, "canvas_views", lambda side, dims: built.append(dims) or base.canvas_views(side, dims))
    t, _ = random_boards(kind)
    for _ in range(2):
        for i in range(len(t)):
            for tag, restated in t.restatements(i, t.X[i]):
                t.restate_back(i, tag, restated)
    fits = [d for d in PLACEMENTS if max(cropped(*d)) <= 26 - 2]
    assert sorted(built) == sorted(fits)                                  # at the first call per placement only
    for maps in t._maps.values():                                         # None: too large, no restatement
        if maps is not None:
            cells, gather, back = maps
            assert cells.dtype == gather.dtype == back.dtype == np.int16
            assert cells.nbytes + gather.nbytes + back.nbytes == 2 * (8 * 26 * 26 + 9 * len(cells))  # 10,816 + 18 n
    if kind == "heyawake":                                                # Heyawake's tokens, per board that fits
        assert sorted(t._tables) == [i for i, d in enumerate(t.dims.tolist()) if max(cropped(*d)) <= 26 - 2]
        for i, (views, own) in t._tables.items():
            n = t.info[i]["h"] * t.info[i]["w"]
            assert views.dtype == own.dtype == np.int16 and views.shape == (8, n, 3) and own.shape == (n, 3)


@pytest.mark.data
@pytest.mark.parametrize("kind", ["nurikabe", "tapa", "heyawake"])
def test_labels_valid_and_restatable(dataset, kind):
    pool = load_pool(dataset(kind))
    t = make_task(kind, pool)
    assert isinstance(t, PPB)
    idx = np.arange(len(pool))
    labels = np.where(pool.labels < 0, pool.inputs, pool.labels)
    assert t.check("raw", idx, labels).all()
    assert not t.check("raw", idx, pool.inputs).any()           # unsolved inputs are not answers
    counts = set()
    for i in range(len(pool)):
        restatements = t.restatements(i, labels[i])
        counts.add(len(restatements))
        for (k4, mir), row in restatements:
            back = t.restate_back(i, (k4, mir), row)
            assert back is not None
            np.testing.assert_array_equal(t._crop(back, i), t._crop(t.pin(i, labels[i]), i))
            if kind == "heyawake":                  # restate_back reads only states: check the structure shown
                assert_heyawake_restatement(t, i, labels[i], (k4, mir), row)
            else:                                   # k4 quarter-turns, then mir: a mirror; at (1, 1) of the canvas
                view = np.rot90(t._crop(labels[i], i), k4)
                view = np.fliplr(view) if mir else view
                h, w = view.shape
                np.testing.assert_array_equal(row.reshape(SIDE, SIDE)[1:h + 1, 1:w + 1], view)
    assert max(counts) == 8 and min(counts) >= 1
