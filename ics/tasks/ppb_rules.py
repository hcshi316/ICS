"""Rule checkers of the PPBench shading puzzles. A state grid holds SHADE or WHITE per cell; every checker returns "OK"
or the name of the first rule that fails."""
from __future__ import annotations

from collections import Counter, deque

EMPTY, SHADE, WHITE = 0, 1, 2


def _components(mask):
    """Components (sets of (r, c)) of the True cells, 4-connectivity."""
    h, w = len(mask), len(mask[0])
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r][c] and not seen[r][c]:
                comp, dq = set(), deque([(r, c)])
                seen[r][c] = True
                while dq:
                    y, x = dq.popleft()
                    comp.add((y, x))
                    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            dq.append((ny, nx))
                out.append(comp)
    return out


def _no22(shaded):
    h, w = len(shaded), len(shaded[0])
    for r in range(h - 1):
        for c in range(w - 1):
            if shaded[r][c] and shaded[r][c + 1] and shaded[r + 1][c] and shaded[r + 1][c + 1]:
                return False
    return True


def nurikabe_valid(clues, state):
    """clues[r][c] = island size (-1 = any size) or None."""
    h, w = len(clues), len(clues[0])
    shaded = [[state[r][c] == SHADE for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            if clues[r][c] is not None and shaded[r][c]:
                return "clue-shaded"
    if not _no22(shaded):
        return "2x2-sea"
    if len(_components(shaded)) > 1:
        return "sea-split"
    for comp in _components([[not shaded[r][c] for c in range(w)] for r in range(h)]):
        cl = [clues[r][c] for (r, c) in comp if clues[r][c] is not None]
        if len(cl) != 1:
            return "island-clues!=1"
        if cl[0] != -1 and cl[0] != len(comp):
            return "island-size"
    return "OK"


def tapa_valid(clues, state):
    """clues[r][c] = list of run lengths of shaded cells around the clue (-1 = any length) or None."""
    h, w = len(clues), len(clues[0])
    shaded = [[state[r][c] == SHADE for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            if clues[r][c] is not None and shaded[r][c]:
                return "clue-shaded"
    if not _no22(shaded):
        return "2x2"
    if len(_components(shaded)) > 1:
        return "split"
    ring = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    for r in range(h):
        for c in range(w):
            cl = clues[r][c]
            if cl is None:
                continue
            cells = [shaded[r + dy][c + dx] if 0 <= r + dy < h and 0 <= c + dx < w else False for dy, dx in ring]
            if all(cells):
                runs = [8]
            else:
                start = cells.index(False)
                rot = cells[start:] + cells[:start]
                runs, cur = [], 0
                for v in rot:
                    if v:
                        cur += 1
                    elif cur:
                        runs.append(cur)
                        cur = 0
                if cur:
                    runs.append(cur)
            exact = sorted(x for x in cl if x > 0)
            n_wild = sum(1 for x in cl if x == -1)
            if not exact and not n_wild:
                if runs:
                    return "ring-mismatch"
                continue
            rest = sorted(runs)
            for v in exact:
                if v in rest:
                    rest.remove(v)
                else:
                    return "ring-mismatch"
            if len(rest) != n_wild:
                return "ring-mismatch"
    return "OK"


def heyawake_valid(rooms, clues, state):
    """rooms[r][c] = room id; clues[r][c] = number of shaded cells in the room (at one cell of it) or None."""
    h, w = len(rooms), len(rooms[0])
    shaded = [[state[r][c] == SHADE for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            if shaded[r][c]:
                for dy, dx in ((0, 1), (1, 0)):
                    ny, nx = r + dy, c + dx
                    if 0 <= ny < h and 0 <= nx < w and shaded[ny][nx]:
                        return "adjacent-shade"
    if len(_components([[not shaded[r][c] for c in range(w)] for r in range(h)])) > 1:
        return "white-split"
    room_clue, room_count = {}, Counter()
    for r in range(h):
        for c in range(w):
            if clues[r][c] is not None:
                rid = rooms[r][c]
                if rid in room_clue:
                    return "two-clues-one-room"
                room_clue[rid] = clues[r][c]
            if shaded[r][c]:
                room_count[rooms[r][c]] += 1
    for rid, cl in room_clue.items():
        if room_count[rid] != cl:
            return "room-count"
    for r in range(h):
        c = 0
        while c < w:
            if shaded[r][c]:
                c += 1
                continue
            c0 = c
            while c < w and not shaded[r][c]:
                c += 1
            if 1 + sum(1 for x in range(c0 + 1, c) if rooms[r][x] != rooms[r][x - 1]) >= 3:
                return "3-room-line"
    for c in range(w):
        r = 0
        while r < h:
            if shaded[r][c]:
                r += 1
                continue
            r0 = r
            while r < h and not shaded[r][c]:
                r += 1
            if 1 + sum(1 for y in range(r0 + 1, r) if rooms[y][c] != rooms[y - 1][c]) >= 3:
                return "3-room-line"
    return "OK"


def rooms_to_borders(rooms):
    h, w = len(rooms), len(rooms[0])
    rb = [[1 if c + 1 < w and rooms[r][c] != rooms[r][c + 1] else 0 for c in range(w)] for r in range(h)]
    db = [[1 if r + 1 < h and rooms[r][c] != rooms[r + 1][c] else 0 for c in range(w)] for r in range(h)]
    return rb, db


def borders_to_rooms(rb, db):
    """Flood fill with walls given by the right/down border bits; returns a room-id grid."""
    h, w = len(rb), len(rb[0])
    out = [[-1] * w for _ in range(h)]
    nid = 0
    for r in range(h):
        for c in range(w):
            if out[r][c] != -1:
                continue
            dq = deque([(r, c)])
            out[r][c] = nid
            while dq:
                y, x = dq.popleft()
                if x + 1 < w and not rb[y][x] and out[y][x + 1] == -1:
                    out[y][x + 1] = nid
                    dq.append((y, x + 1))
                if x - 1 >= 0 and not rb[y][x - 1] and out[y][x - 1] == -1:
                    out[y][x - 1] = nid
                    dq.append((y, x - 1))
                if y + 1 < h and not db[y][x] and out[y + 1][x] == -1:
                    out[y + 1][x] = nid
                    dq.append((y + 1, x))
                if y - 1 >= 0 and not db[y - 1][x] and out[y - 1][x] == -1:
                    out[y - 1][x] = nid
                    dq.append((y - 1, x))
            nid += 1
    return out
