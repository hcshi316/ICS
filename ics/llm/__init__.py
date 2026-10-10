# The rules in the prompts (<task>.txt) are puzz.link's, from pzpr.js (github.com/robx/pzprjs; MIT License:
# LICENSES/MIT-pzprjs.txt), modified; the boards of golden_boards.json are Pencil Puzzle Bench's (MIT License:
# LICENSES/MIT-PPBench.txt), and heyawake.txt's first rule line is quoted from its website, ppbench.com.
"""The LLM baselines: a language model solves the golden boards of Pencil Puzzle Bench, 15 per task, from a text
prompt (python -m ics llm).

golden_boards.json holds each board's givens, no solution: its size h x w and, for Light-Up, its "cells" (. an empty
cell, # a black cell, 0-4 a black cell with that clue); for Nurikabe and Tapa, its "clues"; for Heyawake, its "rooms"
(a room id per cell) and the "clues" of the rooms, each at one cell of its room. A clue is a number (Tapa: a list of
numbers), -1 standing for any number, or null; "url" is the board on puzz.link. <task>.txt is the task's prompt, the
board's size and givens to be written in.

A reply's answer is its last fenced code block (```) or, if it closes none, all after the fence it leaves open: h
lines of w characters, S for a shaded cell and . for an unshaded one on Nurikabe, Tapa and Heyawake; L for a light
and . for any other cell on Light-Up, whose black cells may also be written # or as their digit. Spaces and tabs in a
line are dropped, and lower case is read as upper case. The answer is valid when it solves its board under the rules
of the datasets' checkers (ics/tasks).
"""
from __future__ import annotations

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ics.config import merge_config
from ics.tasks.lightup import BULB, EMPTY, W0, WALL, akari_valid
from ics.tasks.ppb_rules import SHADE, WHITE, heyawake_valid, nurikabe_valid, tapa_valid

TASKS = ("lightup", "nurikabe", "tapa", "heyawake")
HERE = Path(__file__).parent
FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)        # a fenced code block: its fence's line, then up to ```
OPEN = re.compile(r"```[^\n]*\n(.*)\Z", re.DOTALL)            # a fence left open: its line, then all to the end
ANSWER = {"lightup": "L.#01234", "nurikabe": "S.", "tapa": "S.", "heyawake": "S."}    # the characters of an answer
CELLS = {".": EMPTY, "#": WALL, **{str(n): W0 + n for n in range(5)}}                # a Light-Up board's, as tokens
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}                # the API key's variable
BASE_URLS = {"openai": "http://localhost:8000/v1", "anthropic": "https://api.anthropic.com"}
TIMEOUT, RETRY_WAIT, MAX_WAIT = 3600, 30, 300   # seconds: a reply's longest silence; the wait before the retry, unless
                                                # the server's Retry-After says another, at most MAX_WAIT
REFUSED = (400, 401, 403, 404)                  # HTTP codes of a request that every board's would meet


class RequestError(Exception):
    """The server refused the request itself (HTTP 400, 401, 403 or 404), as it would refuse every board's."""


def golden(task: str) -> list[dict]:
    """The task's golden boards, board k at index k."""
    return json.loads((HERE / "golden_boards.json").read_text())[task]


# prompts
def prompt(task: str, board: dict) -> str:
    """The prompt of `board`: the task's template with the board's size and givens written in."""
    fields = {"H": str(board["h"]), "W": str(board["w"])}
    if task == "lightup":
        fields["BOARD"] = "\n".join(board["cells"])
    elif task == "heyawake":
        rooms = board["rooms"]
        clued = {room: n for line, clues in zip(rooms, board["clues"]) for room, n in zip(line, clues) if n is not None}
        unclued = sorted({room for line in rooms for room in line} - clued.keys())
        lines = [f"room {room}: {n}" for room, n in sorted(clued.items())]
        if unclued:
            lines.append(f"Rooms with no clue ({', '.join(map(str, unclued))}) may have any number of shaded cells.")
        fields["BOARD"] = _table([[str(room) for room in line] for line in rooms], width=2)
        fields["CLUES"] = "\n".join(lines)
    else:
        fields["BOARD"] = _table([["." if clue is None else _clue(clue) for clue in line] for line in board["clues"]])
    text = (HERE / f"{task}.txt").read_text()
    for name, value in fields.items():
        text = text.replace("{" + name + "}", value)
    return text


def _clue(clue) -> str:
    """A Nurikabe or Tapa clue as a prompt writes it: its numbers joined by commas, ? for any number."""
    return ",".join("?" if n == -1 else str(n) for n in (clue if isinstance(clue, list) else [clue]))


def _table(cells: list[list[str]], width: int = 1) -> str:
    """Lines of cells joined by spaces, every cell right-aligned to the widest, or to `width` if that is wider."""
    width = max(width, max(len(cell) for line in cells for cell in line))
    return "\n".join(" ".join(cell.rjust(width) for cell in line) for line in cells)


# answers
def answer(task: str, reply: str, h: int, w: int) -> list[str]:
    """The answer in `reply`: its last fenced code block, or all after a fence it leaves open, as h lines of w
    characters, upper case and without spaces (the module docstring); ValueError says what keeps the reply from holding
    one."""
    if blocks := FENCED.findall(reply):
        block = blocks[-1]
    elif opened := OPEN.search(reply):
        block = opened.group(1)
    else:
        raise ValueError("no fenced code block")
    lines = [line.rstrip() for line in block.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if len(lines) != h:
        raise ValueError(f"line count {len(lines)}, not {h}")
    rows = [line.replace(" ", "").replace("\t", "").upper() for line in lines]
    for i, row in enumerate(rows, 1):
        if len(row) != w:
            raise ValueError(f"line {i}: length {len(row)}, not {w}")
        if other := "".join(sorted(set(row) - set(ANSWER[task]))):
            raise ValueError(f"line {i}: {other!r}, not one of {ANSWER[task]!r}")
    return rows


def grade(task: str, board: dict, reply: str) -> dict:
    """{"format_ok", "valid", "reason"}: whether `reply` holds an answer, whether it solves `board`, and why: what
    keeps the reply from holding an answer, or the checker's verdict, "OK" or the first rule the answer breaks
    (Light-Up's checker names none: "not a solution")."""
    try:
        rows = answer(task, reply, board["h"], board["w"])
    except ValueError as e:
        return {"format_ok": False, "valid": False, "reason": str(e)}
    if task == "lightup":
        cells = np.array([[CELLS[c] for c in line] for line in board["cells"]])
        lights = np.where(np.array([list(row) for row in rows]) == "L", BULB, cells)
        reason = "OK" if akari_valid(cells, lights) else "not a solution"
    elif task == "heyawake":
        reason = heyawake_valid(board["rooms"], board["clues"], _states(rows))
    else:
        reason = (tapa_valid if task == "tapa" else nurikabe_valid)(board["clues"], _states(rows))
    return {"format_ok": True, "valid": reason == "OK", "reason": reason}


def _states(rows: list[str]) -> list[list[int]]:
    return [[SHADE if c == "S" else WHITE for c in row] for row in rows]


# the client
def query(prompt: str, model: str, provider: str = "openai", base_url: str | None = None, max_tokens: int = 16000,
          extra: dict | None = None, headers: dict | None = None) -> tuple[str, dict]:
    """`model`'s reply to `prompt`, sent as one user message to an OpenAI-compatible server's /chat/completions or,
    with provider "anthropic", to Anthropic's /v1/messages. `base_url` is the API's root, an OpenAI-compatible server's
    with its /v1 (BASE_URLS: vLLM's on this machine; Anthropic's). The key is read from KEYS[provider]; an
    OpenAI-compatible server may need none. `extra` is merged into the request, a null dropping its field, and `headers`
    are added to it. The reply streams in, unless `extra` sets "stream" false.

    Returned are the reply's text and a record of the call: the request without the prompt, the stop reason, the
    server's token usage and the seconds taken. A reply counts only if it came whole, with a stop reason or the stream's
    end event ([DONE], message_stop): a response that ends before, breaks off, falls silent for TIMEOUT seconds or does
    not parse (a proxy's HTML page, say) raises ConnectionError. An HTTP 400, 401, 403 or 404 raises RequestError, any
    other HTTP error OSError, after one retry of a 429 or 5xx."""
    key = os.environ.get(KEYS[provider], "")
    base = (base_url or BASE_URLS[provider]).rstrip("/")
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}], "stream": True}
    if provider == "anthropic":
        url, auth = f"{base}/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, auth = f"{base}/chat/completions", {"authorization": f"Bearer {key}"} if key else {}
        body["stream_options"] = {"include_usage": True}
    body = _without_nulls(merge_config(body, extra))
    if not body.get("stream"):
        body.pop("stream_options", None)
    start = time.time()
    try:
        with _post(url, {"content-type": "application/json", **auth, **(headers or {})}, body) as response:
            text, stop, usage, ended = (_streamed if body.get("stream") else _whole)(response, provider)
    except (AttributeError, http.client.HTTPException, TimeoutError, TypeError, ValueError) as e:
        raise ConnectionError(f"no complete reply ({e!r})") from None     # cut, silent, not JSON, JSON of another shape
    if not ended:
        raise ConnectionError("no complete reply: the response ended without a stop reason or its end event")
    request = {k: v for k, v in body.items() if k != "messages"}
    return text, {"request": request, "stop_reason": stop, "usage": usage, "seconds": round(time.time() - start, 1)}


def _without_nulls(fields: dict) -> dict:
    """`fields` without those whose value is null, at every depth."""
    return {k: _without_nulls(v) if isinstance(v, dict) else v for k, v in fields.items() if v is not None}


def _post(url: str, headers: dict, body: dict):
    """The open response to `body` POSTed as JSON, which is sent once more if the server answers 429 (too many
    requests) or 5xx, after the seconds its Retry-After gives (at most MAX_WAIT) or RETRY_WAIT, as a line on stderr
    says. An HTTP error raises, with the server's message, RequestError if its code is in REFUSED, else OSError."""
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    for retry in (False, True):
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as e:
            if retry or (e.code != 429 and e.code < 500):
                message = " ".join(e.read().decode(errors="replace").split())[:1000]
                error = RequestError if e.code in REFUSED else OSError
                raise error(f"HTTP {e.code} from {url}: {message}") from None
            after = str(e.headers.get("retry-after") or "").strip() if e.headers else ""
            wait = min(int(after), MAX_WAIT) if after.isdecimal() else RETRY_WAIT
            print(f"HTTP {e.code} from {url}: asking again in {wait} s", file=sys.stderr)
            time.sleep(wait)


def _streamed(response, provider: str) -> tuple[str, str | None, dict, bool]:
    """A reply streamed as server-sent events: its text, stop reason and usage, and whether it came whole (a stop
    reason, or the end event: [DONE], message_stop). Only the text is the reply, not the thinking (Anthropic's
    thinking deltas, a reasoning_content)."""
    text, stop, usage, end = [], None, {}, False
    for line in response:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        if (data := line[5:].strip()) == "[DONE]":
            end = True
            break
        event = _json_object(data)
        if event.get("error") or event.get("object") == "error":
            raise ConnectionError(f"no complete reply: the server broke it off: {event.get('error') or event}")
        if provider == "anthropic":
            kind, delta = event.get("type"), event.get("delta") or {}
            if kind == "message_start":
                usage.update((event.get("message") or {}).get("usage") or {})
            elif kind == "content_block_delta" and delta.get("type") == "text_delta":
                text.append(delta.get("text") or "")
            elif kind == "message_delta":
                stop = delta.get("stop_reason") or stop
                usage.update(event.get("usage") or {})
            end = end or kind == "message_stop"
        else:
            for choice in event.get("choices") or []:
                text.append((choice.get("delta") or {}).get("content") or "")
                stop = choice.get("finish_reason") or stop
            usage.update(event.get("usage") or {})
    return "".join(text), stop, usage, end or stop is not None


def _whole(response, provider: str) -> tuple[str, str | None, dict, bool]:
    """A reply sent whole (stream false): its text, stop reason and usage, and whether it came whole (a stop reason).
    Anthropic's text is its text blocks, not its thinking."""
    reply = _json_object(response.read())
    if provider == "anthropic":
        text = "".join(block.get("text", "") for block in reply.get("content") or [] if block.get("type") == "text")
        stop = reply.get("stop_reason")
    else:
        choice = (reply.get("choices") or [{}])[0]
        text, stop = (choice.get("message") or {}).get("content") or "", choice.get("finish_reason")
    return text, stop, reply.get("usage") or {}, stop is not None


def _json_object(data) -> dict:
    """`data` (bytes or str) parsed as a JSON object; ValueError if it is not JSON, TypeError if not an object."""
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError(f"not a JSON object: {str(value)[:80]}")
    return value
