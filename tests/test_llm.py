import http.client
import io
import json
import re
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fakes import Stop, stop
from ics import llm
from ics.cli import main
from ics.llm import TASKS, answer, golden, grade, prompt, query

# Tiny boards, made by hand, with an answer that solves each and one that does not (and the rule it breaks).
# Light-Up's is test_lightup.py's board: a black cell with clue 1 at (0, 1), one without at (1, 2); its one solution
# has lights at (0, 2), (1, 0) and (1, 3).
BOARDS = {"lightup": {"h": 2, "w": 4, "cells": [".1..", "..#."]},
          "nurikabe": {"h": 2, "w": 3, "clues": [[2, None, None], [None, None, 1]]},
          "tapa": {"h": 2, "w": 3, "clues": [[None, None, None], [None, [3], None]]},
          "heyawake": {"h": 2, "w": 3, "rooms": [[0, 0, 1], [0, 0, 1]], "clues": [[1, None, None], [None, None, None]]}}
SOLUTIONS = {"lightup": ["..L.", "L..L"], "nurikabe": [".SS", ".S."], "tapa": ["SSS", "..."],
             "heyawake": ["...", ".S."]}
WRONG = {"lightup": (["L.L.", "L..L"], "not a solution"), "nurikabe": ([".SS", "SS."], "island-size"),
         "tapa": (["S.S", "..."], "split"), "heyawake": (["SS.", "..."], "adjacent-shade")}


def fenced(rows: list[str]) -> str:
    return "```\n" + "\n".join(rows) + "\n```"


def stream(*events) -> bytes:
    """A server-sent event stream of the events, dicts (each after its event line if it has a type) or "[DONE]"."""
    out = b""
    for e in events:
        if isinstance(e, dict) and "type" in e:
            out += f"event: {e['type']}\n".encode()
        out += b"data: " + (e if isinstance(e, str) else json.dumps(e)).encode() + b"\n\n"
    return out


def chunk(content: str | None = None, finish: str | None = None) -> dict:
    """A chunk of an OpenAI-compatible stream: a content delta and a finish reason."""
    return {"choices": [{"delta": {} if content is None else {"content": content}, "finish_reason": finish}]}


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = http.client.HTTPMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, "error", headers, io.BytesIO(b'{"error": "busy"}'))


class Broken(io.BytesIO):
    """A response that delivers its bytes, then breaks off with `error` (a connection cut, a server gone silent)."""

    def __init__(self, data: bytes, error: Exception):
        super().__init__(data)
        self.error = error

    def __iter__(self):
        yield from iter(self.readline, b"")
        raise self.error

    def read(self, *args):
        super().read(*args)
        raise self.error


@pytest.fixture
def server(monkeypatch):
    """urlopen's stand-in, no network: each call answers with the next of server.replies, the bytes of a response, a
    response, or an exception to raise; server.requests records each call's (request, timeout), and server.waits the
    seconds waited before a retry (none really)."""
    s = types.SimpleNamespace(replies=[], requests=[], waits=[])

    def urlopen(request, timeout):
        s.requests.append((request, timeout))
        reply = s.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply) if isinstance(reply, bytes) else reply

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", s.waits.append)
    return s


@pytest.fixture
def boards(monkeypatch):
    """The golden boards the command line sees: two of each task, its tiny board twice."""
    monkeypatch.setattr("ics.cli.golden", lambda task: [BOARDS[task], BOARDS[task]])


# prompts
@pytest.mark.parametrize("task, board, drawn", [
    ("lightup", BOARDS["lightup"], ".1..\n..#."),
    ("nurikabe", {"h": 2, "w": 2, "clues": [[10, None], [None, 2]]}, "10  .\n .  2"),
    ("tapa", {"h": 2, "w": 2, "clues": [[[1, 3], None], [[-1], [0]]]}, "1,3   .\n  ?   0"),
    ("heyawake", BOARDS["heyawake"], (" 0  0  1\n 0  0  1\n\nRoom clues (required number of shaded cells in that room):"
                                      "\nroom 0: 1\nRooms with no clue (1) may have any number of shaded cells.")),
    ("heyawake", {"h": 2, "w": 2, "rooms": [[0, 0], [0, 0]], "clues": [[None, None], [None, 3]]},
     " 0  0\n 0  0\n\nRoom clues (required number of shaded cells in that room):\nroom 0: 3"),     # every room clued
])
def test_a_prompt_holds_the_board_its_size_and_the_answer_format(task, board, drawn):
    text = prompt(task, board)
    h, w = board["h"], board["w"]
    answer_format = f"Think carefully, then output your final answer as exactly {h} lines of {w} characters"
    assert f":\n{drawn}\n\n{answer_format}" in text
    assert f"The board has {h} rows and {w} columns. Rows are numbered 1-{h} top to bottom, columns 1-{w}" in text
    assert "{" not in text and text.isascii()


# answers
def test_an_answer_is_the_last_fenced_block_spaces_dropped_and_in_upper_case():
    reply = "First try:\n```\nSSS\n...\n```\nNo, rather:\n```text\n\n . s s \n.\tS.  \n\n```\nDone."
    assert answer("nurikabe", reply, 2, 3) == [".SS", ".S."]
    assert answer("lightup", fenced([".1l.", "L.#L"]), 2, 4) == [".1L.", "L.#L"]       # black cells echoed


def test_a_fence_left_open_holds_the_answer_when_the_reply_closes_none():
    assert answer("nurikabe", "My answer:\n```\n.SS\n.S.\n", 2, 3) == [".SS", ".S."]
    assert answer("nurikabe", "```\n.SS\n.S.\n```\nOr rather:\n```\nSSS\n...", 2, 3) == [".SS", ".S."]


def test_light_up_reads_l_as_a_light_and_dot_hash_or_a_digit_as_none():
    for other in ".#01234":
        rows = [row.replace(".", other) for row in SOLUTIONS["lightup"]]
        assert grade("lightup", BOARDS["lightup"], fenced(rows)) == {"format_ok": True, "valid": True, "reason": "OK"}
    for other in "S5x*":
        assert grade("lightup", BOARDS["lightup"], fenced(["..L" + other, "L..L"])) == {
            "format_ok": False, "valid": False, "reason": f"line 1: {other.upper()!r}, not one of 'L.#01234'"}


@pytest.mark.parametrize("task, reply, why", [
    ("nurikabe", ".SS\n.S.", "no fenced code block"),
    ("nurikabe", "```\n.SS\n```", "line count 1, not 2"),
    ("nurikabe", "```\n.SS\n.S\n```", "line 2: length 2, not 3"),
    ("nurikabe", "```\n.SX\n.S.\n```", "line 1: 'X', not one of 'S.'"),
    ("tapa", "```\n.S#\n.S.\n```", "line 1: '#', not one of 'S.'"),
    ("lightup", "```\n.1S.\nL.#L\n```", "line 1: 'S', not one of 'L.#01234'"),
])
def test_a_reply_without_an_answer_says_why(task, reply, why):
    board = BOARDS[task]
    with pytest.raises(ValueError, match=re.escape(why)):
        answer(task, reply, board["h"], board["w"])
    assert grade(task, board, reply) == {"format_ok": False, "valid": False, "reason": why}


@pytest.mark.parametrize("task", TASKS)
def test_a_solution_is_valid_and_another_answer_breaks_a_rule(task):
    assert grade(task, BOARDS[task], fenced(SOLUTIONS[task])) == {"format_ok": True, "valid": True, "reason": "OK"}
    rows, rule = WRONG[task]
    assert grade(task, BOARDS[task], fenced(rows)) == {"format_ok": True, "valid": False, "reason": rule}


def test_a_light_up_board_wider_than_the_datasets_canvas_is_graded():
    board = {"h": 1, "w": 30, "cells": ["." * 14 + "#" + "." * 15]}
    assert prompt("lightup", board).count("." * 14 + "#" + "." * 15) == 1
    assert grade("lightup", board, fenced(["L" + "." * 14 + "L" + "." * 14]))["valid"]
    assert not grade("lightup", board, fenced(["L" + "." * 29]))["valid"]                # the right side is dark


def test_the_golden_boards_are_givens_of_the_size_they_state():
    for task in TASKS:
        boards = golden(task)
        assert [b["board"] for b in boards] == list(range(15))
        grids = {"lightup": ["cells"], "heyawake": ["rooms", "clues"]}.get(task, ["clues"])
        for b in boards:
            assert set(b) == {"board", "h", "w", "url", *grids}
            assert all(len(b[g]) == b["h"] and {len(line) for line in b[g]} == {b["w"]} for g in grids)
            assert "{" not in prompt(task, b)
    assert max(max(b["h"], b["w"]) for b in golden("lightup")) > 24       # larger than the datasets' boards


# the client
def test_query_streams_a_reply_from_an_openai_compatible_server(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    server.replies.append(stream({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                                 {"choices": [{"delta": {"reasoning_content": "hm"}, "finish_reason": None}]},
                                 chunk("```\n.S"), chunk("S\n```", "stop"),
                                 {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}, "[DONE]"))
    text, record = query("the prompt", "some-model", base_url="http://host:1/v1/", max_tokens=99,
                         extra={"temperature": 0.5}, headers={"x-extra": "1"})
    assert text == "```\n.SS\n```"
    (request, timeout), = server.requests
    assert (request.full_url, timeout) == ("http://host:1/v1/chat/completions", llm.TIMEOUT)
    assert [request.get_header(h) for h in ("Authorization", "Content-type", "X-extra")] == [
        "Bearer key", "application/json", "1"]
    settings = {"model": "some-model", "max_tokens": 99, "stream": True, "stream_options": {"include_usage": True},
                "temperature": 0.5}
    assert json.loads(request.data) == {**settings, "messages": [{"role": "user", "content": "the prompt"}]}
    assert record["request"] == settings and record["stop_reason"] == "stop"
    assert record["usage"] == {"prompt_tokens": 9, "completion_tokens": 4}
    monkeypatch.delenv("OPENAI_API_KEY")                                    # a local server needs no key
    server.replies.append(stream({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}))
    assert query("p", "m")[0] == "x"
    request = server.requests[-1][0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions" and not request.has_header("Authorization")


def test_query_streams_a_reply_from_anthropic(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    server.replies.append(stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 9, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "```\n.SS"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "\n.S.\n```"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 30}},
        {"type": "message_stop"}))
    text, record = query("the prompt", "some-model", provider="anthropic")
    assert text == "```\n.SS\n.S.\n```"
    (request, _), = server.requests
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    assert (request.get_header("X-api-key"), request.get_header("Anthropic-version")) == ("key", "2023-06-01")
    assert json.loads(request.data) == {"model": "some-model", "max_tokens": 16000, "stream": True,
                                        "messages": [{"role": "user", "content": "the prompt"}]}
    assert record["stop_reason"] == "end_turn" and record["usage"] == {"input_tokens": 9, "output_tokens": 30}
    server.replies.append(stream({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    with pytest.raises(ConnectionError, match="overloaded_error"):
        query("the prompt", "some-model", provider="anthropic")


def test_a_streamed_reply_is_whole_at_its_stop_reason_or_its_end_event(server):
    server.replies += [stream(chunk("a"), chunk("b"), "[DONE]"),                     # no finish reason, but [DONE]
                       stream(chunk("a", "stop"), chunk("b"), {"choices": [], "usage": {"completion_tokens": 2}}),
                       stream(chunk("a"), chunk("b"))]                                # neither: the reply is cut off
    assert query("p", "m")[0] == "ab"
    text, record = query("p", "m")                                  # chunks after the finish reason are read too
    assert (text, record["stop_reason"], record["usage"]) == ("ab", "stop", {"completion_tokens": 2})
    with pytest.raises(ConnectionError, match="no complete reply: the response ended without a stop reason"):
        query("p", "m")
    a = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "a"}}
    server.replies += [stream(a, {"type": "message_stop"}),
                       stream(a, {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}), stream(a)]
    assert query("p", "m", provider="anthropic")[0] == "a"
    assert query("p", "m", provider="anthropic")[1]["stop_reason"] == "end_turn"
    with pytest.raises(ConnectionError, match="no complete reply"):
        query("p", "m", provider="anthropic")


@pytest.mark.parametrize("response, why", [
    (Broken(stream(chunk("```")), http.client.IncompleteRead(b"")), "IncompleteRead(0 bytes read)"),
    (Broken(stream(chunk("```")), TimeoutError("timed out")), "TimeoutError('timed out')"),
    (b"data: {not json\n\n", "JSONDecodeError"),
    (b"data: \xff\n\n", "UnicodeDecodeError"),
    (b"data: [1, 2]\n\n", "not a JSON object"),
    (stream(chunk("a"), {"error": {"message": "the server is overloaded", "type": "server_error"}}), "overloaded"),
    (stream({"object": "error", "message": "bad request", "type": "BadRequestError", "code": 400}), "bad request"),
    (b"<html><body>502 Bad Gateway</body></html>\n", "the response ended without a stop reason"),   # a proxy's page
    (stream({"choices": ["x"]}), "AttributeError"),                                   # JSON of another shape
])
def test_a_reply_that_breaks_off_or_does_not_parse_raises_a_one_line_connection_error(server, response, why):
    server.replies.append(response)
    with pytest.raises(ConnectionError, match=re.escape(why)) as raised:
        query("p", "m")
    assert str(raised.value).startswith("no complete reply") and "\n" not in str(raised.value)


def test_a_chunk_whose_error_is_null_is_no_error(server):
    server.replies.append(stream({**chunk("ok", "stop"), "error": None}))
    assert query("p", "m")[0] == "ok"


def test_a_reply_comes_whole_when_the_request_sets_stream_false(server):
    whole = lambda reply: json.dumps(reply).encode()
    server.replies += [whole({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                              "usage": {"completion_tokens": 1}}),
                       whole({"content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "a"},
                                          {"type": "text", "text": "b"}], "stop_reason": "end_turn"}),
                       whole({"choices": [{"message": {"content": "hi"}}]}),             # no finish reason
                       Broken(b'{"choices": [', http.client.IncompleteRead(b'{"choices": [')),
                       b"<html><body>502 Bad Gateway</body></html>"]                     # a proxy's page
    text, record = query("p", "m", extra={"stream": False})
    assert (text, record["stop_reason"], record["usage"]) == ("hi", "stop", {"completion_tokens": 1})
    assert json.loads(server.requests[0][0].data) == {"model": "m", "max_tokens": 16000, "stream": False,
                                                      "messages": [{"role": "user", "content": "p"}]}
    assert query("p", "m", provider="anthropic", extra={"stream": False})[0] == "ab"
    with pytest.raises(ConnectionError, match="without a stop reason"):
        query("p", "m", extra={"stream": False})
    with pytest.raises(ConnectionError, match="IncompleteRead"):
        query("p", "m", extra={"stream": False})
    with pytest.raises(ConnectionError, match="JSONDecodeError"):
        query("p", "m", extra={"stream": False})


def test_a_null_in_the_request_drops_its_field(server):
    server.replies.append(stream(chunk("a", "stop")))
    query("p", "m", extra={"max_tokens": None, "max_completion_tokens": 64, "metadata": {"run": None, "seed": 1},
                           "stream_options": None})
    assert json.loads(server.requests[0][0].data) == {"model": "m", "max_completion_tokens": 64, "stream": True,
                                                      "metadata": {"seed": 1},
                                                      "messages": [{"role": "user", "content": "p"}]}


def test_query_retries_a_429_or_5xx_once_waiting_as_the_server_asks_and_tells_a_refusal_apart(server):
    ok = stream(chunk("ok", "stop"))
    server.replies += [http_error(429), ok, http_error(503, "7"), ok, http_error(429, "86400"), ok,
                       http_error(529, "Wed, 21 Oct 2026 07:28:00 GMT"), ok]
    assert [query("p", "m")[0] for _ in range(4)] == ["ok"] * 4
    assert server.waits == [llm.RETRY_WAIT, 7, llm.MAX_WAIT, llm.RETRY_WAIT]      # Retry-After in seconds, capped
    server.replies += [http_error(503), http_error(500)]
    with pytest.raises(OSError, match=re.escape('HTTP 500 from http://localhost:8000/v1/chat/completions: {"error"')):
        query("p", "m")
    server.replies += [http_error(400), http_error(422)]
    with pytest.raises(llm.RequestError, match="HTTP 400"):                  # every board's request would meet it
        query("p", "m")
    with pytest.raises(OSError, match="HTTP 422"):
        query("p", "m")
    assert len(server.requests) == 12


def test_a_retry_says_on_stderr_how_long_it_waits(server, capsys):
    ok = stream(chunk("ok", "stop"))
    server.replies += [http_error(429, "86400"), ok, http_error(503), ok]
    assert [query("p", "m")[0] for _ in range(2)] == ["ok", "ok"] and server.waits == [llm.MAX_WAIT, llm.RETRY_WAIT]
    url = "http://localhost:8000/v1/chat/completions"
    assert capsys.readouterr().err == (f"HTTP 429 from {url}: asking again in {llm.MAX_WAIT} s\n"
                                       f"HTTP 503 from {url}: asking again in {llm.RETRY_WAIT} s\n")


# the command line
@pytest.mark.parametrize("step, option", [("prompt", "--board K"), ("run", "--base-url URL"),
                                          ("run", "--header KEY=VALUE"), ("grade", "--replies")])
def test_llm_help(step, option, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["llm", step, "--help"])
    assert stop.value.code == 0 and option in capsys.readouterr().out


def test_llm_prompt_writes_the_prompts(tmp_path, capsys, boards):
    main(["llm", "prompt", "--task", "tapa", "--out", str(tmp_path / "all")])
    main(["llm", "prompt", "--task", "heyawake", "--board", "1", "--out", str(tmp_path / "one")])
    assert capsys.readouterr().out == f"wrote {tmp_path / 'all'}\nwrote {tmp_path / 'one'}\n"
    assert sorted(p.name for p in (tmp_path / "all").iterdir()) == ["tapa_0.prompt.txt", "tapa_1.prompt.txt"]
    assert (tmp_path / "all" / "tapa_1.prompt.txt").read_bytes() == prompt("tapa", BOARDS["tapa"]).encode()
    assert [p.name for p in (tmp_path / "one").iterdir()] == ["heyawake_1.prompt.txt"]


def test_llm_run_writes_each_reply_with_its_usage_and_keeps_the_replies_there(tmp_path, capsys, boards, server):
    out = tmp_path / "replies"
    out.mkdir()
    (out / "nurikabe_0.txt").write_text("an earlier reply")
    server.replies.append(stream(chunk(fenced(SOLUTIONS["nurikabe"]), "stop"),
                                 {"choices": [], "usage": {"completion_tokens": 5}}))
    main(["llm", "run", "--task", "nurikabe", "--model", "m", "--out", str(out), "--base-url", "http://host/v1",
          "--max-tokens", "64", "--set", "temperature=0.5", "--set", "chat_template_kwargs.enable_thinking=false",
          "--header", "x-extra=a=b"])
    assert capsys.readouterr().out == (f"board 0: kept {out / 'nurikabe_0.txt'}\n"
                                       f"board 1: wrote {out / 'nurikabe_1.txt'} (stop reason stop)\n")
    (request, _), = server.requests
    assert request.full_url == "http://host/v1/chat/completions" and request.get_header("X-extra") == "a=b"
    assert json.loads(request.data)["messages"] == [{"role": "user", "content": prompt("nurikabe", BOARDS["nurikabe"])}]
    assert (out / "nurikabe_0.txt").read_text() == "an earlier reply"
    assert (out / "nurikabe_1.txt").read_text() == fenced(SOLUTIONS["nurikabe"])
    usage = json.loads((out / "nurikabe_1.usage.json").read_text())
    assert usage["request"] == {"model": "m", "max_tokens": 64, "stream": True,
                                "stream_options": {"include_usage": True}, "temperature": 0.5,
                                "chat_template_kwargs": {"enable_thinking": False}}
    assert usage["stop_reason"] == "stop" and usage["usage"] == {"completion_tokens": 5}


def test_llm_run_without_anthropics_key_is_a_usage_error(tmp_path, capsys, boards, server, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        main(["llm", "run", "--task", "tapa", "--model", "m", "--out", str(tmp_path / "r"), "--provider", "anthropic"])
    assert stop.value.code == 2 and not server.requests and not (tmp_path / "r").exists()
    assert ("python -m ics llm run: error: --provider anthropic reads its key from ANTHROPIC_API_KEY, which is not set"
            in capsys.readouterr().err)


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_llm_run_stops_at_once_in_one_line_when_the_server_refuses_the_request(tmp_path, capsys, boards, server, code):
    server.replies.append(http_error(code))
    with pytest.raises(SystemExit) as stop:
        main(["llm", "run", "--task", "tapa", "--model", "m", "--out", str(tmp_path / "r")])
    assert stop.value.code == 1 and len(server.requests) == 1 and not list((tmp_path / "r").iterdir())
    assert capsys.readouterr().err == (f"python -m ics llm run: error: board 0: HTTP {code} from "
                                       'http://localhost:8000/v1/chat/completions: {"error": "busy"}\n')


def test_llm_run_skips_a_board_that_fails_with_no_file_and_a_rerun_asks_it_again(tmp_path, capsys, server,
                                                                                 monkeypatch):
    monkeypatch.setattr("ics.cli.golden", lambda task: [BOARDS["tapa"]] * 7)
    out, solved = tmp_path / "replies", stream(chunk(fenced(SOLUTIONS["tapa"]), "stop"))
    server.replies += [solved,
                       stream(chunk("```\nSS")),                                        # 1: cut short
                       http_error(503), http_error(500),                                # 2: a 5xx after the retry
                       Broken(stream(chunk("```")), http.client.IncompleteRead(b"")),    # 3: the connection cut
                       Broken(b"", TimeoutError("timed out")),                           # 4: a silent server
                       stream({"error": {"message": "overloaded"}}),                     # 5: an error event
                       stream(chunk("", "length"))]                                     # 6: whole, though empty
    with pytest.raises(SystemExit) as stop:
        main(["llm", "run", "--task", "tapa", "--model", "m", "--out", str(out)])
    printed = capsys.readouterr()
    assert stop.value.code == 1 and printed.out == (
        f"board 0: wrote {out / 'tapa_0.txt'} (stop reason stop)\n"
        "board 1: skipped: no complete reply: the response ended without a stop reason or its end event\n"
        'board 2: skipped: HTTP 500 from http://localhost:8000/v1/chat/completions: {"error": "busy"}\n'
        "board 3: skipped: no complete reply (IncompleteRead(0 bytes read))\n"
        "board 4: skipped: no complete reply (TimeoutError('timed out'))\n"
        "board 5: skipped: no complete reply: the server broke it off: {'message': 'overloaded'}\n"
        f"board 6: wrote {out / 'tapa_6.txt'} (stop reason length, an empty reply)\n")
    assert printed.err == ("HTTP 503 from http://localhost:8000/v1/chat/completions: asking again in 30 s\n"   # 2
                           "python -m ics llm run: error: skipped board 1, 2, 3, 4, 5; run the command again to ask "
                           "them again\n")
    assert sorted(p.name for p in out.iterdir()) == ["tapa_0.txt", "tapa_0.usage.json", "tapa_6.txt",
                                                     "tapa_6.usage.json"]
    assert (out / "tapa_6.txt").read_text() == "" and len(server.requests) == 8
    server.replies += [solved] * 5
    main(["llm", "run", "--task", "tapa", "--model", "m", "--out", str(out)])           # boards 1 to 5 alone
    assert len(server.requests) == 13 and all((out / f"tapa_{k}.txt").read_text() == fenced(SOLUTIONS["tapa"])
                                              for k in range(6))


def test_llm_run_skips_a_board_whose_reply_has_another_shape_in_one_line(tmp_path, capsys, boards, server):
    server.replies += [stream({"choices": ["x"]}), stream(chunk(fenced(SOLUTIONS["tapa"]), "stop"))]
    with pytest.raises(SystemExit) as stop:
        main(["llm", "run", "--task", "tapa", "--model", "m", "--out", str(tmp_path)])
    printed = capsys.readouterr()
    assert stop.value.code == 1 and printed.out == (
        "board 0: skipped: no complete reply (AttributeError(\"'str' object has no attribute 'get'\"))\n"
        f"board 1: wrote {tmp_path / 'tapa_1.txt'} (stop reason stop)\n")
    assert printed.err == "python -m ics llm run: error: skipped board 0; run the command again to ask them again\n"


def test_llm_run_names_a_reply_as_a_reply_only_once_it_is_written_whole(tmp_path, boards, server, monkeypatch):
    server.replies.append(stream(chunk(fenced(SOLUTIONS["tapa"]), "stop")))
    monkeypatch.setattr(Path, "replace", stop)                  # the run killed between the write and the rename
    with pytest.raises(Stop):
        main(["llm", "run", "--task", "tapa", "--model", "m", "--board", "0", "--out", str(tmp_path)])
    assert not (tmp_path / "tapa_0.txt").exists() and (tmp_path / "tapa_0.txt.part").exists()


def test_llm_grade_prints_each_verdict_and_the_count_and_writes_them(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("ics.cli.golden", lambda task: [BOARDS["lightup"]] * 4)
    replies = tmp_path / "replies"
    replies.mkdir()
    (replies / "lightup_0.txt").write_text("The lights:\n" + fenced(SOLUTIONS["lightup"]) + "\n")
    (replies / "lightup_1.txt").write_text(fenced(WRONG["lightup"][0]))
    (replies / "lightup_2.txt").write_text("I cannot solve it.")
    (replies / "lightup_3.prompt.txt").write_text(fenced(SOLUTIONS["lightup"]))       # a prompt: not a reply
    main(["llm", "grade", "--task", "lightup", "--replies", str(replies)])
    assert capsys.readouterr().out == ("board 0: valid\nboard 1: invalid (not a solution)\n"
                                       "board 2: no answer (no fenced code block)\nboard 3: no answer (no reply)\n"
                                       f"lightup: 1/4 rule-valid; wrote {replies / 'lightup_grades.json'}\n")
    verdicts = [(True, True, "OK"), (True, False, "not a solution"), (False, False, "no fenced code block"),
                (False, False, "no reply")]
    assert json.loads((replies / "lightup_grades.json").read_text()) == {
        "task": "lightup", "valid": 1, "boards": [{"board": k, "format_ok": f, "valid": v, "reason": r}
                                                  for k, (f, v, r) in enumerate(verdicts)]}
    with pytest.raises(SystemExit) as stop:
        main(["llm", "grade", "--task", "lightup", "--replies", str(tmp_path / "none")])
    assert stop.value.code == 2 and f"--replies {tmp_path / 'none'} is not a directory" in capsys.readouterr().err


def test_llm_grade_reads_a_reply_that_is_not_utf8(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("ics.cli.golden", lambda task: [BOARDS["lightup"]])
    (tmp_path / "lightup_0.txt").write_bytes(b"The lights \xff\xfe:\n" + fenced(SOLUTIONS["lightup"]).encode())
    main(["llm", "grade", "--task", "lightup", "--replies", str(tmp_path)])
    assert capsys.readouterr().out.startswith("board 0: valid\n")
