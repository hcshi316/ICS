"""Command line:
  python -m ics data          --task TASK --out DIR [--seed S] [--source DIR]
  python -m ics download      --model {trm,eqr,gram,attractor,verifier} --task TASK --seed K [--out DIR] [--repo REPO]
  python -m ics download      --dataset --task TASK [--out DIR] [--repo REPO]
  python -m ics train         --model {trm,eqr,gram,attractor,verifier} --task TASK --data DIR --out DIR [--init CKPT]
                              [--set KEY=VALUE ...]   (torchrun for several GPUs)
  python -m ics verifier-data --task TASK --data DIR --solver RUN_OR_CKPT [--solver ...] --out DIR
  python -m ics pick-verifier RUN [RUN ...] --out DIR
  python -m ics eval          --method {trm,ptrm,ics,gram,eqr,attractor} --task TASK --ckpt CKPT --data DIR --out DIR
                              [--verifier CKPT]
  python -m ics llm prompt    --task TASK --out DIR [--board K]
  python -m ics llm run       --task TASK --model MODEL --out DIR [--board K] [--provider {openai,anthropic}]
                              [--base-url URL] [--max-tokens N] [--set KEY=VALUE ...] [--header KEY=VALUE ...]
  python -m ics llm grade     --task TASK --replies DIR
Parsing touches no CUDA, so that train.deterministic sets PyTorch's deterministic mode before CUDA starts
(ics/train.py)."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

import torch
import yaml

from ics.builders import overlap
from ics.builders.ppb import PPBenchUnavailable
from ics.config import merge_config, parse_yaml
from ics.evaluate import REGIMES, evaluate
from ics.llm import TASKS as LLM_TASKS
from ics.llm import RequestError, golden, grade, prompt, query
from ics.registry import HEADS, METHODS, TASKS
from ics.train import train
from ics.verifier.data import build_verifier_data
from ics.verifier.pick import pick_verifier


def _setting(text: str) -> dict:
    """KEY=VALUE (VALUE parsed in the YAML dialect of ics/config.py, as the recipes and eval configs are; KEY may be
    dotted, e.g. cert.levels=2) -> nested dict."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    try:
        out = parse_yaml(value)
    except yaml.YAMLError:
        raise argparse.ArgumentTypeError(f"the value in {text!r} is not valid YAML") from None
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def _overrides(settings: list[dict]) -> dict:
    out = {}
    for over in settings:
        out = merge_config(out, over)
    return out


def _at_least(least: int):
    """The argparse type of an integer option whose values start at `least` (--seed 0, --batch 1)."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        if value < least:
            raise argparse.ArgumentTypeError(f"expected {least} or more, got {value}")
        return value

    return parse


def _device(device: str | None) -> str:
    """--device, by default cuda if available, else cpu."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _taken(out: Path, text: str) -> str | None:
    """Why python -m ics data (download, verifier-data, pick-verifier) must not write into `out` (--out `text`), or
    None. It deletes nothing: it writes only into a missing directory or one holding nothing but dotfiles, which it
    leaves alone."""
    if out.exists() and not out.is_dir():
        return f"--out {text} is not a directory; choose another --out"
    if out.is_dir() and (names := sorted(p.name for p in out.iterdir() if not p.name.startswith("."))):
        shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
        return f"--out {text} is not empty ({shown}); choose another --out, or remove it if it holds an earlier build"
    return None


def _data_args(sub):
    d = sub.add_parser("data", help="build a dataset")
    d.add_argument("--task", choices=TASKS, required=True,
                   help="lightup, nurikabe, tapa and heyawake need the [ppbench] extra and Node.js 16 or newer")
    d.add_argument("--out", required=True, help="dataset directory, missing or empty (dotfiles aside)")
    d.add_argument("--seed", type=_at_least(0),
                   help="seed of the train split's draws (default 0; not lightup or maze); evaluate on the same build")
    d.add_argument("--source", help="directory of the source files (default: download them from Hugging Face)")
    return d


def _data(a, parser):
    options = {"source": a.source}
    if a.seed is not None:
        if "seed" not in inspect.signature(TASKS[a.task].build).parameters:
            parser.error(f"the {a.task} build takes no seed: its train split has no random draws")
        options["seed"] = a.seed
    if taken := _taken(Path(a.out), a.out):
        parser.error(taken)
    try:
        TASKS[a.task].build(a.out, **options)
    except PPBenchUnavailable as e:                  # an expected failure: one line; any other keeps its traceback
        parser.exit(1, f"{parser.prog}: error: {e}\n")
    print(f"wrote {a.out}")


# python -m ics download: a released checkpoint, <model>/<task>/seed<k>/ in MODEL_REPO, or a built dataset, its
# DATA_FOLDERS folder in DATA_REPO (Sudoku's and Maze's are built from their sources: python -m ics data)
MODEL_REPO, DATA_REPO = "hcshi/ICS", "hcshi/ICS-data"
DATA_FOLDERS = {"lightup": "lightup-ms26", "nurikabe": "ppb-nurikabe-ms26", "tapa": "ppb-tapa-ms26",
                "heyawake": "ppb-heyawake-ms26"}


def _download_args(sub):
    d = sub.add_parser("download", help="download a released checkpoint, or a built dataset, from Hugging Face")
    what = d.add_mutually_exclusive_group(required=True)
    what.add_argument("--model", choices=HEADS, help="a checkpoint of this model, with --task and --seed")
    what.add_argument("--dataset", action="store_true", help=f"the task's built dataset ({', '.join(DATA_FOLDERS)})")
    d.add_argument("--task", choices=TASKS, required=True)
    d.add_argument("--seed", type=_at_least(0), help="the checkpoint's training seed")
    d.add_argument("--out", help="directory, missing or empty (dotfiles aside); default ckpt/MODEL/TASK/seedSEED, or "
                                 "with --dataset data/TASK")
    d.add_argument("--repo", help=f"Hugging Face repo (default {MODEL_REPO}, or with --dataset {DATA_REPO})")
    return d


def _fetch(repo: str, repo_type: str, folder: str, out: Path) -> bool:
    """Copy `folder` of a Hugging Face repo into `out` through the Hugging Face cache, as writable files (the cache's
    are read-only); False if the repo has none."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type=repo_type, allow_patterns=f"{folder}/*"))
    if not (root / folder).is_dir():
        return False
    shutil.copytree(root / folder, out, copy_function=shutil.copyfile, dirs_exist_ok=True)
    return True


def _download(a, parser):
    if a.dataset:
        if a.seed is not None:
            parser.error("--seed goes with --model")
        out = a.out or f"data/{a.task}"
        if a.task not in DATA_FOLDERS:
            parser.error(f"the {a.task} dataset is built from its source, not downloaded: python -m ics data --task "
                         f"{a.task} --out {out}")
        repo, repo_type, folder = a.repo or DATA_REPO, "dataset", DATA_FOLDERS[a.task]
    else:
        if a.seed is None:
            parser.error("--model needs --seed")
        out = a.out or f"ckpt/{a.model}/{a.task}/seed{a.seed}"
        repo, repo_type, folder = a.repo or MODEL_REPO, "model", f"{a.model}/{a.task}/seed{a.seed}"
    if taken := _taken(Path(out), out):
        parser.error(taken)
    try:
        fetched = _fetch(repo, repo_type, folder, Path(out))
    except OSError as e:                     # the hub's failures (no network, no such repo): its message, no traceback
        parser.exit(1, f"{parser.prog}: error: {repo}: {e}\n")
    if not fetched:
        parser.error(f"{repo} holds no {folder}/")
    print(f"wrote {out}")


def _train_args(sub):
    t = sub.add_parser("train", help="train one model on one task")
    t.add_argument("--model", choices=HEADS, required=True)
    t.add_argument("--task", choices=TASKS, required=True)
    t.add_argument("--data", required=True, help="built dataset directory (a verifier's: python -m ics verifier-data)")
    t.add_argument("--out", required=True, help="run directory; rerunning the same command resumes the run")
    t.add_argument("--init", metavar="CKPT",
                   help="release checkpoint whose weights the run starts from; the config is the recipe over the "
                        "defaults, but the verifier, which needs --init (SOLVER/last), keeps the checkpoint's config "
                        "with its recipe's model settings on top")
    t.add_argument("--set", type=_setting, action="append", default=[], metavar="KEY=VALUE",
                   help="override a recipe setting, e.g. train.seed=1")
    t.add_argument("--device", help="default: cuda if available (cuda:LOCAL_RANK under torchrun), else cpu")
    return t


def _train(a, parser):
    if getattr(HEADS[a.model], "needs_init", False) and a.init is None:
        parser.error(f"the {a.model} fine-tunes a checkpoint: pass --init, the solver's checkpoint")
    train(a.model, a.task, a.data, a.out, overrides=_overrides(a.set), device=a.device, init=a.init)


def _verifier_data_args(sub):
    v = sub.add_parser("verifier-data", help="build a verifier's training data from a solver's decodes of the train "
                                             "split")
    v.add_argument("--task", choices=TASKS, required=True)
    v.add_argument("--data", required=True, help="the task's built dataset (only its train split is read)")
    v.add_argument("--solver", action="append", required=True, metavar="RUN_OR_CKPT",
                   help="a solver checkpoint, or a solver's training run (see --snapshots); repeatable")
    v.add_argument("--out", required=True,
                   help="verifier dataset directory (splits train and val), missing or empty (dotfiles aside)")
    v.add_argument("--snapshots", type=_at_least(0), default=10,
                   help="per training run, the kept snapshots to decode with, spread by step (default 10; 0: last/)")
    v.add_argument("--hypotheses", type=_at_least(0), default=4,
                   help="hypothesis re-decodes per board and checkpoint (default 4)")
    v.add_argument("--boards", type=_at_least(2), help="use the train split's first BOARDS boards (default: all)")
    v.add_argument("--batch", type=_at_least(1), default=256, help="rows per decoding batch (default 256)")
    v.add_argument("--device", help="default: cuda if available, else cpu")
    return v


def _verifier_data(a, parser):
    if why := overlap(a.out, a.data, ("--out", "--data")) or _taken(Path(a.out), a.out):
        parser.error(why)
    s = build_verifier_data(a.task, a.data, a.solver, a.out, snapshots=a.snapshots, hypotheses=a.hypotheses,
                            boards=a.boards, batch=a.batch, device=_device(a.device))
    for split in ("train", "val"):
        print(f"{split}: {s[split]['candidates']} candidates, {s[split]['valid']} valid")
    print(f"wrote {a.out}")


def _pick_args(sub):
    p = sub.add_parser("pick-verifier", help="keep, among verifier runs that differ in train.seed alone, the one whose "
                                             "best/ has the highest AUC on its held-out val split")
    p.add_argument("runs", nargs="+", metavar="RUN",
                   help="a verifier's training run, e.g. one per --set train.seed=0, 1, 2; ties go to the lower seed")
    p.add_argument("--out", required=True,
                   help="checkpoint directory for the chosen run's best/, missing or empty (dotfiles aside)")
    return p


def _pick(a, parser):
    if taken := _taken(Path(a.out), a.out):
        parser.error(taken)
    picked = pick_verifier(a.runs, a.out)
    width = max(len(run) for run in ["run", *a.runs])
    print(f"{'run':{width}}  {'seed':>4}  {'step':>6}  val AUC")
    for i, (run, c) in enumerate(zip(a.runs, picked["candidates"])):
        print(f"{run:{width}}  {c['seed']:4d}  {c['step']:6d}  {c['auc']:.6f}"
              f"{'  picked' if i == picked['chosen'] else ''}")
    print(f"wrote {a.out}")


def _eval_args(sub):
    e = sub.add_parser("eval", help="evaluate one method on one task")
    e.add_argument("--method", choices=METHODS, required=True)
    e.add_argument("--task", choices=TASKS, required=True)
    e.add_argument("--ckpt", required=True,
                   help="checkpoint directory (config.json + model.safetensors) of the method's model")
    e.add_argument("--data", required=True, help="built dataset directory")
    e.add_argument("--out", required=True, help="results directory")
    e.add_argument("--verifier", metavar="CKPT",
                   help="verifier checkpoint: ics without a certificate selects with it (on maze it always does)")
    e.add_argument("--regime", choices=REGIMES, action="append", help="ics only; default: both")
    e.add_argument("--start", type=int, default=0,
                   help="first board of the test split (ptrm, gram, eqr, attractor: a multiple of batch)")
    e.add_argument("--limit", type=int, help="number of boards (default: all, or the method's configured rows)")
    e.add_argument("--batch", type=int, help="override every configured batch size (also the seeding blocks' size)")
    e.add_argument("--set", type=_setting, action="append", default=[], metavar="KEY=VALUE",
                   help="override a config setting")
    e.add_argument("--device", help="default: cuda if available, else cpu")
    e.add_argument("--dtype", choices=("bfloat16", "float32"), help="override the checkpoint's forward dtype")
    e.add_argument("--overwrite", action="store_true", help="replace existing result files of this run")
    return e


def _eval(a, parser):
    s = evaluate(a.method, a.task, a.ckpt, a.data, a.out, regimes=tuple(a.regime or REGIMES), verifier=a.verifier,
                 start=a.start, limit=a.limit, device=_device(a.device), dtype=a.dtype, batch=a.batch,
                 overrides=_overrides(a.set), overwrite=a.overwrite)
    for key, rec in s["results"].items():
        print(f"{key:28s} {rec['accuracy']:.4f} ({rec['correct']}/{rec['n']})")


# python -m ics llm: a language model on PPBench's golden boards (ics/llm). Board K's prompt is
# <task>_<K>.prompt.txt, its reply <task>_<K>.txt.
def _header(text: str) -> tuple[str, str]:
    """KEY=VALUE -> (KEY, VALUE), a header of the request."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def _llm_args(sub):
    g = sub.add_parser("llm", help="test a language model on PPBench's golden boards: prompt, run, grade")
    steps = g.add_subparsers(dest="step", required=True)
    p = steps.add_parser("prompt", help="write the prompts of a task's golden boards, for a client of your own (run "
                                        "builds its own)")
    r = steps.add_parser("run", help="ask a model for the answers to a task's golden boards")
    d = steps.add_parser("grade", help="grade the replies to a task's golden boards")
    for step in (p, r, d):
        step.add_argument("--task", choices=LLM_TASKS, required=True)
    for step in (p, r):
        step.add_argument("--board", type=int, choices=range(15), metavar="K", help="board K alone, 0 to 14")
    p.add_argument("--out", required=True, help="directory of the prompts, <task>_<K>.prompt.txt")
    r.add_argument("--model", required=True, help="the model's name at the endpoint")
    r.add_argument("--out", required=True, help="directory of the replies, <task>_<K>.txt, each saved whole with its "
                                                "token usage, <task>_<K>.usage.json; a board whose reply is there is "
                                                "kept (delete the reply to ask again), and one whose request fails is "
                                                "skipped, to be asked again by a rerun (an HTTP 400, 401, 403 or 404 "
                                                "stops the run)")
    r.add_argument("--provider", choices=("openai", "anthropic"), default="openai",
                   help="openai (default): an OpenAI-compatible /chat/completions, its key, if it needs one, in "
                        "OPENAI_API_KEY (start a vLLM server with the model's reasoning parser, so that a reply "
                        "holds the answer alone); anthropic: Anthropic's /v1/messages, its key in ANTHROPIC_API_KEY")
    r.add_argument("--base-url", metavar="URL",
                   help="the API's root: an OpenAI-compatible server's with its /v1 (default "
                        "http://localhost:8000/v1, a vLLM server's), or Anthropic's (default https://api.anthropic.com)")
    r.add_argument("--max-tokens", type=_at_least(1), default=16000, metavar="N",
                   help="the longest reply in tokens, its thinking included (default 16000)")
    r.add_argument("--set", type=_setting, action="append", default=[], metavar="KEY=VALUE",
                   help="a field of the request, e.g. temperature=0.6 (a dotted KEY nests); null drops the field, "
                        "e.g. for OpenAI's reasoning models max_tokens=null with max_completion_tokens=N")
    r.add_argument("--header", type=_header, action="append", default=[], metavar="KEY=VALUE",
                   help="a header to add to the request")
    d.add_argument("--replies", required=True, help="directory of the replies, <task>_<K>.txt (from run, or by hand)")
    for step, handle in ((p, _llm_prompt), (r, _llm_run), (d, _llm_grade)):
        step.set_defaults(handle=handle, step_parser=step)
    return g


def _llm(a, parser):
    a.handle(a, a.step_parser)


def _llm_prompt(a, parser):
    out, boards = Path(a.out), golden(a.task)
    out.mkdir(parents=True, exist_ok=True)
    for k in (range(len(boards)) if a.board is None else [a.board]):
        (out / f"{a.task}_{k}.prompt.txt").write_text(prompt(a.task, boards[k]), newline="\n")
    print(f"wrote {out}")


def _llm_run(a, parser):
    if a.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("--provider anthropic reads its key from ANTHROPIC_API_KEY, which is not set")
    out, boards, skipped = Path(a.out), golden(a.task), []
    out.mkdir(parents=True, exist_ok=True)
    for k in (range(len(boards)) if a.board is None else [a.board]):
        reply = out / f"{a.task}_{k}.txt"
        if reply.exists():
            print(f"board {k}: kept {reply}")
            continue
        try:
            text, record = query(prompt(a.task, boards[k]), a.model, a.provider, a.base_url, a.max_tokens,
                                 _overrides(a.set), dict(a.header))
        except RequestError as e:       # refused as every board's request would be: stop, in one line
            parser.exit(1, f"{parser.prog}: error: board {k}: {e}\n")
        except OSError as e:            # no connection, a 429 or 5xx after the retry, no complete reply: skip it
            skipped.append(k)
            print(f"board {k}: skipped: {e}")
            continue
        (out / f"{a.task}_{k}.usage.json").write_text(json.dumps(record, indent=1) + "\n", newline="\n")
        part = reply.with_name(reply.name + ".part")                  # renamed once whole: never a partial reply
        part.write_text(text, encoding="utf-8", newline="\n")
        part.replace(reply)
        empty = "" if text.strip() else ", an empty reply"
        print(f"board {k}: wrote {reply} (stop reason {record['stop_reason']}{empty})")
    if skipped:
        parser.exit(1, f"{parser.prog}: error: skipped board {', '.join(map(str, skipped))}; run the command again to "
                       f"ask them again\n")


def _llm_grade(a, parser):
    replies = Path(a.replies)
    if not replies.is_dir():
        parser.error(f"--replies {a.replies} is not a directory")
    verdicts = []
    for k, board in enumerate(golden(a.task)):
        path = replies / f"{a.task}_{k}.txt"
        v = (grade(a.task, board, path.read_text(encoding="utf-8", errors="replace")) if path.is_file()
             else {"format_ok": False, "valid": False, "reason": "no reply"})
        verdicts.append({"board": k, **v})
        verdict = "valid" if v["valid"] else f"{'invalid' if v['format_ok'] else 'no answer'} ({v['reason']})"
        print(f"board {k}: {verdict}")
    valid, path = sum(v["valid"] for v in verdicts), replies / f"{a.task}_grades.json"
    path.write_text(json.dumps({"task": a.task, "valid": valid, "boards": verdicts}, indent=1) + "\n", newline="\n")
    print(f"{a.task}: {valid}/{len(verdicts)} rule-valid; wrote {path}")


COMMANDS = {"data": (_data_args, _data), "download": (_download_args, _download), "train": (_train_args, _train),
            "verifier-data": (_verifier_data_args, _verifier_data), "pick-verifier": (_pick_args, _pick),
            "eval": (_eval_args, _eval), "llm": (_llm_args, _llm)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ics")
    sub = ap.add_subparsers(dest="command", required=True)
    parsers = {name: add(sub) for name, (add, _) in COMMANDS.items()}
    a = ap.parse_args(argv)
    COMMANDS[a.command][1](a, parsers[a.command])
