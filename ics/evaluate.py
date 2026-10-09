"""Evaluate one method on one task: per-board answers plus a summary of accuracies.

Settings: ics/configs/eval/<method>.yaml, a task's entry under `tasks:` on top, then the overrides (--set), which may
change only a setting the method has on the task. ics runs the requested regimes (cert, model); every other method
writes both. Without a certificate, ICS on Maze, and on another task given a verifier (--verifier), selects with the
verifier among the decodes of each board's restatements (ics/verifier/select.py) instead of searching. gram, eqr and
attractor run their protocols (ics_baselines/*/predict.py). A checkpoint of another model, or one built for another
dataset, is refused before any board runs.
Outputs: <out>/<stem>.npz holds the answers and per-board extras under the keys of ics/methods/__init__.py ("/" written
as "__") and `index`, the boards' rows in the test split; <out>/<stem>.json the accuracy of every key, the settings and
the provenance. The stem is <method>-<task>, then .<regime> for a single ICS regime, then .rows<start>-<stop> for a
shard (start > 0 or a limit), e.g. ics-maze.cert.rows0-64; an existing result is replaced only with --overwrite.
Sharding: ptrm, gram, eqr and attractor work in blocks of `batch` boards, so a shard must start on a multiple of batch;
one that also ends on one, or at the end of the split, reproduces the full run's answers on its boards. A config's
`rows` (Attractor on Sudoku: 3072) limits a run to the split's first rows boards."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ics.checkpoint import WEIGHTS_FILE, sha256
from ics.config import check_settings, load_yaml, merge_config
from ics.data import Pool, check_dims, load_pool
from ics.methods.ics import ICSConfig, run_ics
from ics.registry import METHODS
from ics.tasks import make_task
from ics.tasks.base import SCORINGS
from ics.trm.model import load_trm
from ics.verifier.select import run_ics_verifier

CONFIG_DIR = Path(__file__).parent / "configs" / "eval"
REGIMES = ("cert", "model")


def eval_config(method: str, task: str) -> dict:
    raw = load_yaml(CONFIG_DIR / f"{method}.yaml")
    tasks = raw.pop("tasks", None) or {}
    return merge_config(raw, tasks.get(task))


def summarize(task, answers: dict) -> dict:
    """Accuracy of every answer array ([N, L] under a "<row>/<regime>/<scoring>" key) in its own scoring."""
    idx = np.arange(len(task))
    golden = task.pool.is_golden
    out = {}
    for key, ans in answers.items():
        parts = key.split("/")
        if len(parts) != 3 or parts[2] not in SCORINGS or np.ndim(ans) != 2:
            continue
        ok = task.check(parts[2], idx, ans)
        rec = {"accuracy": round(float(ok.mean()), 6), "correct": int(ok.sum()), "n": int(ok.size)}
        if golden is not None:
            rec["golden_correct"] = int(ok[golden == 1].sum())
        out[key] = rec
    return out


def _regime_settings(cfg: dict, regime: str) -> dict:
    """An ICS regime's settings: the shared batch, window, seed and restatements, overridden by the regime's own. A
    `selector`, if present, must be "verifier"."""
    rc = merge_config({k: cfg[k] for k in ("batch", "window", "seed", "restatements") if k in cfg}, cfg[regime])
    if "selector" in rc and rc["selector"] != "verifier":
        raise ValueError(f"unknown selector {rc['selector']!r}")
    return rc


def _ics(model, verifier, task_name: str, pool: Pool, plans: dict) -> dict:
    """Each regime of `plans` (regime -> its settings), window by window: the verifier's selection among the
    restatements' decodes if the regime's selector says so, else the search."""
    answers = {}
    for regime, rc in plans.items():
        window = rc.get("window") or len(pool)
        parts = []
        for a in range(0, len(pool), window):
            task = make_task(task_name, pool.take(slice(a, a + window)), restatements=rc["restatements"],
                             seed=rc["seed"], block=rc["batch"])
            if rc.get("selector") == "verifier":
                parts.append(run_ics_verifier(model, verifier, task, T=rc["T"], verifier_T=rc["verifier_T"],
                                              batch=rc["batch"]))
            else:
                parts.append(run_ics(model, task, ICSConfig.from_settings(rc), regime))
        answers.update({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    return answers


def _set_batch(cfg: dict, batch: int) -> dict:
    return {k: (_set_batch(v, batch) if isinstance(v, dict) else batch if k == "batch" else v) for k, v in cfg.items()}


def _for_dataset(model, ckpt, meta: dict):
    """The model loaded from `ckpt`, once check_dims finds it built for the dataset (meta: its dataset.json)."""
    check_dims(ckpt, asdict(model.config), meta)
    return model


def _regimes(method: str, regimes) -> tuple[str, ...]:
    """The requested regimes, each once. Every method but ics has no regime choice: it always writes both."""
    out = tuple(dict.fromkeys(regimes))
    if not out:
        raise ValueError(f"no regime given; expected some of {REGIMES}")
    for regime in out:
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    if method != "ics" and set(out) != set(REGIMES):
        raise ValueError(f"{method} has no regime choice: it always writes both regimes {REGIMES}, got {out}")
    return out


def evaluate(method: str, task_name: str, ckpt, data, out_dir, *, regimes=REGIMES, verifier=None, start: int = 0,
             limit: int | None = None, device="cpu", dtype: str | None = None, batch: int | None = None,
             overrides: dict | None = None, overwrite: bool = False) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {tuple(METHODS)}")
    spec = METHODS[method]
    regimes = _regimes(method, regimes)
    cfg = eval_config(method, task_name)
    check_settings(cfg, overrides or {})
    cfg = merge_config(cfg, overrides)
    if batch is not None:
        cfg = _set_batch(cfg, batch)
    shard = start > 0 or limit is not None
    if cfg.get("rows") is not None:
        if start >= cfg["rows"]:
            raise ValueError(f"{method} on {task_name} evaluates the first {cfg['rows']} test boards; "
                             f"got start={start}")
        limit = cfg["rows"] - start if limit is None else min(limit, cfg["rows"] - start)
    if spec.blocked and (cfg["batch"] < 1 or start % cfg["batch"]):
        raise ValueError(f"a {method} run needs batch >= 1 and a start on a multiple of batch, where its blocks "
                         f"start; got start={start}, batch={cfg['batch']}")
    if method == "ics" and verifier is not None and "model" in regimes:
        cfg = merge_config(cfg, {"model": {"selector": "verifier"}})  # a given verifier selects without a certificate
    plans = {r: _regime_settings(cfg, r) for r in regimes} if method == "ics" else {}
    uses_verifier = any(rc.get("selector") == "verifier" for rc in plans.values())
    if uses_verifier and verifier is None:
        raise ValueError(f"ics on {task_name} without a certificate needs a verifier checkpoint (--verifier)")
    pool = load_pool(data, "test", start, limit)
    if len(pool) == 0:
        raise ValueError(f"no test boards in {data} at start={start}, limit={limit}")
    stem = f"{method}-{task_name}"
    if method == "ics" and len(regimes) == 1:
        stem += f".{regimes[0]}"
    if shard:
        stem += f".rows{start}-{start + len(pool)}"
    out = Path(out_dir)
    if not overwrite and any((out / f"{stem}.{ext}").exists() for ext in ("npz", "json")):
        raise FileExistsError(f"{out / stem}.npz or .json exists; pass overwrite=True (--overwrite) to replace it")
    model = _for_dataset(spec.load(ckpt, device, dtype), ckpt, pool.meta)
    verifier_model = _for_dataset(load_trm(verifier, device, dtype), verifier, pool.meta) if uses_verifier else None
    weights = {"ckpt_sha256": sha256(Path(ckpt) / WEIGHTS_FILE),
               "verifier_sha256": sha256(Path(verifier) / WEIGHTS_FILE) if uses_verifier else None}
    t0 = time.time()
    if method == "ics":
        answers = _ics(model, verifier_model, task_name, pool, plans)
    else:
        answers = spec.run(model, make_task(task_name, pool), **{k: v for k, v in cfg.items() if k != "rows"})
    gpu = torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None
    summary = {"method": method, "task": task_name, "ckpt": str(ckpt),
               "verifier": str(verifier) if uses_verifier else None, **weights,
               "data": str(data), "start": start, "n": len(pool), "device": str(device), "gpu": gpu,
               "dtype": model.config.forward_dtype, "config": cfg, "seconds": round(time.time() - t0, 1),
               "torch": torch.__version__, "results": summarize(make_task(task_name, pool), answers)}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}.npz", index=pool.index, **{k.replace("/", "__"): v for k, v in answers.items()})
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    return summary
