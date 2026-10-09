"""Seeds (pick_verifier; python -m ics pick-verifier RUN [RUN ...] --out DIR). Among runs that differ in train.seed
alone, the one whose best/ has the highest val AUC, the lower seed on a tie; its best/ is written to DIR, with every
candidate's AUC and the choice in its provenance ("picked")."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ics.checkpoint import CONFIG_FILE, WEIGHTS_FILE, sha256
from ics.trm.model import load_trm


def _flat(d: dict, prefix: str = "") -> dict:
    """A nested dict's leaves under dotted keys: {"train": {"seed": 0}} -> {"train.seed": 0}."""
    out = {}
    for key, value in d.items():
        out.update(_flat(value, f"{prefix}{key}.") if isinstance(value, dict) else {f"{prefix}{key}": value})
    return out


def pick_verifier(runs, out) -> dict:
    """The verifier among training runs (python -m ics train --model verifier) that differ in train.seed alone: the
    run whose best/ has the highest AUC on its data's val split; on a tie the lower seed, then the earlier run. Runs
    that differ in anything else (task, train split, init, eval pool, recipe, model but batch_size) are refused before
    anything is written. The chosen best/ is written to `out`, its provenance with "picked": the candidates (directory
    name, seed, step, AUC, the weights' sha256) and the index of the one chosen. Returns that record."""
    paths, metas = [Path(run) for run in runs], []
    if not paths:
        raise ValueError("pick a verifier among one run or more")
    for i, path in enumerate(paths):
        if path.resolve() in {p.resolve() for p in paths[:i]}:
            raise ValueError(f"{path} is given twice")
        config = path / "best" / CONFIG_FILE
        meta = json.loads(config.read_text()) if config.is_file() else {}
        if meta.get("provenance", {}).get("model") != "verifier":
            raise ValueError(f"{path}: no best/ checkpoint of a verifier's training run (python -m ics train --model "
                             f"verifier)")
        if (split := meta["provenance"].get("eval_split")) != "val":
            raise ValueError(f"{path}: its best/ was chosen on the {split} split; pick-verifier picks by the AUC on "
                             f"val, boards held out from the train split")
        metas.append(meta)

    def shared(meta: dict) -> dict:                 # what the runs must have in common
        prov = meta["provenance"]
        train = {k: v for k, v in prov["recipe"]["train"].items() if k != "seed"}
        return _flat({**{k: prov[k] for k in ("task", "train_sha256", "init_sha256", "eval_split", "eval_boards")},
                      "optim": prov["recipe"]["optim"], "train": train,
                      "model": {k: v for k, v in meta["model"].items() if k != "batch_size"}})

    first = shared(metas[0])
    for path, meta in zip(paths[1:], metas[1:]):
        other = shared(meta)
        if differ := sorted(k for k in first.keys() | other.keys() if first.get(k) != other.get(k)):
            raise ValueError(f"{path} and {paths[0]} differ in {', '.join(differ)}: pick among runs that differ in "
                             f"train.seed alone")
    candidates = [{"run": path.resolve().name, "seed": meta["provenance"]["recipe"]["train"]["seed"],
                   "step": meta["provenance"]["step"], "auc": meta["provenance"]["auc"],
                   "sha256": sha256(path / "best" / WEIGHTS_FILE)} for path, meta in zip(paths, metas)]
    chosen = min(range(len(paths)), key=lambda i: (-candidates[i]["auc"], candidates[i]["seed"], i))
    picked = {"candidates": candidates, "chosen": chosen}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths[chosen] / "best" / WEIGHTS_FILE, out / WEIGHTS_FILE)
    meta = metas[chosen]
    (out / CONFIG_FILE).write_text(json.dumps({"model": meta["model"],
                                               "provenance": {**meta["provenance"], "picked": picked}}, indent=1))
    load_trm(out)
    return picked
