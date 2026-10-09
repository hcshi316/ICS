# Adapted from github.com/SamsungSAILMontreal/TinyRecursiveModels@c0110373 (pretrain.py and models/ema.py; MIT
# License: see ics/trm/layers.py). Modified.
"""Training: one loop for every model, through its head (ics/registry.py, HEADS). What the loop needs of a head:
  - config_class, its model's config, and model, the one module the loop trains, averages (EMA) and saves; on every
    rank its backward reaches the same parameters, whose gradients the loop sums over the ranks;
  - initial_carry(batch) and forward(carry, batch) -> (carry, loss summed over the rows, statistics name -> (sum,
    count)), run `steps` times per micro-batch; the carry persists across updates;
  - optimizers(**optim) -> [(optimizer, base lr)], from the recipe's `optim` section;
  - evaluate(model, pool, task, batch) -> a value per board of the weights it is given, and metric(pool, values) -> a
    finite number, logged as metric_name;
  - needs_init: it fine-tunes a checkpoint (the verifier); metric_held_out: its metric is scored on boards held out
    from its training data, so the run also keeps its best evaluation.
Recipe: ics/configs/train/<model>.yaml lists what differs from the defaults of the model's config and of TrainConfig;
a task's entry under `tasks:` overrides it, and --set overrides both. The loop sets seq_len, vocab_size and
num_puzzle_identifiers from the train split, and batch_size to the micro-batch. With --init the model starts from a
release checkpoint's weights; the verifier also keeps its config, under the recipe's model settings.
Updates: each global batch is cut into micro-batches over the ranks (train.accumulate per rank, or train.micro_batch
rows each); each step's loss, times 1 / global_batch, is backpropagated; the gradients are summed over the ranks and
clipped to train.clip, every optimizer steps at lr(s) of its base lr, and the EMA follows:
    lr(s) = base * s / warmup                                                                     while s < warmup,
    lr(s) = base * (min_ratio + (1 - min_ratio) * (1 + cos(pi * (s - warmup) / (total - warmup))) / 2)   after.
Run directory, written by rank 0 at every evaluation (every eval_every updates and after the last): last/, the release
checkpoint (ics/checkpoint.py) of the evaluated weights; best/, the same at the best evaluation, for a held-out metric
only; snapshots/step_<update>/, every evaluation's, with train.keep_every_eval; state.pt, the state a rerun of the
same command resumes from (exactly on CPU, and on GPU with train.deterministic); log.jsonl, the lr and statistics
every log_every updates and every evaluation's metric.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ics.checkpoint import WEIGHTS_FILE, model_section, save_checkpoint, sha256
from ics.config import check_settings, load_yaml, merge_config
from ics.data import Pool, check_dims, dataset_dims, load_pool, load_train, train_batches
from ics.registry import HEADS

CONFIG_DIR = Path(__file__).parent / "configs" / "train"


@dataclass
class TrainConfig:
    global_batch: int
    epochs: int | None = None           # passes over the train split's groups; or
    updates: int | None = None          # the number of updates (exactly one of the two)
    accumulate: int = 1                 # micro-batches per rank and update (gradient accumulation); or
    micro_batch: int | None = None      # the rows of a micro-batch, fixed: accumulate then follows the ranks
    epochs_per_block: int = 5000        # epochs per block of the training order (ics/data.py)
    lr_warmup: int = 2000
    lr_min_ratio: float = 1.0
    ema: float = 0.999                  # EMA decay in [0, 1); 0: no EMA, the live weights are evaluated and exported
    clip: float | None = None           # the gradients' global norm is clipped to this (> 0); None: no clipping
    eval_every: int = 500               # updates between evaluations
    eval_boards: int | None = None      # boards of the eval split to evaluate; None: all
    keep_every_eval: bool = False       # also keep every evaluation's release checkpoint (snapshots/step_<update>/)
    log_every: int = 100                # updates between log lines
    compile: bool = False               # torch.compile the training step
    deterministic: bool = False         # PyTorch's deterministic mode for the rest of the process: a run repeats
    seed: int = 0

    def __post_init__(self):
        for name in ("eval_every", "log_every", "epochs_per_block", "eval_boards"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"train.{name} must be at least 1, got {value}")
        if self.clip is not None and self.clip <= 0:
            raise ValueError(f"train.clip must be positive (null: no clipping), got {self.clip}")
        if not 0 <= self.ema < 1:
            raise ValueError(f"train.ema must be in [0, 1) (0: no EMA), got {self.ema}")
        if (self.epochs is None) == (self.updates is None):
            raise ValueError(f"set exactly one of train.epochs and train.updates, got {self.epochs} and {self.updates}")
        if self.micro_batch is not None and self.accumulate != 1:
            raise ValueError(f"set train.accumulate or train.micro_batch, not both (got {self.accumulate} and "
                             f"{self.micro_batch})")


DATA_SET = ("seq_len", "vocab_size", "num_puzzle_identifiers", "batch_size")    # the loop sets these from the data


def recipe(model: str, task: str, overrides: dict | None = None) -> dict:
    """The model's recipe for the task, ics/configs/train/<model>.yaml: its shared settings, the task's entry under
    `tasks:` on top, then `overrides`. A recipe lists only what differs from the defaults of TrainConfig and of the
    head's config class. An override must name a setting: a field of the config class (but those the data sets), an
    optim argument of the recipe, or a TrainConfig field."""
    raw = load_yaml(CONFIG_DIR / f"{model}.yaml")
    tasks = raw.pop("tasks", None) or {}
    cfg = merge_config({"model": {}, "train": {}}, merge_config(raw, tasks.get(task)))
    # the names an override may set; None marks a single value (check_settings)
    known = {"model": dict.fromkeys(f.name for f in fields(HEADS[model].config_class) if f.name not in DATA_SET),
             "optim": dict.fromkeys(cfg["optim"]), "train": dict.fromkeys(f.name for f in fields(TrainConfig))}
    check_settings(known, overrides or {})
    return merge_config(cfg, overrides)


def lr_at(step: int, base: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


class EMA:
    """shadow <- (1 - decay) * parameter + decay * shadow after every update; decay 0: copy() gives the live weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad and decay > 0}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = (1.0 - self.decay) * p + self.decay * self.shadow[n]

    @torch.no_grad()
    def copy(self, model: nn.Module) -> nn.Module:
        """A copy of `model` with the averaged parameters (and the model's buffers)."""
        out = copy.deepcopy(model)
        for n, p in out.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])
        return out


def train(model: str, task: str, data, out, overrides: dict | None = None, device=None, init=None) -> dict:
    """Train `model` on `task` (module docstring); returns the last step, its metric and the best evaluation."""
    if model not in HEADS:
        raise ValueError(f"unknown model {model!r}; expected one of {tuple(HEADS)}")
    head_class, out = HEADS[model], Path(out)
    if getattr(head_class, "needs_init", False) and init is None:
        raise ValueError(f"the {model} fine-tunes a checkpoint: pass init (--init), the solver's checkpoint")
    cfg = recipe(model, task, overrides)
    tc = TrainConfig(**cfg["train"])
    if tc.deterministic:                # before the run touches CUDA: cuBLAS reads the variable when it starts
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    rank, world, device = _distributed(device)
    accumulate = _accumulate(tc, world)
    micro, split = tc.global_batch // (world * accumulate), load_train(data)
    total = _total(tc, split)
    head = _build(head_class, cfg["model"], data, init, batch_size=micro, seed=tc.seed + rank, device=device,
                  world=world)
    optimizers, ema = head.optimizers(**cfg["optim"]), EMA(head.model, tc.ema)
    eval_split, pool = _eval_pool(data, tc.eval_boards)
    digests = {k: sha256(Path(data) / "train" / f"all__{k}.npy") for k in ("inputs", "labels")}
    init_sha256 = None if init is None else sha256(Path(init) / WEIGHTS_FILE)
    run = {"model": model, "task": task, "world": world, "recipe": cfg, "train_sha256": digests,
           "init_sha256": init_sha256}
    provenance = {"format": "train", "model": model, "task": task, "ema": tc.ema > 0, "eval_split": eval_split,
                  "eval_boards": len(pool), "world": world, "train_sha256": digests, "init_sha256": init_sha256,
                  "torch": torch.__version__, "deterministic": torch.are_deterministic_algorithms_enabled(),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "recipe": {"optim": cfg["optim"], "train": asdict(tc)}}
    metric, held_out = head.metric_name, getattr(head, "metric_held_out", False)
    step, position, best, carries, score = 0, (1, 0), {"step": 0, metric: -math.inf}, [None] * accumulate, None
    if (out / "state.pt").exists():
        step, position, best, carries, score = _resume(out / "state.pt", run, head, optimizers, ema, rank, device)
    with torch.no_grad():                           # a task the data cannot serve fails here, before any update
        head.evaluate(head.model.eval(), pool.take(slice(0, 0)), task, micro)
    head.train()
    if rank == 0:
        _restart_log(out, step)
        if step >= total:
            print(f"{out}: already complete at step {step}; best {best}", flush=True)
    step_fn = torch.compile(head) if tc.compile else head
    batches = train_batches(split.group_starts, tc.seed, tc.epochs_per_block, tc.global_batch, position)
    window, t0 = {}, time.time()
    while step < total:
        block, index, rows = next(batches)
        for a in range(accumulate):
            v = rank * accumulate + a                                       # the virtual rank
            carries[a] = _micro_batch(head, step_fn, split.rows(rows[v * micro:(v + 1) * micro]), carries[a],
                                      device=device, global_batch=tc.global_batch, stats=window)
        step += 1
        lr = _update(head, optimizers, step, total, tc, world)
        ema.update(head.model)
        if step % tc.log_every == 0 or step == total:
            means = {k: round(float(s) / max(float(c), 1.0), 6) for k, (s, c) in window.items()}
            _log(out, rank, {"step": step, "lr": lr, **means, "seconds": round(time.time() - t0, 1)})
            window = {}
        if step % tc.eval_every == 0 or step == total:
            evaluated = ema.copy(head.model).eval()
            score = _evaluate(head, evaluated, pool, task, rank, world, micro)
            improved = score > best[metric]
            if improved:
                best = {"step": step, metric: score}
            ranks = _gather(_rank_state(carries, device), rank, world)
            if rank == 0:
                state = {"run": run, "step": step, metric: score, "position": [block, index + 1], "best": best,
                         "model": head.model.state_dict(), "optimizers": [o.state_dict() for o, _ in optimizers],
                         "ema": ema.shadow, "ranks": ranks}
                _save(out, state, evaluated, {**provenance, "step": step, metric: score}, improved and held_out,
                      tc.keep_every_eval)
            _log(out, rank, {"step": step, metric: round(score, 6), "best": best})
    if world > 1:
        dist.destroy_process_group()
    return {"step": step, metric: score, "best": best}


def _accumulate(tc: TrainConfig, world: int) -> int:
    """Micro-batches per rank and update: train.accumulate, or as many as train.micro_batch rows make."""
    accumulate = tc.accumulate
    if tc.micro_batch is not None:              # the recipe fixes the rows of a micro-batch, whatever the ranks
        if tc.micro_batch < 1 or tc.global_batch % (world * tc.micro_batch):
            raise ValueError(f"global_batch {tc.global_batch} is not a multiple of ranks x micro_batch = {world} x "
                             f"{tc.micro_batch}")
        accumulate = tc.global_batch // (world * tc.micro_batch)
    if accumulate < 1 or tc.global_batch % (world * accumulate):
        raise ValueError(f"global_batch {tc.global_batch} is not a multiple of ranks x accumulate = {world} x "
                         f"{accumulate}")
    return accumulate


def _total(tc: TrainConfig, split) -> int:
    """The run's updates: train.updates, or int(epochs * total_groups * mean_puzzle_examples / global_batch)."""
    if tc.updates is not None:
        length, total = f"train.updates {tc.updates}", tc.updates
    else:
        length = f"train.epochs {tc.epochs}"
        total = int(tc.epochs * split.meta["total_groups"] * split.meta["mean_puzzle_examples"] / tc.global_batch)
    if total < 1:
        raise ValueError(f"{length} gives {total} updates of global_batch {tc.global_batch}; a run needs at least 1")
    return total


def _build(head_class, settings: dict, data, init, *, batch_size: int, seed: int, device, world: int):
    """The head, its weights drawn after torch.manual_seed(seed) or the init's, then rank 0's on every rank; its config
    is the recipe's model settings over the class's defaults (needs_init: over the init's), then the data's sizes."""
    dims = dataset_dims(data, "train")
    if not getattr(head_class, "needs_init", False):            # an init gives its weights, the recipe the rest
        settings = {**{f.name: f.default for f in fields(head_class.config_class)
                       if f.default is not MISSING and f.name not in DATA_SET}, **settings}
    if init is not None:
        saved = model_section(init, head_class.config_class)
        check_dims(init, saved, dims)
        settings = {**saved, **settings}
    declared = {f.name for f in fields(head_class.config_class)}
    given = {k: v for k, v in {**dims, "batch_size": batch_size}.items() if k in declared}
    config = head_class.config_class(**{**settings, **given})
    torch.manual_seed(seed)
    with torch.device(device):
        head = head_class(config)
    if init is not None:
        from safetensors.torch import load_file

        head.model.load_state_dict(load_file(str(Path(init) / WEIGHTS_FILE)), strict=True)
    if world > 1:
        with torch.no_grad():
            for t in [*head.model.parameters(), *head.model.buffers()]:
                dist.broadcast(t, 0)
    return head


def _eval_pool(data, eval_boards: int | None) -> tuple[str, Pool]:
    """The val split if the data has one, else test, and its pool: eval_boards boards spread over it, or all."""
    eval_split = "val" if (Path(data) / "val").is_dir() else "test"
    pool = load_pool(data, eval_split)
    if eval_boards is not None:
        n = min(eval_boards, len(pool))
        pool = pool.take(np.arange(n, dtype=np.int64) * len(pool) // n)
    return eval_split, pool


def _micro_batch(head, step_fn, part: dict, carry, *, device, global_batch: int, stats: dict):
    """The head's steps on a micro-batch, each step's loss backpropagated, its statistics added into `stats`."""
    batch = {k: torch.from_numpy(x).to(device) for k, x in part.items()}
    if carry is None:
        carry = head.initial_carry(batch)
    for _ in range(head.steps):
        carry, loss, step_stats = step_fn(carry, batch)
        ((1 / global_batch) * loss).backward()
        for k, (s, c) in step_stats.items():
            s0, c0 = stats.get(k, (0, 0))
            stats[k] = (s0 + s, c0 + c)
    return carry


def _update(head, optimizers, step: int, total: int, tc: TrainConfig, world: int) -> float:
    """Update `step`: the gradients summed over the ranks and clipped, every optimizer's step; returns the lr."""
    if world > 1:
        for p in head.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
    if tc.clip is not None:
        torch.nn.utils.clip_grad_norm_(head.model.parameters(), tc.clip)
    for opt, base in optimizers:
        lr = lr_at(step, base, tc.lr_warmup, total, tc.lr_min_ratio)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad()
    return lr


def _rank_state(carries: list, device) -> dict:
    """This rank's part of state.pt: its virtual ranks' carries and its generators' states."""
    return {"carries": [{k: x.cpu() for k, x in c.items()} for c in carries],
            "rng": {"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}}


def _distributed(device) -> tuple[int, int, torch.device]:
    """(rank, world size, device). Under torchrun, every rank joins the process group (nccl on GPUs, else gloo)."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return 0, 1, torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    local = int(os.environ["LOCAL_RANK"])
    gpu = torch.cuda.is_available() and torch.device(device or "cuda").type != "cpu"
    if gpu:
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if gpu else "gloo")
    return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local}" if gpu else "cpu")


@torch.no_grad()
def _evaluate(head, model, pool, task: str, rank: int, world: int, batch: int) -> float:
    """The head's metric of `model` on the eval pool, each rank scoring its contiguous share."""
    values = head.evaluate(model, pool.take(np.array_split(np.arange(len(pool)), world)[rank]), task, batch)
    if world > 1:
        parts = [None] * world
        dist.all_gather_object(parts, values)
        values = np.concatenate(parts)
    return head.metric(pool, values)


def _log(out: Path, rank: int, record: dict) -> None:
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def _restart_log(out: Path, step: int) -> None:
    """Keep the complete records of out/log.jsonl up to `step`, the step the run starts from (none on a fresh start)."""
    path = out / "log.jsonl"
    if path.exists():
        records = path.read_text().splitlines(keepends=True)
        path.write_text("".join(r for r in records if r.endswith("\n") and json.loads(r)["step"] <= step))


def _gather(obj, rank: int, world: int) -> list | None:
    """Every rank's `obj`, in rank order, on rank 0 (None on the other ranks)."""
    if world == 1:
        return [obj]
    out = [None] * world if rank == 0 else None
    dist.gather_object(obj, out, dst=0)
    return out


def _save(out: Path, state: dict, evaluated: nn.Module, provenance: dict, best: bool, keep: bool) -> None:
    """last/, best/ if `best` (an improved held-out metric), the snapshot if `keep`, then state.pt."""
    weights = evaluated.state_dict()
    save_checkpoint(weights, evaluated.config, out / "last", provenance)
    if best:
        save_checkpoint(weights, evaluated.config, out / "best", provenance)
    if keep:
        save_checkpoint(weights, evaluated.config, out / "snapshots" / f"step_{state['step']}", provenance)
    torch.save(state, out / "state.pt.tmp")
    os.replace(out / "state.pt.tmp", out / "state.pt")


def _resume(path: Path, run: dict, head, optimizers, ema: EMA, rank: int, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["run"] != run:
        raise ValueError(f"{path} belongs to another run; resuming needs the same model, task, recipe, train split, "
                         f"init and number of ranks (saved: {state['run']})")
    head.model.load_state_dict(state["model"])
    for (opt, _), saved in zip(optimizers, state["optimizers"]):
        opt.load_state_dict(saved)
    ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
    mine = state["ranks"][rank]
    torch.set_rng_state(mine["rng"]["cpu"])
    if mine["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(mine["rng"]["cuda"], device)
    carries = [{k: x.to(device) for k, x in c.items()} for c in mine["carries"]]
    return state["step"], tuple(state["position"]), state["best"], carries, state[head.metric_name]
