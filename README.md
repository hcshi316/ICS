# ICS: Input-Conditioned Search

A recursive reasoner, such as the [Tiny Recursive Model (TRM)](https://arxiv.org/abs/2510.04871), solves a problem by
iterating a learned update map on a latent state, so its reasoning is a dynamical system. Test-time compute can run the
map longer (depth) or perturb the state (width), but a trajectory that settles into an incorrect attractor can stay
trapped under both. Input-Conditioned Search (ICS) makes the update map itself an axis of test-time scaling: the map is
conditioned on the input, so ICS changes the input, by writing the model's own tentative predictions into it or by
restating the task equivalently, and restarts reasoning under each new map, with the weights frozen.

ICS is one simple way to shape reasoning dynamics, and it leaves open questions this code is meant to help explore:
Can we train reasoners whose dynamics are easier to steer toward correct solutions, or more diverse? Can test-time
search find good dynamics with less compute, by changing the input, the weights, or both?

This repository holds ICS, TRM, the baselines of the paper's main table (PTRM, GRAM, EqR, Attractor) with their
training, and the builders of the six puzzle datasets. Project page: https://icsearch.github.io

## Install

```
pip install -e .                  # Python 3.10 or newer, PyTorch 2.4 or newer
pip install -e ".[ppbench]"       # to build Light-Up, Nurikabe, Tapa and Heyawake (also needs Node.js 16 or newer)
pip install -e ".[dev]"           # to run the tests (pytest)
```

## Data

```
python -m ics data --task sudoku --out data/sudoku       # tasks: sudoku maze lightup nurikabe tapa heyawake
python -m ics download --dataset --task tapa             # or download one built: lightup nurikabe tapa heyawake
```

Sudoku and Maze come from Hugging Face at pinned revisions; the other four are built from Pencil Puzzle Bench's
puzzles, without any train board that poses a test board's puzzle. A dataset is a directory in TRM's layout
(`ics/data.py`). `download --dataset` fetches a built Light-Up, Nurikabe, Tapa or Heyawake dataset from Hugging Face
(`hcshi/ICS-data`) into `data/<task>`, with no need for ppbench or Node.js.

## Checkpoints

```
python -m ics download --model trm --task maze --seed 0  # models: trm eqr gram attractor verifier
```

The released checkpoints are on Hugging Face (`hcshi/ICS`, whose card lists them); `download` writes one into
`ckpt/<model>/<task>/seed<k>` (`--out` to choose another directory, `--repo` to read another repo of the same layout).
A training run's checkpoint is `RUN/last`, what `--ckpt` takes; a verifier run also keeps `RUN/best`, its best evaluation
on boards held out from its training data, and `pick-verifier` copies the best seed's into its `--out`.

## Evaluate

```
python -m ics eval --method trm  --task sudoku --ckpt CKPT --data data/sudoku --out results  # TRM, + greedy depth, + depth scaling
python -m ics eval --method ptrm --task sudoku --ckpt CKPT --data data/sudoku --out results
python -m ics eval --method ics  --task sudoku --ckpt CKPT --data data/sudoku --out results  # with and without a certificate
python -m ics eval --method ics  --task maze   --ckpt CKPT --data data/maze --out results --verifier VERIFIER
python -m ics eval --method gram --task sudoku --ckpt GRAM_CKPT --data data/sudoku --out results     # also eqr, attractor
```

A method's protocol is `ics/configs/eval/<method>.yaml`, a task's differences under `tasks:`. `--set KEY=VALUE`
changes a setting, and `--start`/`--limit` evaluate a slice of the test split. A run writes `<method>-<task>.npz`,
every board's answers under the keys of `ics/methods/__init__.py`, and a `.json` of the accuracies (raw and pinned,
with and without a certificate).

## Train

```
python -m ics train --model trm --task sudoku --data data/sudoku --out runs/trm-sudoku    # models: trm eqr gram attractor
torchrun --nproc-per-node 8 -m ics train --model trm ...                                   # one process per GPU
```

A recipe, `ics/configs/train/<model>.yaml`, lists what it sets on top of the defaults of `TrainConfig` (`ics/train.py`)
and of the model's config. `--set train.seed=1` changes one setting; rerunning a command resumes its run.

The verifier, which selects for ICS without a certificate (on Maze always), is a TRM fine-tuned from the solver on its
own decodes:

```
python -m ics train --model trm --task maze --data data/maze --out SOLVER --set train.keep_every_eval=true
python -m ics verifier-data --task maze --data data/maze --solver SOLVER --out VDATA
for s in 0 1 2; do
  python -m ics train --model verifier --task maze --data VDATA --init SOLVER/last --out V$s --set train.seed=$s
done
python -m ics pick-verifier V0 V1 V2 --out VERIFIER
```

## Code map

```
ics/registry.py    every task, model and method by name: a new task or baseline is registered here
ics/tasks/         a puzzle type's rules, pinning, hypotheses and restatements (base.py: the Task interface)
ics/builders/      the datasets (python -m ics data)
ics/trm/           the TRM: model, rolls, training head
ics/methods/       standard TRM and depth (trm.py), PTRM (ptrm.py), ICS (ics.py)
ics/select.py      the block loop of the sampling methods; ics/seeding.py: their per-block seeds (a new one takes an id)
ics/verifier/      the verifier: its data, training, pick, and the selection it makes
ics/train.py       the training loop of every model         ics/evaluate.py   the evaluation of every method
ics/configs/       train/<model>.yaml recipes, eval/<method>.yaml protocols
ics_baselines/     GRAM, EqR, Attractor: model.py, predict.py (protocol), train.py (training head)
tests/             pytest; ICS_DATA_ROOT=data also runs the data tests on the datasets built there
```

## Citation

The BibTeX entry will appear here once the paper is public.

## License

MIT (`LICENSE`), except the adapted files listed in `THIRD_PARTY_NOTICES.md`, which keep their licences.
