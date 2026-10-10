# Third-party notices

Our code is under the MIT License (`LICENSE`). The files adapted from the projects below keep their licences; each
names its source at its top and says that it was modified (the data files of `ics/llm/` do so at the top of its
`__init__.py`).

| Component | Source | Licence | Adapted in | Licence text |
|---|---|---|---|---|
| Tiny Recursive Models (TRM) | github.com/SamsungSAILMontreal/TinyRecursiveModels, commit `c0110373` | MIT, © 2025 Samsung Electronics Co., Ltd. | `ics/trm/{layers,model,train}.py`, `ics/train.py`, `ics/optim.py`, `ics/data.py`, `ics/builders/{__init__,sudoku,maze}.py` | the docstring of `ics/trm/layers.py` |
| Hierarchical Reasoning Model (HRM) | github.com/sapientinc/HRM | Apache License 2.0 | `ics/builders/{__init__,sudoku,maze}.py`, through TRM's copies | `LICENSES/Apache-2.0.txt` |
| EqR | github.com/locuslab/EqR, commit `aba94e9` | Apache License 2.0 | `ics_baselines/eqr/` | `ics_baselines/eqr/LICENSE` |
| Attractor | github.com/jacobfa/Attractor, commit `fcf045f9` | MIT, © 2026 Jacob Fein-Ashley, Paria Rashidinejad | `ics_baselines/attractor/model.py`, `train.py` | `ics_baselines/attractor/LICENSE` |
| Pencil Puzzle Bench (PPBench) | github.com/approximatelabs/pencil-puzzle-bench (`ppbench` 0.1.0); huggingface.co/datasets/bluecoconut/pencil-puzzle-bench, revision `3ac6add` | MIT, © 2026 Justin Waugh / Approximate Labs | `ics/builders/ppb.py` (`solution_moves`); `ics/llm/golden_boards.json` (the golden boards' givens) | `LICENSES/MIT-PPBench.txt` |
| pzpr.js, the puzzle engine of puzz.link | github.com/robx/pzprjs | MIT, © 2011, 2014 Kobayashi, Daisuke (sabo2); © 2019 Robert Vollmert and contributors | `ics/llm/{lightup,nurikabe,tapa,heyawake}.txt` (the rules) | `LICENSES/MIT-pzprjs.txt` |

The first rule line of `ics/llm/heyawake.txt` is quoted from PPBench's website, ppbench.com.

TRM's training code is itself based on HRM's (TRM's README); the files adapted from it keep TRM's MIT terms.
`ics/optim.py`'s `AdamATan2` reimplements the update rule of adam-atan2 0.0.3 (github.com/imoneoi/adam-atan2, Apache
License 2.0); no code of it is copied. `ics_baselines/gram/` is our open-source reproduction of GRAM (Baek et al.,
2026), whose code is not public; it builds on the TRM code above. Building the Light-Up and PPBench datasets runs the
optional `ppbench` package, which reads each board through pzpr.js under Node.js; neither program is part of this
repository.
