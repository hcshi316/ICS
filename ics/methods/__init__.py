"""Test-time methods; the baselines' protocols (ics_baselines/*/predict.py) follow the same conventions. Each run_*
returns a dict of arrays under keys of one grammar:
  <row>/<regime>/<scoring>           answers, one [N, L] array
  <row>/<regime>/<extra>             a per-board extra, e.g. ics/cert/segs
  <row>/<regime>/<scoring>/<extra>   a per-board extra that differs by scoring, e.g. ptrm/cert/raw/rollout
where the row is the paper's table row, the regime is "cert" (a task certificate is available at selection) or "model"
(it is not), and the scoring is "raw" or "pinned" (answers under "pinned" are decodes too, which task.check pins)."""
import numpy as np

ANSWER_DTYPE = np.int16
