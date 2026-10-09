"""The verifier: a TRM fine-tuned from a solver to tell, through its q_halt head, whether a candidate answer is valid.
ICS without a certificate selects with it in place of the search, on Maze always and on any task given --verifier:

    python -m ics train --model trm --task TASK --data DATA --out SOLVER --set train.keep_every_eval=true
    python -m ics verifier-data --task TASK --data DATA --solver SOLVER --out VDATA
    python -m ics train --model verifier --task TASK --data VDATA --init SOLVER/last --out V0 --set train.seed=0
    python -m ics pick-verifier V0 V1 V2 --out VERIFIER                          (V1, V2: train.seed=1, 2)
    python -m ics eval --method ics --task TASK --ckpt SOLVER/last --data DATA --out RESULTS --verifier VERIFIER

A verifier run keeps its best evaluation by the AUC on boards held out from its train split (best/), and pick-verifier
the seed whose best/ scores highest. Modules: data.py (build_verifier_data), train.py (VerifierHead), pick.py
(pick_verifier), select.py (run_ics_verifier, the selection)."""
