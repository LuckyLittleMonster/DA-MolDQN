"""Unified final-molecule analysis across every arm.

Primary metric per the goal: the property of the FINAL molecule of the FINAL episode -- the
converged greedy policy's deterministic output (eps = 0.0015 by the end). best/top-k are
whole-run maxima and are dominated by whichever epsilon-greedy step got lucky.

Its reward has to be RECOMPUTED: the pipeline never scores s_T (trainer.py appends the rewards
from st = T-1), so the recorded value belongs to the PENULTIMATE molecule.
"""
import gzip, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def finals(exp, trial):
    p = os.path.join(HERE, "Experiments", f"{exp}_{trial}", f"{exp}_{trial}.pickle.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        d = pickle.load(f)
    out = []
    for _, P in sorted(d["paths"].items()):
        L = P.get("last")
        if not L:
            continue
        slots, _ = L[-1]
        out += [path[-1] for path in slots if path]
    return out


def summarize(name, groups):
    """groups: {arm: [list_of_mols_per_seed]}; scorer chosen by caller."""
    print(f"\n===== {name} =====")
    return groups
