"""Penalized logP reward component.

Per-molecule normalized score = norm(logP) - norm(SA) - norm(largest-ring-excess).
Normalization constants and behaviour match the original agent.find_plogp_reward.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.reward.sa import sa_score

_LOGP_MEAN = 2.4570953396190123
_LOGP_STD = 1.434324401111988
_SA_MEAN = 3.0525811293166134
_SA_STD = 0.8335207024513095
_CYCLE_MEAN = 0.0485696876403053
_CYCLE_STD = 0.2860212110245455


def plogp_value(mol) -> float:
    try:
        mol.UpdatePropertyCache()
        cycles = Chem.GetSymmSSSR(mol)
        if cycles:
            max_cycle = max(len(cycle) for cycle in cycles)
            cycle = max(0, max_cycle - 6)
        else:
            cycle = 0
        logp = (Descriptors.MolLogP(mol) - _LOGP_MEAN) / _LOGP_STD
        sa = (sa_score(mol) - _SA_MEAN) / _SA_STD
        cyc = (cycle - _CYCLE_MEAN) / _CYCLE_STD
        return logp - sa - cyc
    except Chem.AtomValenceException:
        return -30
