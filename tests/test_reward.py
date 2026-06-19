"""Phase 3: reward component primitives match the original inline logic."""
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from src.reward.qed import qed_value
from src.reward.sa import sa_score
from src.reward.plogp import plogp_value


def test_qed_matches_rdkit():
    m = Chem.MolFromSmiles("CCO")
    assert abs(qed_value(m) - QED.qed(m)) < 1e-12


def test_sa_in_range():
    m = Chem.MolFromSmiles("CCO")
    s = sa_score(m)
    assert isinstance(s, float)
    assert 1.0 <= s <= 10.0


def test_plogp_matches_formula():
    m = Chem.MolFromSmiles("c1ccccc1CCO")
    # Recompute the original penalized-logP formula independently.
    logp_mean, logp_std = 2.4570953396190123, 1.434324401111988
    sa_mean, sa_std = 3.0525811293166134, 0.8335207024513095
    cycle_mean, cycle_std = 0.0485696876403053, 0.2860212110245455
    m.UpdatePropertyCache()
    cycles = Chem.GetSymmSSSR(m)
    cycle = max(0, max(len(c) for c in cycles) - 6) if cycles else 0
    logp = (Descriptors.MolLogP(m) - logp_mean) / logp_std
    sa = (sa_score(m) - sa_mean) / sa_std
    cyc = (cycle - cycle_mean) / cycle_std
    expected = logp - sa - cyc
    assert abs(plogp_value(m) - expected) < 1e-9
