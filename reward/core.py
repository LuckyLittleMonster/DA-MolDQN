"""Shared reward utilities: SA Score, Lipinski, PAINS."""

import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ---------------------------------------------------------------------------
# SA Score helper (lazy loaded)
# ---------------------------------------------------------------------------

_sascorer = None


def _load_sascorer():
    global _sascorer
    if _sascorer is None:
        from rdkit import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        _sascorer = sascorer
    return _sascorer


# ---------------------------------------------------------------------------
# Lipinski filter
# ---------------------------------------------------------------------------

def _check_lipinski(mol, cfg_reward):
    """Return True if mol passes Lipinski constraints from cfg."""
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = 0
    if mw > cfg_reward.max_mw:
        violations += 1
    if logp > cfg_reward.max_logp:
        violations += 1
    if hbd > cfg_reward.max_hbd:
        violations += 1
    if hba > cfg_reward.max_hba:
        violations += 1
    return violations <= 1  # Lipinski allows 1 violation


# ---------------------------------------------------------------------------
# PAINS filter
# ---------------------------------------------------------------------------

_pains_catalog = None


def _check_pains(mol):
    """Return True if mol contains PAINS substructures (bad)."""
    global _pains_catalog
    if _pains_catalog is None:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog(params)
    return _pains_catalog.HasMatch(mol)
