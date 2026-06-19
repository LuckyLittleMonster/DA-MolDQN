"""Synthetic Accessibility (SA Score) reward component.

Centralises locating RDKit's ``sascorer`` contrib, which lives at a relative
``Contrib`` path under source builds but at ``$PREFIX/share/RDKit/Contrib`` in
conda-forge rdkit.
"""
import os
import sys

import rdkit
from rdkit.Chem import RDConfig

_sa_dir = os.path.join(os.path.dirname(rdkit.__file__), RDConfig.RDContribDir, "SA_Score")
if not os.path.isdir(_sa_dir):
    _sa_dir = os.path.join(sys.prefix, "share", "RDKit", "Contrib", "SA_Score")
if _sa_dir not in sys.path:
    sys.path.append(_sa_dir)

import sascorer  # noqa: E402


def sa_score(mol) -> float:
    return sascorer.calculateScore(mol)
