"""Shared utility functions for model_reactions package."""

from collections import OrderedDict
from typing import Optional

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


# Shared fingerprint generator (Morgan, radius=2, 2048 bits)
_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string.

    Returns None if the SMILES is invalid or cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def tanimoto_similarity(smi1: str, smi2: str) -> float:
    """Compute Tanimoto similarity between two SMILES using Morgan fingerprints.

    Uses rdFingerprintGenerator (Morgan, radius=2, 2048 bits).
    Returns 0.0 if either SMILES is invalid.
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = _FP_GEN.GetFingerprint(mol1)
    fp2 = _FP_GEN.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def tanimoto_from_mols(mol_a, mol_b) -> float:
    """Compute Tanimoto similarity between two RDKit Mol objects.

    Faster than tanimoto_similarity() when you already have Mol objects.
    """
    fp_a = _FP_GEN.GetFingerprint(mol_a)
    fp_b = _FP_GEN.GetFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


class LRUCache:
    """Simple LRU cache with fixed capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.cache)
