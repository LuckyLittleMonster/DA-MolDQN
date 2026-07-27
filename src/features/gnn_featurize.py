"""Atom/bond featurisation for the frozen property teacher.

Copied from rep_gnn/featurize.py so the 12-d atom / 6-d bond layout matches the trained
checkpoints exactly. Takes an RDKit Mol (the production env already hands out Mols in
``valid_actions``), so no SMILES round-trip is needed on the hot path.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 3
FP_BITS = 2048
_morgan = rdFingerprintGenerator.GetMorganGenerator(fpSize=FP_BITS, radius=FP_RADIUS)


def atom_features(atom) -> list:
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        (atom.GetFormalCharge() + 2.0) / 4.0,
        atom.GetHybridization().real / 6.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs(includeNeighbors=False) / 4.0,
        (atom.GetExplicitValence() + atom.GetImplicitValence()) / 6.0,
        atom.GetMass() / 200.0,
        float(atom.GetNumRadicalElectrons()),
        atom.GetChiralTag().real / 3.0,
        float(atom.IsInRing()),
        atom.GetImplicitValence() / 6.0,
    ]


def bond_features(bond) -> list:
    return [
        bond.GetBondType().real / 3.0,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        bond.GetStereo().real / 6.0,
        float(bond.GetIsAromatic()),
        bond.GetBondDir().real / 6.0,
    ]


def mol_to_graph(mol):
    # imported here, not at module scope: the fingerprint path must not pay for
    # torch_geometric (it is loaded by every rank at startup otherwise)
    from torch_geometric.data import Data
    """RDKit mol -> torch_geometric Data (no label)."""
    af = [atom_features(a) for a in mol.GetAtoms()]
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        ei.extend([[i, j], [j, i]])
        ea.extend([bf, bf])
    x = torch.tensor(af, dtype=torch.float)
    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(ea, dtype=torch.float)
    else:  # single atom, no bonds
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 6, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


