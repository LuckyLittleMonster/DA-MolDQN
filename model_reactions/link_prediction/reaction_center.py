"""
Reaction Center Extraction and Prediction.

Extracts reaction center information from USPTO reaction SMILES using atom mapping,
and provides a lightweight GNN predictor for unseen molecules.

Reaction centers are atoms that participate in bond changes during a reaction.
This information is used as additional atom-level features in the molecular encoder.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_changed_atoms_from_rxn_smiles(rxn_smiles: str) -> Dict[str, set]:
    """
    Extract atoms involved in bond changes from a mapped reaction SMILES.

    Args:
        rxn_smiles: Reaction SMILES string (reactants>>products)

    Returns:
        Dict mapping reactant canonical SMILES -> set of atom indices
        that participate in the reaction center.
    """
    if '>>' not in rxn_smiles:
        return {}

    parts = rxn_smiles.split('>>')
    reactant_str = parts[0]
    product_str = parts[1] if len(parts) > 1 else ''

    reactant_mols = []
    for s in reactant_str.split('.'):
        s = s.strip()
        if s:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                reactant_mols.append((s, mol))

    product_mols = []
    for s in product_str.split('.'):
        s = s.strip()
        if s:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                product_mols.append(mol)

    if not reactant_mols or not product_mols:
        return {}

    # Build atom map -> (mol_idx, atom_idx) for reactants
    reactant_map = {}
    for mol_i, (smi, mol) in enumerate(reactant_mols):
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                reactant_map[map_num] = (mol_i, atom.GetIdx())

    # Build atom map -> atom_idx for products
    product_map = {}
    for mol in product_mols:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                product_map[map_num] = atom

    # Build bond sets for reactants
    reactant_bonds = {}
    for mol_i, (smi, mol) in enumerate(reactant_mols):
        for bond in mol.GetBonds():
            a_map = bond.GetBeginAtom().GetAtomMapNum()
            b_map = bond.GetEndAtom().GetAtomMapNum()
            if a_map > 0 and b_map > 0:
                key = (min(a_map, b_map), max(a_map, b_map))
                reactant_bonds[key] = (mol_i, bond.GetBondType())

    # Build bond sets for products
    product_bonds = {}
    for mol in product_mols:
        for bond in mol.GetBonds():
            a_map = bond.GetBeginAtom().GetAtomMapNum()
            b_map = bond.GetEndAtom().GetAtomMapNum()
            if a_map > 0 and b_map > 0:
                key = (min(a_map, b_map), max(a_map, b_map))
                product_bonds[key] = bond.GetBondType()

    # Find changed bonds (broken, formed, or changed type)
    changed_atom_maps = set()

    # Bonds broken (in reactants but not in products)
    for bond_key in reactant_bonds:
        if bond_key not in product_bonds:
            changed_atom_maps.add(bond_key[0])
            changed_atom_maps.add(bond_key[1])
        elif reactant_bonds[bond_key][1] != product_bonds[bond_key]:
            changed_atom_maps.add(bond_key[0])
            changed_atom_maps.add(bond_key[1])

    # Bonds formed (in products but not in reactants)
    for bond_key in product_bonds:
        if bond_key not in reactant_bonds:
            changed_atom_maps.add(bond_key[0])
            changed_atom_maps.add(bond_key[1])

    if not changed_atom_maps:
        return {}

    # Map back to per-reactant atom indices
    result = {}
    for mol_i, (smi, mol) in enumerate(reactant_mols):
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if canon_smi is None:
            continue
        changed_indices = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in changed_atom_maps:
                changed_indices.add(atom.GetIdx())
        if changed_indices:
            if canon_smi not in result:
                result[canon_smi] = set()
            result[canon_smi].update(changed_indices)

    return result


def extract_reaction_centers(db, verbose: bool = True) -> Dict[int, np.ndarray]:
    """
    Extract reaction center information for all molecules in the database.

    For each molecule, produces a per-atom score indicating how often each atom
    participates as a reaction center across all known reactions.

    Args:
        db: ReactionDatabase instance
        verbose: Print progress

    Returns:
        Dict mapping molecule index -> np.ndarray of shape (n_atoms,)
        with per-atom reaction center frequency scores (0-1 normalized).
    """
    import pandas as pd
    from pathlib import Path

    if verbose:
        print("Extracting reaction centers from atom-mapped SMILES...")

    # We need the raw rxn_smiles from CSV to get atom mappings
    mol_center_counts = defaultdict(lambda: defaultdict(int))
    mol_total_rxns = defaultdict(int)
    n_mapped = 0
    n_total = 0

    for split in ['train', 'val', 'test']:
        csv_path = Path(db.data_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rxn_smiles = str(row.get('rxn_smiles', ''))
            if '>>' not in rxn_smiles:
                continue
            n_total += 1

            changed = _get_changed_atoms_from_rxn_smiles(rxn_smiles)
            if changed:
                n_mapped += 1
                for smi, atom_indices in changed.items():
                    canon_smi = smi
                    mol_idx = db.smiles_to_idx.get(canon_smi)
                    if mol_idx is not None:
                        mol_total_rxns[mol_idx] += 1
                        for atom_idx in atom_indices:
                            mol_center_counts[mol_idx][atom_idx] += 1

    if verbose:
        print(f"  Processed {n_total} reactions, {n_mapped} had valid atom mapping")
        print(f"  Found reaction centers for {len(mol_center_counts)} molecules")

    # Convert to per-atom arrays
    result = {}
    for mol_idx, center_counts in mol_center_counts.items():
        smiles = db.reactant_smiles[mol_idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_atoms = mol.GetNumAtoms()
        scores = np.zeros(n_atoms, dtype=np.float32)
        total = mol_total_rxns[mol_idx]
        if total > 0:
            for atom_idx, count in center_counts.items():
                if atom_idx < n_atoms:
                    scores[atom_idx] = count / total
        result[mol_idx] = scores

    if verbose:
        print(f"  Extracted per-atom reaction center scores for {len(result)} molecules")

    return result


def heuristic_reaction_center_scores(smiles: str) -> Optional[np.ndarray]:
    """
    Compute heuristic reaction center scores for an unseen molecule.

    Uses chemical heuristics:
    - Heteroatoms (N, O, S, P, etc.) get higher scores
    - Atoms adjacent to heteroatoms get moderate scores
    - Unsaturated carbons (double/triple bonds) get higher scores
    - Aromatic atoms get moderate scores

    Args:
        smiles: SMILES string

    Returns:
        np.ndarray of shape (n_atoms,) with scores in [0, 1], or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    scores = np.zeros(n_atoms, dtype=np.float32)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        score = 0.0

        # Heteroatoms are more reactive
        atomic_num = atom.GetAtomicNum()
        if atomic_num != 6 and atomic_num != 1:  # Not C or H
            score += 0.5

        # Formal charge indicates reactivity
        if atom.GetFormalCharge() != 0:
            score += 0.3

        # Unsaturated bonds
        for bond in atom.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                score += 0.3
            elif bt == Chem.BondType.TRIPLE:
                score += 0.4

        # Aromatic
        if atom.GetIsAromatic():
            score += 0.1

        # Radical electrons
        if atom.GetNumRadicalElectrons() > 0:
            score += 0.4

        scores[idx] = min(score, 1.0)

    # Also boost neighbors of high-scoring atoms
    neighbor_boost = np.zeros(n_atoms, dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            neighbor_boost[n_idx] = max(neighbor_boost[n_idx], scores[idx] * 0.3)

    scores = np.minimum(scores + neighbor_boost, 1.0)
    return scores


class ReactionCenterPredictor(nn.Module):
    """
    Lightweight GNN for predicting reaction center atoms on unseen molecules.

    Trained on extracted reaction center data, then used at inference
    to predict which atoms in a new molecule are likely reaction centers.
    """

    def __init__(self, atom_feature_dim: int = 39, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(atom_feature_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'msg': nn.Linear(hidden_dim * 2, hidden_dim),
                'upd': nn.Linear(hidden_dim * 2, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
            }))

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_atoms, atom_feature_dim) atom features
            edge_index: (2, num_edges) edge indices

        Returns:
            scores: (num_atoms,) reaction center probability per atom
        """
        h = self.input_proj(x)

        for layer in self.layers:
            row, col = edge_index
            msg = layer['msg'](torch.cat([h[row], h[col]], dim=-1))

            aggr = torch.zeros_like(h)
            count = torch.zeros(h.size(0), 1, device=h.device)
            aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)
            count.scatter_add_(0, col.unsqueeze(-1), torch.ones(col.size(0), 1, device=h.device))
            aggr = aggr / count.clamp(min=1)

            h = layer['norm'](h + layer['upd'](torch.cat([h, aggr], dim=-1)))

        return self.output(h).squeeze(-1)
