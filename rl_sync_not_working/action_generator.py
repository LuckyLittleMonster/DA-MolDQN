"""ActionGenerator: Unified interface for molecular action generation.

All action generators (template, hypergraph, two-model, hybrid) implement
this ABC to provide a consistent get_valid_actions() API for RL training.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class ActionGenerator(ABC):
    """Base class for all action generators.

    Subclasses must implement load() and get_valid_actions().
    get_valid_actions_batch() has a default serial implementation.
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Initialize models/data (lazy loading)."""
        pass

    @abstractmethod
    def get_valid_actions(
        self, mol, top_k: int = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Generate valid actions for a molecule.

        Args:
            mol: RDKit Mol or SMILES string.
            top_k: Max actions to return.

        Returns:
            co_reactants: List of co-reactant SMILES ("" for unimolecular).
            products: List of product SMILES.
            scores: np.float32 array of confidence scores.
        """
        pass

    def get_valid_actions_batch(self, mols, top_k: int = None) -> list:
        """Batch version (default: serial fallback)."""
        return [self.get_valid_actions(mol, top_k) for mol in mols]


class AtomLevelActionGenerator(ActionGenerator):
    """Original MolDQN atom-level operations (self-contained).

    Generates atom add / bond add / bond remove actions directly using RDKit.
    No dependency on the legacy environment.Molecule class.

    Products are SMILES of modified molecules, co_reactants are always "".
    """

    name = "atom_level"

    def __init__(self, atom_types=None, allow_removal=True,
                 allow_no_modification=True):
        import hyp
        self.atom_types = set(atom_types or hyp.atom_types)
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification

    def load(self) -> None:
        pass

    def get_valid_actions(
        self, mol, top_k: int = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        from rdkit import Chem

        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return [], [], np.array([], dtype=np.float32)
            mol = Chem.RWMol(mol)

        smiles = Chem.MolToSmiles(mol)
        products = []

        # Keep-same action
        if self.allow_no_modification:
            products.append(smiles)

        # Add atoms
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetImplicitValence() == 0:
                continue
            for atom_type in self.atom_types:
                try:
                    new_mol = Chem.RWMol(mol)
                    new_atom_idx = new_mol.AddAtom(Chem.Atom(atom_type))
                    new_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(new_mol)
                    new_smi = Chem.MolToSmiles(new_mol)
                    if new_smi != smiles:
                        products.append(new_smi)
                except Exception:
                    continue

        # Add bonds between existing atoms
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                if mol.GetBondBetweenAtoms(i, j) is None:
                    try:
                        new_mol = Chem.RWMol(mol)
                        new_mol.AddBond(i, j, Chem.BondType.SINGLE)
                        Chem.SanitizeMol(new_mol)
                        new_smi = Chem.MolToSmiles(new_mol)
                        if new_smi != smiles:
                            products.append(new_smi)
                    except Exception:
                        continue

        # Remove bonds
        if self.allow_removal:
            for bond in mol.GetBonds():
                if bond.GetIsAromatic():
                    continue
                try:
                    new_mol = Chem.RWMol(mol)
                    bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    new_mol.RemoveBond(bi, bj)
                    frags = Chem.GetMolFrags(new_mol)
                    if len(frags) == 1:
                        Chem.SanitizeMol(new_mol)
                        new_smi = Chem.MolToSmiles(new_mol)
                        if new_smi != smiles:
                            products.append(new_smi)
                except Exception:
                    continue

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in products:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        products = unique

        if top_k is not None:
            products = products[:top_k]

        co_reactants = [""] * len(products)
        scores = np.ones(len(products), dtype=np.float32)
        return co_reactants, products, scores
