"""Building block library for template-based reactions."""

import gzip
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol as RDMol


class BuildingBlockLibrary:
    """Loads and stores a library of building block molecules.

    Building blocks are loaded lazily from a SMILES file (.smi or .smi.gz).
    Each line should be tab-separated: SMILES<tab>ID (ID is optional).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._smiles: list[str] = []
        self._mols: list[RDMol] = []
        self._loaded = False

    def load(self) -> None:
        """Parse the SMILES file and convert to RDKit Mol objects."""
        if self._loaded:
            return

        raw_lines = []
        if self.path.suffix == '.gz':
            with gzip.open(self.path, 'rt') as f:
                raw_lines = f.readlines()
        else:
            with open(self.path) as f:
                raw_lines = f.readlines()

        smiles_list = []
        mol_list = []
        for line in raw_lines:
            parts = line.strip().split()
            if not parts:
                continue
            smi = parts[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))  # canonical
                mol_list.append(mol)

        self._smiles = smiles_list
        self._mols = mol_list
        self._loaded = True

    @property
    def smiles_list(self) -> list[str]:
        return self._smiles

    @property
    def mol_list(self) -> list[RDMol]:
        return self._mols

    def __len__(self) -> int:
        return len(self._smiles)

    def __getitem__(self, idx: int) -> tuple[str, RDMol]:
        return self._smiles[idx], self._mols[idx]
