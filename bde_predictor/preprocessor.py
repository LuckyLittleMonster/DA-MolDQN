"""Pure Python/RDKit molecular preprocessor for BDE prediction.

Replaces nfp.SmilesBondIndexPreprocessor without TF dependency.
Loads the trained tokenizer vocabulary from preprocessor.json.
"""
import json
import numpy as np
from rdkit import Chem
from pathlib import Path


def get_ring_size(x, max_size=6):
    """Get smallest ring size for an atom or bond, matching nfp implementation.

    Uses range(max_size), i.e. checks sizes 0..max_size-1.
    With max_size=6, sizes 3,4,5 are detectable; size 6+ maps to 'max'.
    """
    if not x.IsInRing():
        return 0
    for i in range(max_size):
        if x.IsInRingSize(i):
            return i
    return "max"


def atom_featurizer(atom):
    """Atom feature string hash, identical to preprocess_inputs_cfc.py.

    Returns str((Symbol, NumRadicalElectrons, FormalCharge, ChiralTag,
                 IsAromatic, get_ring_size(atom), Degree, TotalNumHs)).

    Note: str() on a tuple calls repr() on each element, so ChiralTag
    renders as 'rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED' etc.,
    matching the vocab keys in preprocessor.json.
    """
    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
    ))


def bond_featurizer(bond, flipped=False):
    """Bond feature string hash, identical to preprocess_inputs_cfc.py.

    Returns "AtomA-AtomB BondType [Ring]" with direction sensitivity.
    Example: "C-C AROMATIC Rmax", "O-H SINGLE"
    """
    if not flipped:
        atoms = "{}-{}".format(
            bond.GetBeginAtom().GetSymbol(),
            bond.GetEndAtom().GetSymbol())
    else:
        atoms = "{}-{}".format(
            bond.GetEndAtom().GetSymbol(),
            bond.GetBeginAtom().GetSymbol())

    btype = str(bond.GetBondType())
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''

    return " ".join([atoms, btype, ring]).strip()


class Tokenizer:
    """Frozen tokenizer that maps feature strings to integer IDs.

    Token 0 = padding (never assigned by tokenizer).
    Token 1 = unknown (unk).
    Tokens 2+ = known features from training vocabulary.
    """
    def __init__(self, data: dict, num_classes: int):
        self._data = data
        self._num_classes = num_classes

    def __call__(self, item: str) -> int:
        return self._data.get(item, self._data.get('unk', 1))

    @property
    def num_classes(self):
        """Max token value (not count). Embedding size = num_classes + 1."""
        return self._num_classes


class BDEPreprocessor:
    """Convert SMILES to padded PyTorch tensors for BDE prediction.

    Replaces nfp.SmilesBondIndexPreprocessor:
      - Loads frozen vocab from preprocessor.json (no training mode)
      - explicit_hs=True: molecules always have explicit hydrogens
      - Directed graph: each RDKit bond -> 2 directed edges (forward + reverse)
      - bond_indices: RDKit bond.GetIdx() per directed edge
    """

    def __init__(self, preprocessor_json_path: str):
        with open(preprocessor_json_path) as f:
            config = json.load(f)
        self.atom_tokenizer = Tokenizer(
            config['atom_tokenizer']['_data'],
            config['atom_tokenizer']['num_classes'],
        )
        self.bond_tokenizer = Tokenizer(
            config['bond_tokenizer']['_data'],
            config['bond_tokenizer']['num_classes'],
        )

    def process_smiles(self, smiles: str) -> dict:
        """Convert a single SMILES to numpy arrays.

        Returns dict with:
          atom:         int32 [num_atoms]        - atom token IDs
          bond:         int32 [num_directed_edges] - bond token IDs
          connectivity: int64 [num_directed_edges, 2] - (src, dst) pairs
          bond_indices: int32 [num_directed_edges] - RDKit bond index per edge
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        atom_tokens = np.array(
            [self.atom_tokenizer(atom_featurizer(a)) for a in mol.GetAtoms()],
            dtype=np.int32)

        bonds = list(mol.GetBonds())
        if len(bonds) == 0:
            return {
                'atom': atom_tokens,
                'bond': np.zeros(0, dtype=np.int32),
                'connectivity': np.zeros((0, 2), dtype=np.int64),
                'bond_indices': np.zeros(0, dtype=np.int32),
            }

        bond_tokens = []
        connectivity = []
        bond_indices_list = []

        for bond in bonds:
            idx = bond.GetIdx()
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            # Forward edge: begin -> end
            bond_tokens.append(self.bond_tokenizer(
                bond_featurizer(bond, flipped=False)))
            connectivity.append([begin, end])
            bond_indices_list.append(idx)

            # Reverse edge: end -> begin
            bond_tokens.append(self.bond_tokenizer(
                bond_featurizer(bond, flipped=True)))
            connectivity.append([end, begin])
            bond_indices_list.append(idx)

        return {
            'atom': atom_tokens,
            'bond': np.array(bond_tokens, dtype=np.int32),
            'connectivity': np.array(connectivity, dtype=np.int64),
            'bond_indices': np.array(bond_indices_list, dtype=np.int32),
        }

    def collate(self, graphs: list, device='cpu') -> dict:
        """Pad and batch a list of graph dicts into PyTorch tensors.

        Returns dict with:
          atom:         long [B, max_atoms]
          bond:         long [B, max_edges]
          connectivity: long [B, max_edges, 2]
          bond_indices: long [B, max_edges]

        Padding uses 0 (the null token).
        """
        import torch

        B = len(graphs)
        max_atoms = max(len(g['atom']) for g in graphs)
        max_edges = max(len(g['bond']) for g in graphs) if graphs else 0

        atom = torch.zeros(B, max_atoms, dtype=torch.long, device=device)
        bond = torch.zeros(B, max_edges, dtype=torch.long, device=device)
        connectivity = torch.zeros(B, max_edges, 2, dtype=torch.long, device=device)
        bond_indices = torch.zeros(B, max_edges, dtype=torch.long, device=device)

        for i, g in enumerate(graphs):
            na = len(g['atom'])
            ne = len(g['bond'])
            atom[i, :na] = torch.from_numpy(g['atom']).long()
            bond[i, :ne] = torch.from_numpy(g['bond']).long()
            connectivity[i, :ne] = torch.from_numpy(g['connectivity']).long()
            bond_indices[i, :ne] = torch.from_numpy(g['bond_indices']).long()

        return {
            'atom': atom,
            'bond': bond,
            'connectivity': connectivity,
            'bond_indices': bond_indices,
        }

    def get_oh_bond_indices(self, smiles: str) -> list:
        """Get RDKit bond indices for all O-H bonds in a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        mol = Chem.AddHs(mol)
        oh_ids = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomicNum()
            a2 = bond.GetEndAtom().GetAtomicNum()
            if (a1 == 8 and a2 == 1) or (a1 == 1 and a2 == 8):
                oh_ids.append(bond.GetIdx())
        return oh_ids
