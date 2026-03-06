import numpy as np
import pytest
from bde_predictor.preprocessor import BDEPreprocessor, atom_featurizer, bond_featurizer, get_ring_size
from rdkit import Chem

PP_JSON = 'BDE-db2/Example-BDE-prediction/model_3_tfrecords_multi_halo_cfc/preprocessor.json'

@pytest.fixture
def pp():
    return BDEPreprocessor(PP_JSON)

def test_tokenizer_vocab_size(pp):
    assert pp.atom_tokenizer.num_classes == 170
    assert pp.bond_tokenizer.num_classes == 199

def test_process_phenol(pp):
    g = pp.process_smiles('c1ccc(O)cc1')
    mol = Chem.AddHs(Chem.MolFromSmiles('c1ccc(O)cc1'))
    assert len(g['atom']) == mol.GetNumAtoms()
    assert len(g['bond']) == 2 * mol.GetNumBonds()  # directed
    assert g['connectivity'].shape == (len(g['bond']), 2)
    assert g['bond_indices'].shape == g['bond'].shape

def test_process_water(pp):
    g = pp.process_smiles('O')
    mol = Chem.AddHs(Chem.MolFromSmiles('O'))
    assert len(g['atom']) == 3  # O + 2H
    assert len(g['bond']) == 4  # 2 bonds x 2 directions

def test_all_tokens_nonzero(pp):
    """All real atoms/bonds should have non-zero tokens (0 = padding)."""
    g = pp.process_smiles('c1ccc(O)cc1')
    assert np.all(g['atom'] > 0)
    assert np.all(g['bond'] > 0)

def test_collate_batch(pp):
    import torch
    g1 = pp.process_smiles('c1ccc(O)cc1')
    g2 = pp.process_smiles('O')
    batch = pp.collate([g1, g2])
    assert batch['atom'].shape[0] == 2
    assert batch['bond'].dtype == torch.long
    # Padding: smaller mol should have zeros at the end
    n2_atoms = len(g2['atom'])
    assert batch['atom'][1, n2_atoms:].sum() == 0

def test_oh_detection_phenol(pp):
    oh_ids = pp.get_oh_bond_indices('c1ccc(O)cc1')
    assert len(oh_ids) == 1

def test_oh_detection_hydroquinone(pp):
    oh_ids = pp.get_oh_bond_indices('Oc1ccc(O)cc1')
    assert len(oh_ids) == 2

def test_oh_detection_no_oh(pp):
    oh_ids = pp.get_oh_bond_indices('CCCC')
    assert len(oh_ids) == 0

def test_bond_directionality(pp):
    """Each undirected bond should produce 2 directed edges with same bond_index."""
    g = pp.process_smiles('O')  # water: O-H, O-H
    # Should have 4 directed edges (2 bonds x 2 directions)
    assert len(g['bond']) == 4
    # bond_indices should be [0, 0, 1, 1] (pair-wise same)
    assert g['bond_indices'][0] == g['bond_indices'][1]
    assert g['bond_indices'][2] == g['bond_indices'][3]
    # Connectivity should be reversed for each pair
    assert g['connectivity'][0][0] == g['connectivity'][1][1]
    assert g['connectivity'][0][1] == g['connectivity'][1][0]

def test_atom_featurizer_consistency():
    """Verify atom featurizer matches the format in preprocessor.json vocab."""
    mol = Chem.AddHs(Chem.MolFromSmiles('C'))  # methane
    c_atom = mol.GetAtomWithIdx(0)
    feat = atom_featurizer(c_atom)
    assert feat.startswith("('C',")
    assert isinstance(feat, str)

def test_invalid_smiles(pp):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        pp.process_smiles('invalid_smiles_xyz')

def test_ring_size_benzene():
    """Benzene with max_size=6 should give 'max' (range(6) = 0..5, misses 6)."""
    mol = Chem.MolFromSmiles('c1ccccc1')
    mol = Chem.AddHs(mol)
    c_atom = mol.GetAtomWithIdx(0)
    assert get_ring_size(c_atom, max_size=6) == 'max'

def test_ring_size_cyclopentane():
    """Cyclopentane should give ring size 5."""
    mol = Chem.MolFromSmiles('C1CCCC1')
    mol = Chem.AddHs(mol)
    c_atom = mol.GetAtomWithIdx(0)
    assert get_ring_size(c_atom, max_size=6) == 5

def test_ring_size_no_ring():
    """Non-ring atom should give 0."""
    mol = Chem.MolFromSmiles('CC')
    mol = Chem.AddHs(mol)
    c_atom = mol.GetAtomWithIdx(0)
    assert get_ring_size(c_atom, max_size=6) == 0

def test_bond_featurizer_benzene_ring():
    """Benzene C-C bond should produce 'C-C AROMATIC Rmax' (6-membered ring -> max)."""
    mol = Chem.MolFromSmiles('c1ccccc1')
    mol = Chem.AddHs(mol)
    # First bond is C-C aromatic in ring
    cc_bond = None
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetSymbol() == 'C' and
            bond.GetEndAtom().GetSymbol() == 'C'):
            cc_bond = bond
            break
    assert cc_bond is not None
    feat = bond_featurizer(cc_bond, flipped=False)
    assert feat == 'C-C AROMATIC Rmax'

def test_atom_feature_matches_vocab(pp):
    """Verify that common atom features produce known vocab tokens (not unknown=1)."""
    # Aromatic carbon in benzene should be in vocab
    mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    c_atom = mol.GetAtomWithIdx(0)
    feat = atom_featurizer(c_atom)
    token = pp.atom_tokenizer(feat)
    assert token > 1, f"Aromatic C got unknown token for feature: {feat}"

    # Hydrogen should be in vocab
    h_atom = mol.GetAtomWithIdx(6)  # first H
    feat_h = atom_featurizer(h_atom)
    token_h = pp.atom_tokenizer(feat_h)
    assert token_h > 1, f"H got unknown token for feature: {feat_h}"

def test_bond_feature_matches_vocab(pp):
    """Verify that common bond features produce known vocab tokens."""
    mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    # C-C aromatic bond
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetSymbol() == 'C' and
            bond.GetEndAtom().GetSymbol() == 'C'):
            feat = bond_featurizer(bond, flipped=False)
            token = pp.bond_tokenizer(feat)
            assert token > 1, f"C-C aromatic got unknown token for: {feat}"
            break

    # C-H single bond
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetSymbol() == 'C' and
            bond.GetEndAtom().GetSymbol() == 'H'):
            feat = bond_featurizer(bond, flipped=False)
            token = pp.bond_tokenizer(feat)
            assert token > 1, f"C-H single got unknown token for: {feat}"
            break
