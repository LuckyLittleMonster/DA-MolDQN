"""Tests for high-level BDE prediction API."""
import pytest
import numpy as np

WEIGHTS = 'bde_predictor/weights/bde_db2_model3.npz'


@pytest.fixture
def model():
    from bde_predictor.predict import BDEModel
    return BDEModel(WEIGHTS)


def test_predict_phenol_has_oh(model):
    results = model.predict_smiles(['c1ccc(O)cc1'])
    assert len(results) == 1
    r = results[0]
    assert r['valid']
    assert r['oh_bde'] is not None
    assert 50 < r['oh_bde'] < 110
    assert len(r['oh_bond_indices']) == 1


def test_predict_oh_bde_batch(model):
    bdes, valids = model.predict_oh_bde(['c1ccc(O)cc1', 'Oc1ccc(O)cc1', 'CCCC'])
    assert valids[0] is True   # phenol has O-H
    assert valids[1] is True   # hydroquinone has O-H
    assert valids[2] is False  # butane has no O-H
    assert 50 < bdes[0] < 110
    assert 50 < bdes[1] < 110


def test_predict_invalid_smiles(model):
    bdes, valids = model.predict_oh_bde(['invalid_xyz', 'c1ccc(O)cc1'])
    assert valids[0] is False
    assert valids[1] is True


def test_predict_all_bonds(model):
    results = model.predict_smiles(['c1ccc(O)cc1'])
    r = results[0]
    assert r['bde'].shape[0] == 13  # phenol has 13 bonds
    assert r['bdfe'].shape[0] == 13
    assert np.all(r['bde'] > 0)  # all BDEs should be positive


def test_predict_hydroquinone_two_oh(model):
    results = model.predict_smiles(['Oc1ccc(O)cc1'])
    r = results[0]
    assert len(r['oh_bond_indices']) == 2
    assert r['oh_bde'] is not None


def test_batch_consistency(model):
    """Single vs batch predictions should match."""
    smiles = ['c1ccc(O)cc1', 'Oc1ccc(O)cc1']
    batch_results = model.predict_smiles(smiles)
    for i, smi in enumerate(smiles):
        single_result = model.predict_smiles([smi])[0]
        n = len(single_result['bde'])
        np.testing.assert_allclose(
            batch_results[i]['bde'][:n], single_result['bde'][:n], atol=1e-6)
