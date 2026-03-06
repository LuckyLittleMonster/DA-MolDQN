"""Numerical equivalence tests: PyTorch BDE predictor vs TF reference.

The bond/edge ordering may differ between our preprocessor and nfp
(networkx DiGraph iteration order vs sequential RDKit bond iteration),
but final predictions must match because scatter operations are order-invariant.
"""
import json
import numpy as np
import torch
import pytest

REFERENCE_PATH = 'bde_predictor/tests/reference_predictions.json'
WEIGHTS_PATH = 'bde_predictor/weights/bde_db2_model3.npz'
PP_JSON = 'BDE-db2/Example-BDE-prediction/model_3_tfrecords_multi_halo_cfc/preprocessor.json'


@pytest.fixture
def reference():
    with open(REFERENCE_PATH) as f:
        return json.load(f)


@pytest.fixture
def pytorch_model():
    from bde_predictor.model import BDEPredictor
    return BDEPredictor.from_npz(WEIGHTS_PATH)


@pytest.fixture
def preprocessor():
    from bde_predictor.preprocessor import BDEPreprocessor
    return BDEPreprocessor(PP_JSON)


def test_atom_tokens_match_nfp(reference, preprocessor):
    """Atom tokens must match exactly (same iteration order)."""
    for smi, ref in reference.items():
        graph = preprocessor.process_smiles(smi)
        ref_atoms = np.array(ref['inputs']['atom'])
        np.testing.assert_array_equal(graph['atom'], ref_atoms,
            err_msg=f"Atom tokens mismatch for {smi}")


def test_bond_count_matches_nfp(reference, preprocessor):
    """Number of directed edges must match (order may differ)."""
    for smi, ref in reference.items():
        graph = preprocessor.process_smiles(smi)
        ref_bonds = np.array(ref['inputs']['bond'])
        assert len(graph['bond']) == len(ref_bonds), \
            f"Bond count mismatch for {smi}: {len(graph['bond'])} vs {len(ref_bonds)}"


def test_bond_tokens_same_set(reference, preprocessor):
    """Bond token multisets must be identical (order may differ)."""
    for smi, ref in reference.items():
        graph = preprocessor.process_smiles(smi)
        ref_bonds = sorted(ref['inputs']['bond'])
        our_bonds = sorted(graph['bond'].tolist())
        assert our_bonds == ref_bonds, \
            f"Bond token multiset mismatch for {smi}"


@pytest.mark.parametrize("smi", [
    'c1ccc(O)cc1',
    'CC(=O)Oc1ccccc1O',
    'Oc1ccc(O)cc1',
    'CC(C)(C)c1cc(O)cc(C(C)(C)C)c1O',
    'O=C(O)c1ccccc1O',
])
def test_predictions_match_tf(smi, reference, pytorch_model, preprocessor):
    """PyTorch predictions must match TF within atol=1e-4."""
    ref = reference[smi]
    graph = preprocessor.process_smiles(smi)
    batch = preprocessor.collate([graph])

    with torch.no_grad():
        pred = pytorch_model(**batch).numpy()[0]  # [n_bonds, 2]

    ref_pred = np.array(ref['predictions'])
    n_bonds = min(pred.shape[0], ref_pred.shape[0])

    np.testing.assert_allclose(
        pred[:n_bonds], ref_pred[:n_bonds],
        atol=1e-4, rtol=1e-4,
        err_msg=f"Prediction mismatch for {smi}")


def test_batch_predictions_match_single(pytorch_model, preprocessor):
    """Batch predictions must match single-molecule predictions."""
    smiles = ['c1ccc(O)cc1', 'Oc1ccc(O)cc1', 'O=C(O)c1ccccc1O']

    # Single predictions
    singles = []
    for smi in smiles:
        graph = preprocessor.process_smiles(smi)
        batch = preprocessor.collate([graph])
        with torch.no_grad():
            pred = pytorch_model(**batch).numpy()[0]
        n_bonds = len(graph['bond']) // 2
        singles.append(pred[:n_bonds])

    # Batch prediction
    graphs = [preprocessor.process_smiles(smi) for smi in smiles]
    batch = preprocessor.collate(graphs)
    with torch.no_grad():
        batch_pred = pytorch_model(**batch).numpy()

    for i, (smi, single) in enumerate(zip(smiles, singles)):
        n_bonds = len(graphs[i]['bond']) // 2
        np.testing.assert_allclose(
            batch_pred[i, :n_bonds], single,
            atol=1e-6,
            err_msg=f"Batch vs single mismatch for {smi}")
