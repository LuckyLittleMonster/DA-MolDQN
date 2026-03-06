import torch
import numpy as np
import pytest

WEIGHTS_PATH = 'bde_predictor/weights/bde_db2_model3.npz'

def test_model_param_count():
    from bde_predictor.model import BDEPredictor
    model = BDEPredictor.from_npz(WEIGHTS_PATH)
    total = sum(p.numel() for p in model.parameters())
    assert total == 1_659_920, f"Expected 1,659,920 params, got {total}"

def test_model_forward_shape():
    from bde_predictor.model import BDEPredictor
    model = BDEPredictor(num_atom_types=171, num_bond_types=200)
    B, N, E = 2, 10, 16  # E must be even (directed pairs)
    atom = torch.randint(1, 171, (B, N))
    bond = torch.randint(1, 200, (B, E))
    conn = torch.randint(0, N, (B, E, 2))
    bi = torch.arange(E // 2).unsqueeze(0).expand(B, -1)
    bi = torch.cat([bi, bi], dim=1)  # [0,1,...,7,0,1,...,7]
    out = model(atom, bond, conn, bi)
    assert out.shape == (B, E // 2, 2)

def test_model_forward_with_padding():
    from bde_predictor.model import BDEPredictor
    model = BDEPredictor(num_atom_types=171, num_bond_types=200)
    B, N, E = 2, 10, 16
    atom = torch.zeros(B, N, dtype=torch.long)
    bond = torch.zeros(B, E, dtype=torch.long)
    atom[0, :5] = torch.randint(1, 171, (5,))
    bond[0, :8] = torch.randint(1, 200, (8,))
    conn = torch.zeros(B, E, 2, dtype=torch.long)
    bi = torch.zeros(B, E, dtype=torch.long)
    out = model(atom, bond, conn, bi)
    assert out.shape == (B, E // 2, 2)
    assert torch.isfinite(out).all()

def test_from_npz_loads_weights():
    from bde_predictor.model import BDEPredictor
    model = BDEPredictor.from_npz(WEIGHTS_PATH)
    # Check some weights are non-zero (loaded correctly)
    assert model.atom_embedding.weight.abs().sum() > 0
    assert model.bond_embedding.weight.abs().sum() > 0
    assert model.edge_updates[0].concat_dense.dense1.weight.abs().sum() > 0
    assert model.node_updates[0].message_dense.dense1.weight.abs().sum() > 0
    assert model.bde_no_mean.weight.abs().sum() > 0

def test_from_npz_weight_shapes():
    from bde_predictor.model import BDEPredictor
    data = np.load(WEIGHTS_PATH)
    model = BDEPredictor.from_npz(WEIGHTS_PATH)

    # Verify transposition was done correctly
    tf_kernel = data['edge_update/concat_dense/dense/kernel']  # (384, 256) in TF
    pt_weight = model.edge_updates[0].concat_dense.dense1.weight.data.numpy()  # (256, 384) in PyTorch
    assert pt_weight.shape == (256, 384)
    np.testing.assert_allclose(pt_weight, tf_kernel.T, atol=1e-7)

def test_end_to_end_with_preprocessor():
    """Test full pipeline: SMILES -> preprocessor -> model -> predictions."""
    from bde_predictor.model import BDEPredictor
    from bde_predictor.preprocessor import BDEPreprocessor

    model = BDEPredictor.from_npz(WEIGHTS_PATH)
    pp = BDEPreprocessor('BDE-db2/Example-BDE-prediction/model_3_tfrecords_multi_halo_cfc/preprocessor.json')

    graph = pp.process_smiles('c1ccc(O)cc1')  # phenol
    batch = pp.collate([graph])

    with torch.no_grad():
        pred = model(**batch)

    n_bonds = len(graph['bond']) // 2
    assert pred.shape == (1, n_bonds, 2)
    # Predictions should be in reasonable BDE range (30-150 kcal/mol)
    assert pred[0, :, 0].min() > 30
    assert pred[0, :, 0].max() < 150
