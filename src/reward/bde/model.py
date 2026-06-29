"""High-level BDE prediction API for RL integration.

Usage:
    from src.reward.bde.model import BDEModel
    model = BDEModel('src/reward/bde/weights/alfabet.npz', device='cuda')
    bdes, valids = model.predict_oh_bde(['c1ccc(O)cc1', 'Oc1ccc(O)cc1'])
"""
import torch
from pathlib import Path
from .net import BDENet
from .preprocessor import BDEPreprocessor

_DEFAULT_PP = str(
    Path(__file__).resolve().parent.parent /
    'BDE-db2/Example-BDE-prediction/model_3_tfrecords_multi_halo_cfc/preprocessor.json')


class BDEModel:
    """BDE prediction model with preprocessing and batch inference."""

    def __init__(self, weights_path: str, preprocessor_path: str = None,
                 device: str = 'cpu', dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.preprocessor = BDEPreprocessor(preprocessor_path or _DEFAULT_PP)
        self.model = BDENet.from_npz(weights_path, device=self.device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def predict_oh_bde(self, smiles_list: list, batch_size: int = 256) -> tuple:
        """Predict the minimum O-H BDE for each molecule.

        Returns:
            bdes:   list[float] - min O-H BDE per mol (kcal/mol), 0.0 if invalid / no O-H
            valids: list[bool]
        """
        n = len(smiles_list)
        oh_bde = [None] * n
        valid = [False] * n

        # Preprocess once per molecule. process_smiles already records the O-H bond
        # indices in the graph, so the O-H BDE read-off needs no second SMILES parse.
        graphs = []
        graph_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                g = self.preprocessor.process_smiles(smi)
            except Exception:
                continue  # unparseable -> valid stays False
            valid[i] = True
            if len(g['bond']) == 0:
                continue  # valid but bondless -> no O-H BDE
            graphs.append(g)
            graph_indices.append(i)

        # Batched MPNN forward; keep only the BDE channel, read off the O-H bonds.
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch_gi = graph_indices[start:start + batch_size]
            batch = self.preprocessor.collate(batch_graphs, device=str(self.device))
            bde = self.model(**batch)[..., 0].cpu().float().numpy()  # [B, max_bonds]

            for j, gi in enumerate(batch_gi):
                n_bonds = len(batch_graphs[j]['bond']) // 2
                bde_vals = bde[j, :n_bonds]
                oh_ids = [k for k in batch_graphs[j]['oh_bond_indices'] if k < n_bonds]
                if oh_ids:
                    oh_bde[gi] = min(float(bde_vals[k]) for k in oh_ids)

        bdes = [oh_bde[i] if (valid[i] and oh_bde[i] is not None) else 0.0
                for i in range(n)]
        valids = [bool(valid[i] and oh_bde[i] is not None) for i in range(n)]
        return bdes, valids
