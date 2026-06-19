"""High-level BDE prediction API for RL integration.

Usage:
    from bde_predictor.predict import BDEModel
    model = BDEModel('bde_predictor/weights/bde_db2_model3.npz', device='cuda')
    bdes, valids = model.predict_oh_bde(['c1ccc(O)cc1', 'Oc1ccc(O)cc1'])
"""
import torch
import numpy as np
from pathlib import Path
from .model import BDEPredictor
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
        self.model = BDEPredictor.from_npz(weights_path, device=self.device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def predict_smiles(self, smiles_list: list, batch_size: int = 256) -> list:
        """Predict BDE/BDFE for a list of SMILES strings.

        Returns list of dicts per molecule:
            bde: np.array [n_bonds] - BDE per bond (kcal/mol)
            bdfe: np.array [n_bonds] - BDFE per bond (kcal/mol)
            oh_bde: float or None - minimum O-H BDE (kcal/mol)
            oh_bond_indices: list[int] - O-H bond indices
            valid: bool
        """
        # Preprocess all molecules
        graphs = []
        graph_indices = []  # maps graph idx -> original smiles idx
        results = [None] * len(smiles_list)

        for i, smi in enumerate(smiles_list):
            try:
                g = self.preprocessor.process_smiles(smi)
                if len(g['bond']) == 0:
                    results[i] = {'bde': np.array([]), 'bdfe': np.array([]),
                                  'oh_bde': None, 'oh_bond_indices': [], 'valid': True}
                    continue
                graphs.append(g)
                graph_indices.append(i)
            except Exception:
                results[i] = {'bde': None, 'bdfe': None,
                              'oh_bde': None, 'oh_bond_indices': [], 'valid': False}

        # Batch inference
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch_gi = graph_indices[start:start + batch_size]
            batch = self.preprocessor.collate(batch_graphs, device=str(self.device))

            pred = self.model(**batch).cpu().float().numpy()  # [B, max_bonds, 2]

            for j, gi in enumerate(batch_gi):
                smi = smiles_list[gi]
                n_bonds = len(batch_graphs[j]['bond']) // 2
                bde_vals = pred[j, :n_bonds, 0]
                bdfe_vals = pred[j, :n_bonds, 1]

                oh_ids = self.preprocessor.get_oh_bond_indices(smi)
                oh_bde = None
                if oh_ids:
                    oh_bdes = [float(bde_vals[idx]) for idx in oh_ids if idx < n_bonds]
                    if oh_bdes:
                        oh_bde = min(oh_bdes)

                results[gi] = {
                    'bde': bde_vals,
                    'bdfe': bdfe_vals,
                    'oh_bde': oh_bde,
                    'oh_bond_indices': oh_ids,
                    'valid': True,
                }

        return results

    def predict_oh_bde(self, smiles_list: list, batch_size: int = 256) -> tuple:
        """Predict minimum O-H BDE for each molecule.

        Returns:
            bdes: list[float] (min O-H BDE per mol, 0.0 if invalid/no O-H)
            valids: list[bool]
        """
        results = self.predict_smiles(smiles_list, batch_size)
        bdes = []
        valids = []
        for r in results:
            if r['valid'] and r['oh_bde'] is not None:
                bdes.append(r['oh_bde'])
                valids.append(True)
            else:
                bdes.append(0.0)
                valids.append(False)
        return bdes, valids

    # --- Three-phase split for GIL-free overlap ---

    def prep_batch(self, smiles_list: list) -> tuple:
        """Phase 1 (Python, needs GIL): preprocess SMILES → GPU-ready batch.

        Returns: (batch, graphs, graph_indices, results)
            batch: dict of tensors on self.device (or None if no valid graphs)
            graphs: list of graph dicts
            graph_indices: maps graph position → original smiles index
            results: partial results list (pre-filled for trivial/failed mols)
        """
        graphs = []
        graph_indices = []
        results = [None] * len(smiles_list)

        for i, smi in enumerate(smiles_list):
            try:
                g = self.preprocessor.process_smiles(smi)
                if len(g['bond']) == 0:
                    results[i] = {'oh_bde': None, 'valid': True}
                    continue
                graphs.append(g)
                graph_indices.append(i)
            except Exception:
                results[i] = {'oh_bde': None, 'valid': False}

        batch = None
        if graphs:
            batch = self.preprocessor.collate(graphs, device=str(self.device))
        return batch, graphs, graph_indices, results

    @torch.no_grad()
    def forward_batch(self, batch):
        """Phase 2 (GPU, GIL-free during kernel): run MPNN forward only.

        Returns: raw prediction numpy array [B, max_bonds, 2], or None.
        """
        if batch is None:
            return None
        return self.model(**batch).cpu().float().numpy()

    def postprocess_oh_bde(self, pred, graphs, graph_indices, results, smiles_list):
        """Phase 3 (Python, needs GIL): extract O-H BDE from raw predictions.

        Returns: (bdes, valids) lists.
        """
        if pred is not None:
            for j, gi in enumerate(graph_indices):
                smi = smiles_list[gi]
                n_bonds = len(graphs[j]['bond']) // 2
                bde_vals = pred[j, :n_bonds, 0]
                oh_ids = self.preprocessor.get_oh_bond_indices(smi)
                oh_bde = None
                if oh_ids:
                    oh_bdes = [float(bde_vals[idx]) for idx in oh_ids if idx < n_bonds]
                    if oh_bdes:
                        oh_bde = min(oh_bdes)
                results[gi] = {'oh_bde': oh_bde, 'valid': True}

        bdes = []
        valids = []
        for r in results:
            if r is not None and r.get('valid') and r.get('oh_bde') is not None:
                bdes.append(r['oh_bde'])
                valids.append(True)
            else:
                bdes.append(0.0)
                valids.append(False)
        return bdes, valids
