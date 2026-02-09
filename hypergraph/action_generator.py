"""AIOActionGenerator: All-in-one directed hypergraph action generator.

Unlike the two-model pipeline (Model 1: co-reactants + Model 2: T5v2 products),
the AIO model predicts both co-reactants AND products in a single forward pass
using directed hypergraph neighbor prediction.

Provides the ActionGenerator interface for RL integration.
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from action_generator import ActionGenerator
from model_reactions.utils import canonicalize_smiles, tanimoto_from_mols


class AIOActionGenerator(ActionGenerator):
    """All-in-one action generator using DirectedHypergraphNet.

    Wraps DirectedHypergraphNeighborPredictor to provide the ActionGenerator
    interface for RL integration.
    """

    name = "aio"

    def __init__(self, checkpoint_path: str = None, data_dir: str = "Data/uspto",
                 device: str = 'auto', top_k: int = 10, max_index_mols: int = 10000,
                 filter_products: bool = True,
                 filter_min_tanimoto: float = 0.2, filter_max_mw_ratio: float = 1.3,
                 filter_max_mw_delta: float = 200.0, filter_min_mw_ratio: float = 0.5,
                 filter_reject_si: bool = True):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.device_str = device
        self.top_k = top_k
        self.max_index_mols = max_index_mols

        self.filter_products = filter_products
        self.filter_min_tanimoto = filter_min_tanimoto
        self.filter_max_mw_ratio = filter_max_mw_ratio
        self.filter_max_mw_delta = filter_max_mw_delta
        self.filter_min_mw_ratio = filter_min_mw_ratio
        self.filter_reject_si = filter_reject_si

        self._predictor = None
        self._loaded = False

    def load(self):
        """Lazy-load the directed hypergraph predictor."""
        if self._loaded:
            return

        print("=" * 60)
        print("Initializing AIO ActionGenerator (DirectedHypergraphNet)")
        print("=" * 60)
        start = time.time()

        from hypergraph.hypergraph_neighbor_predictor import DirectedHypergraphNeighborPredictor

        checkpoint = self.checkpoint_path
        if checkpoint is None:
            checkpoint = self._find_checkpoint()

        self._predictor = DirectedHypergraphNeighborPredictor(
            checkpoint_path=checkpoint,
            data_dir=self.data_dir,
            device=self.device_str,
            top_k=self.top_k,
            max_index_mols=self.max_index_mols,
        )

        elapsed = time.time() - start
        print(f"AIO ActionGenerator ready. Init time: {elapsed:.1f}s")
        print("=" * 60)
        self._loaded = True

    def _find_checkpoint(self) -> Optional[str]:
        """Find the directed hypergraph checkpoint."""
        search_paths = [
            "hypergraph/checkpoints/directed_predictor_best.pt",
            "hypergraph/checkpoints/directed_predictor_final.pt",
        ]
        for p in search_paths:
            if os.path.exists(p):
                return p
        print("Warning: No directed hypergraph checkpoint found. "
              f"Searched: {search_paths}")
        return None

    def get_valid_actions(self, mol, top_k: int = None) -> Tuple[List[str], List[str], np.ndarray]:
        """Get valid reaction actions for a molecule."""
        if not self._loaded:
            self.load()

        if top_k is None:
            top_k = self.top_k

        if isinstance(mol, str):
            mol_smiles = mol
        else:
            mol_smiles = Chem.MolToSmiles(mol)

        canon_mol = canonicalize_smiles(mol_smiles)
        if canon_mol is None:
            return [], [], np.array([], dtype=np.float32)

        products_raw, co_reactants_raw, scores_raw = self._predictor.predict_neighbors(canon_mol)

        reactant_mol = Chem.MolFromSmiles(canon_mol)
        if reactant_mol is None:
            return [], [], np.array([], dtype=np.float32)

        if not self.filter_products:
            co_reactants = []
            products = []
            scores = []
            for i, prod in enumerate(products_raw[:top_k]):
                canon_prod = canonicalize_smiles(prod)
                if canon_prod is None or canon_prod == canon_mol:
                    continue
                co_smi = co_reactants_raw[i] if i < len(co_reactants_raw) else ""
                products.append(canon_prod)
                co_reactants.append(co_smi)
                scores.append(float(scores_raw[i]) if i < len(scores_raw) else 0.0)
            return co_reactants, products, np.array(scores, dtype=np.float32)

        reactant_mw = Descriptors.ExactMolWt(reactant_mol)
        co_reactants = []
        products = []
        scores = []
        best_fallback = None
        fallback_tanimoto = -1.0

        for i, prod in enumerate(products_raw[:top_k]):
            canon_prod = canonicalize_smiles(prod)
            if canon_prod is None or canon_prod == canon_mol:
                continue

            prod_mol = Chem.MolFromSmiles(canon_prod)
            if prod_mol is None:
                continue

            co_smi = co_reactants_raw[i] if i < len(co_reactants_raw) else ""
            canon_co = canonicalize_smiles(co_smi) if co_smi else None
            if canon_co is not None and canon_prod == canon_co:
                continue

            tani = tanimoto_from_mols(reactant_mol, prod_mol)
            if tani > fallback_tanimoto:
                fallback_tanimoto = tani
                best_fallback = (canon_prod, co_smi, float(scores_raw[i]) if i < len(scores_raw) else 0.0)

            if tani < self.filter_min_tanimoto:
                continue

            prod_mw = Descriptors.ExactMolWt(prod_mol)
            if abs(prod_mw - reactant_mw) > self.filter_max_mw_delta:
                continue
            if reactant_mw > 0:
                mw_ratio = prod_mw / reactant_mw
                if mw_ratio > self.filter_max_mw_ratio:
                    continue
                if mw_ratio < self.filter_min_mw_ratio:
                    continue

            if self.filter_reject_si:
                has_si = any(atom.GetAtomicNum() == 14 for atom in prod_mol.GetAtoms())
                if has_si:
                    continue

            products.append(canon_prod)
            co_reactants.append(co_smi)
            scores.append(float(scores_raw[i]) if i < len(scores_raw) else 0.0)

        if not products and best_fallback is not None:
            products = [best_fallback[0]]
            co_reactants = [best_fallback[1]]
            scores = [best_fallback[2]]

        return co_reactants, products, np.array(scores, dtype=np.float32)

    def get_valid_actions_batch(self, mols, top_k: int = None) -> list:
        """Batch get valid actions (serial for now)."""
        return [self.get_valid_actions(mol, top_k=top_k) for mol in mols]

    def get_stats(self):
        """Get performance statistics."""
        return {
            'loaded': self._loaded,
            'type': 'aio_directed_hypergraph',
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AIO ActionGenerator test")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    args = parser.parse_args()

    predictor = AIOActionGenerator(
        device=args.device,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )

    test_mols = ["CCO", "c1ccccc1Br", "CC(=O)O", "c1ccccc1N"]

    for mol in test_mols:
        print(f"\n{'='*60}")
        print(f"Input: {mol}")
        print(f"{'='*60}")

        co_reactants, products, scores = predictor.get_valid_actions(mol, top_k=args.top_k)

        if not co_reactants:
            print("  No reactions found.")
            continue

        for i, (co, prod, score) in enumerate(zip(co_reactants, products, scores)):
            print(f"  [{i+1}] {mol} + {co}")
            print(f"       -> {prod}  (score={score:.4f})")
