"""
Hybrid Reaction Predictor: AIO initial retrieval + V3 re-ranking.

Uses AIO (DirectedHypergraphNet) for fast initial co-reactant retrieval,
then V3 (HypergraphLinkPredictorV3) with full NCN + CF-NCN features to
re-rank the top-N candidates. Product prediction uses AIO's embedding search.

Benchmark results (50K USPTO):
  - AIO alone:     MRR=0.013, Recall@10=0.031
  - AIO + V3 rerank: MRR=0.025, Recall@10=0.061 (~2x improvement)

Usage:
    from model_reactions.hybrid_predictor import HybridReactionPredictor

    predictor = HybridReactionPredictor(device='cuda')
    co_reactants, products, scores = predictor.get_valid_actions("CCO", top_k=5)
"""

import os
import sys
import time
import pickle
import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.utils import canonicalize_smiles, tanimoto_from_mols


class HybridReactionPredictor:
    """
    Hybrid predictor: AIO embedding retrieval + V3 pair-wise re-ranking.

    Two-stage co-reactant retrieval:
      Stage 1: AIO co_reactant_predictor -> fast embedding search -> top-N candidates
      Stage 2: V3 compute_pair_score with NCN + CF-NCN -> re-rank top-N

    Product prediction: AIO embedding search (same as pure AIO).
    """

    DEFAULT_AIO_CHECKPOINT = "hypergraph/checkpoints/directed_predictor_best.pt"
    DEFAULT_V3_CHECKPOINT = "model_reactions/checkpoints/cf_ncn_v2/hypergraph_link_v3_best.pt"
    DEFAULT_KNN_PATH = "Data/precomputed/knn_50k_k200.pkl"

    def __init__(self, aio_checkpoint: str = None, v3_checkpoint: str = None,
                 data_dir: str = "Data/uspto", device: str = 'auto',
                 top_k: int = 10, rerank_topn: int = 100,
                 knn_path: str = None,
                 filter_products: bool = True,
                 filter_min_tanimoto: float = 0.2,
                 filter_max_mw_ratio: float = 1.3,
                 filter_max_mw_delta: float = 200.0,
                 filter_min_mw_ratio: float = 0.5,
                 filter_reject_si: bool = True):
        self.aio_checkpoint = aio_checkpoint
        self.v3_checkpoint = v3_checkpoint
        self.data_dir = data_dir
        self.device_str = device
        self.top_k = top_k
        self.rerank_topn = rerank_topn
        self.knn_path = knn_path

        self.filter_products = filter_products
        self.filter_min_tanimoto = filter_min_tanimoto
        self.filter_max_mw_ratio = filter_max_mw_ratio
        self.filter_max_mw_delta = filter_max_mw_delta
        self.filter_min_mw_ratio = filter_min_mw_ratio
        self.filter_reject_si = filter_reject_si

        self._loaded = False

    def load(self):
        """Initialize both models and precompute all embeddings."""
        if self._loaded:
            return

        print("=" * 60)
        print("Initializing Hybrid ReactionPredictor (AIO + V3 Re-rank)")
        print("=" * 60)
        start = time.time()

        if self.device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device_str)

        # Find checkpoints
        aio_ckpt = self.aio_checkpoint or self._find_checkpoint(
            [self.DEFAULT_AIO_CHECKPOINT], "AIO")
        v3_ckpt = self.v3_checkpoint or self._find_checkpoint(
            [self.DEFAULT_V3_CHECKPOINT], "V3")

        # Load AIO model + predictor
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from hypergraph.hypergraph_neighbor_predictor import (
            DirectedHypergraphNeighborPredictor
        )
        print(f"\n[AIO] Loading DirectedHypergraphNeighborPredictor...")
        self._aio_predictor = DirectedHypergraphNeighborPredictor(
            checkpoint_path=aio_ckpt,
            data_dir=self.data_dir,
            device=self.device_str,
            top_k=max(self.top_k, self.rerank_topn),
            max_index_mols=50000,
        )

        # Load V3 model
        from model_reactions.link_prediction.hypergraph_link_predictor_v3 import (
            HypergraphLinkConfigV3, HypergraphLinkPredictorV3,
            smiles_to_rich_graph, build_adjacency, get_common_neighbors,
            _prepare_cf_cn_batch,
        )
        sys.modules["__main__"].HypergraphLinkConfigV3 = HypergraphLinkConfigV3

        print(f"\n[V3] Loading HypergraphLinkPredictorV3 from {v3_ckpt}...")
        ckpt = torch.load(v3_ckpt, map_location=self.device, weights_only=False)
        self.v3_config = ckpt["config"]
        self.v3_model = HypergraphLinkPredictorV3(self.v3_config).to(self.device)
        self.v3_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.v3_model.eval()

        # Save references for V3 inference
        self._smiles_to_rich_graph = smiles_to_rich_graph
        self._build_adjacency = build_adjacency
        self._get_common_neighbors = get_common_neighbors
        self._prepare_cf_cn_batch = _prepare_cf_cn_batch

        # Build train adjacency for NCN/CF-NCN
        self._build_train_adjacency()

        # Load kNN graph for CF-NCN
        self._load_knn_graph()

        # Precompute V3 embeddings for database molecules
        self._precompute_v3_embeddings()

        elapsed = time.time() - start
        print(f"\nHybrid ReactionPredictor ready. Init time: {elapsed:.1f}s")
        print("=" * 60)
        self._loaded = True

    def _find_checkpoint(self, search_paths: List[str], name: str) -> Optional[str]:
        for p in search_paths:
            if os.path.exists(p):
                return p
        print(f"Warning: No {name} checkpoint found. Searched: {search_paths}")
        return None

    def _build_train_adjacency(self):
        """Build adjacency from train edges for NCN/CF-NCN features."""
        import pandas as pd
        print("Building train adjacency...")
        train_csv = os.path.join(self.data_dir, "train.csv")
        if not os.path.exists(train_csv):
            self.adj = {}
            return

        smiles_to_idx = {}
        for i, smi in enumerate(self._aio_predictor.reactant_smiles):
            smiles_to_idx[smi] = i

        df = pd.read_csv(train_csv)
        train_edges = []
        for _, row in df.iterrows():
            rxn = str(row.get("rxn_smiles", ""))
            if ">>" not in rxn:
                continue
            parts = rxn.split(">>")
            reactants = []
            for r in parts[0].split("."):
                mol = Chem.MolFromSmiles(r)
                if mol:
                    smi = Chem.MolToSmiles(mol)
                    idx = smiles_to_idx.get(smi)
                    if idx is not None:
                        reactants.append(idx)
            for ia in range(len(reactants)):
                for ib in range(ia + 1, len(reactants)):
                    train_edges.append((reactants[ia], reactants[ib]))

        self.adj = self._build_adjacency(train_edges)
        self.smiles_to_idx = smiles_to_idx
        print(f"  {len(train_edges)} train edges, adj covers {len(self.adj)} molecules")

    def _load_knn_graph(self):
        """Load precomputed kNN graph for CF-NCN features."""
        knn_path = self.knn_path or self.DEFAULT_KNN_PATH
        self.knn_graph = {}
        if os.path.exists(knn_path):
            print(f"Loading kNN graph from {knn_path}...")
            with open(knn_path, "rb") as f:
                knn_data = pickle.load(f)
            self.knn_graph = knn_data["knn_graph"]
            print(f"  {len(self.knn_graph)} molecules in kNN graph")
        else:
            print(f"  kNN graph not found at {knn_path}, CF-NCN features will be empty")

    def _precompute_v3_embeddings(self):
        """Precompute V3 link embeddings for all database molecules."""
        all_smiles = self._aio_predictor.reactant_smiles
        n_mols = len(all_smiles)
        v3_emb_dim = self.v3_config.mol_embedding_dim

        print(f"Precomputing V3 embeddings for {n_mols} molecules...")
        self.v3_embeddings = torch.zeros(n_mols, v3_emb_dim, device=self.device)
        batch_size = 256

        with torch.no_grad():
            for start in range(0, n_mols, batch_size):
                end = min(start + batch_size, n_mols)
                batch_smi = all_smiles[start:end]

                atom_parts, edge_parts, bond_parts, batch_parts = [], [], [], []
                offset = 0
                valid_indices = []

                for j, smi in enumerate(batch_smi):
                    graph = self._smiles_to_rich_graph(smi)
                    if graph is None:
                        continue
                    n = graph["n_atoms"]
                    af = torch.tensor(graph["atom_features"], dtype=torch.float)
                    ei = torch.tensor(graph["edge_index"], dtype=torch.long)
                    bf = torch.tensor(graph["bond_features"], dtype=torch.float)
                    atom_parts.append(af)
                    if ei.numel() > 0:
                        edge_parts.append(ei + offset)
                    bond_parts.append(bf)
                    batch_parts.append(torch.full((n,), len(valid_indices), dtype=torch.long))
                    offset += n
                    valid_indices.append(start + j)

                if not valid_indices:
                    continue

                af_t = torch.cat(atom_parts, dim=0).to(self.device)
                ei_t = (torch.cat(edge_parts, dim=0).t().contiguous().to(self.device)
                        if edge_parts else torch.zeros(2, 0, dtype=torch.long, device=self.device))
                bf_t = torch.cat(bond_parts, dim=0).to(self.device)
                b_t = torch.cat(batch_parts, dim=0).to(self.device)

                _, link_emb = self.v3_model.encode_molecule(af_t, ei_t, bf_t, b_t)
                for k, idx in enumerate(valid_indices):
                    self.v3_embeddings[idx] = link_emb[k]

                if start > 0 and start % 10000 == 0:
                    print(f"  V3 encoded {start}/{n_mols}")

        print(f"  V3 embeddings ready: {n_mols} x {v3_emb_dim}")

    def _v3_rerank(self, query_smi: str, candidate_indices: List[int],
                   candidate_scores: List[float]) -> List[Tuple[int, float]]:
        """Re-rank candidate indices using V3 full pair scoring."""
        query_idx = self.smiles_to_idx.get(query_smi)
        if query_idx is None or not candidate_indices:
            return list(zip(candidate_indices, candidate_scores))

        N = len(candidate_indices)
        max_cn = getattr(self.v3_config, 'max_cn', 32)
        max_bridge = getattr(self.v3_config, 'max_bridge', 16)
        use_ncn = getattr(self.v3_config, 'use_ncn', False)
        use_cf_ncn = getattr(self.v3_config, 'use_cf_ncn', False)
        cf_ncn_use_count = getattr(self.v3_config, 'cf_ncn_use_count', False)
        v3_emb_dim = self.v3_config.mol_embedding_dim

        idx_a = torch.tensor([query_idx] * N, dtype=torch.long)
        idx_b = torch.tensor(candidate_indices, dtype=torch.long)

        emb_a = self.v3_embeddings[idx_a.tolist()]
        emb_b = self.v3_embeddings[candidate_indices]

        # NCN features
        cn_emb = None
        if use_ncn and self.v3_model.ncn_aggregator is not None:
            cn_parts = []
            cn_mask_parts = []
            for j in range(N):
                cn_list = self._get_common_neighbors(
                    self.adj, query_idx, candidate_indices[j], max_cn)
                cn_embs_j = torch.zeros(max_cn, v3_emb_dim, device=self.device)
                cn_mask_j = torch.zeros(max_cn, dtype=torch.bool, device=self.device)
                for ci, cn_idx in enumerate(cn_list):
                    if 0 <= cn_idx < self.v3_embeddings.shape[0]:
                        cn_embs_j[ci] = self.v3_embeddings[cn_idx]
                        cn_mask_j[ci] = True
                cn_parts.append(cn_embs_j)
                cn_mask_parts.append(cn_mask_j)
            cn_embs_batch = torch.stack(cn_parts, dim=0)
            cn_mask_batch = torch.stack(cn_mask_parts, dim=0)
            cn_emb = self.v3_model.ncn_aggregator(cn_embs_batch, cn_mask_batch)

        # CF-NCN features
        cf_bridge_emb = None
        cf_bridge_count = None
        if use_cf_ncn and self.v3_model.cf_ncn_aggregator is not None and self.knn_graph:
            bridge_embs, bridge_mask, bridge_weights = self._prepare_cf_cn_batch(
                idx_a, idx_b, self.knn_graph, self.adj,
                self.v3_embeddings, max_bridge, self.device)
            cf_bridge_result = self.v3_model.cf_ncn_aggregator(
                bridge_embs, bridge_mask, bridge_weights,
                max_bridge, emb_a, emb_b)
            cf_bridge_emb = cf_bridge_result[0]
            if cf_ncn_use_count and cf_bridge_result[1] is not None:
                cf_bridge_count = cf_bridge_result[1]

        # Full pair score
        with torch.no_grad():
            v3_scores = self.v3_model.compute_pair_score(
                emb_a, emb_b, cn_emb=cn_emb,
                cf_bridge_emb=cf_bridge_emb,
                cf_bridge_count=cf_bridge_count
            ).cpu().numpy()

        # Sort by V3 score
        order = np.argsort(v3_scores)[::-1]
        return [(candidate_indices[j], float(v3_scores[j])) for j in order]

    def get_valid_actions(self, mol, top_k: int = None) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Get valid reaction actions using hybrid AIO + V3 retrieval.

        Args:
            mol: RDKit Mol object or SMILES string
            top_k: Number of actions to return

        Returns:
            co_reactants: List of co-reactant SMILES
            products: List of product SMILES
            scores: np.ndarray of scores
        """
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

        # Stage 1: AIO initial retrieval (more candidates than needed)
        aio_top_k = max(self.rerank_topn, top_k * 5)
        products_raw, co_reactants_raw, scores_raw = self._aio_predictor.predict_neighbors(canon_mol)

        # V3 re-rank co-reactants if we have enough
        if len(co_reactants_raw) > 1 and hasattr(self, 'smiles_to_idx'):
            # Get indices for co-reactant candidates
            co_indices = []
            co_smi_list = []
            for smi in co_reactants_raw[:self.rerank_topn]:
                if isinstance(smi, tuple):
                    smi = smi[0]
                idx = self.smiles_to_idx.get(smi)
                if idx is not None:
                    co_indices.append(idx)
                    co_smi_list.append(smi)

            if co_indices:
                reranked = self._v3_rerank(
                    canon_mol, co_indices,
                    [float(scores_raw[i]) if i < len(scores_raw) else 0.0
                     for i in range(len(co_indices))])

                # Rebuild co-reactant list in V3-reranked order
                idx_to_smi = {idx: smi for idx, smi in zip(co_indices, co_smi_list)}
                co_reactants_raw = [idx_to_smi[idx] for idx, _ in reranked]
                scores_raw = np.array([score for _, score in reranked], dtype=np.float32)
                # Keep products_raw from AIO (they correspond to the original order)
                # We'll re-pair them below

        reactant_mol = Chem.MolFromSmiles(canon_mol)
        if reactant_mol is None:
            return [], [], np.array([], dtype=np.float32)

        if not self.filter_products:
            co_reactants = []
            products = []
            scores = []
            for i, co_smi in enumerate(co_reactants_raw[:top_k]):
                if isinstance(co_smi, tuple):
                    co_smi = co_smi[0]
                prod = products_raw[i] if i < len(products_raw) else ""
                if isinstance(prod, tuple):
                    prod = prod[0]
                canon_prod = canonicalize_smiles(prod)
                if canon_prod is None or canon_prod == canon_mol:
                    continue
                products.append(canon_prod)
                co_reactants.append(co_smi)
                scores.append(float(scores_raw[i]) if i < len(scores_raw) else 0.0)
            return co_reactants, products, np.array(scores, dtype=np.float32)

        # Filtered path
        reactant_mw = Descriptors.ExactMolWt(reactant_mol)
        co_reactants = []
        products = []
        scores = []
        best_fallback = None
        fallback_tanimoto = -1.0

        for i, prod in enumerate(products_raw[:top_k * 3]):
            if isinstance(prod, tuple):
                prod = prod[0]
            canon_prod = canonicalize_smiles(prod)
            if canon_prod is None or canon_prod == canon_mol:
                continue

            prod_mol = Chem.MolFromSmiles(canon_prod)
            if prod_mol is None:
                continue

            co_smi = co_reactants_raw[i] if i < len(co_reactants_raw) else ""
            if isinstance(co_smi, tuple):
                co_smi = co_smi[0]
            canon_co = canonicalize_smiles(co_smi) if co_smi else None
            if canon_co is not None and canon_prod == canon_co:
                continue

            tani = tanimoto_from_mols(reactant_mol, prod_mol)
            score_val = float(scores_raw[i]) if i < len(scores_raw) else 0.0
            if tani > fallback_tanimoto:
                fallback_tanimoto = tani
                best_fallback = (canon_prod, co_smi, score_val)

            if tani < self.filter_min_tanimoto:
                continue

            prod_mw = Descriptors.ExactMolWt(prod_mol)
            if abs(prod_mw - reactant_mw) > self.filter_max_mw_delta:
                continue
            if reactant_mw > 0:
                mw_ratio = prod_mw / reactant_mw
                if mw_ratio > self.filter_max_mw_ratio or mw_ratio < self.filter_min_mw_ratio:
                    continue

            if self.filter_reject_si:
                has_si = any(atom.GetAtomicNum() == 14 for atom in prod_mol.GetAtoms())
                if has_si:
                    continue

            products.append(canon_prod)
            co_reactants.append(co_smi)
            scores.append(score_val)

            if len(products) >= top_k:
                break

        if not products and best_fallback is not None:
            products = [best_fallback[0]]
            co_reactants = [best_fallback[1]]
            scores = [best_fallback[2]]

        return co_reactants, products, np.array(scores, dtype=np.float32)

    def get_valid_actions_batch(self, mols, top_k: int = None) -> List[Tuple[List[str], List[str], np.ndarray]]:
        """Batch get valid actions."""
        return [self.get_valid_actions(mol, top_k=top_k) for mol in mols]

    def get_stats(self):
        return {
            'loaded': self._loaded,
            'type': 'hybrid_aio_v3_rerank',
            'rerank_topn': self.rerank_topn,
        }
