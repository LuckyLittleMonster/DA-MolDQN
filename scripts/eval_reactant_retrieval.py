#!/usr/bin/env python
"""
Evaluate co-reactant retrieval quality across different methods.

Metrics: Recall@K (K=10,50,100), MRR (Mean Reciprocal Rank), NDCG@K
Methods: V3 Link Predictor, AIO (DirectedHypergraphNet) Stage 1, Fingerprint baseline

Given a query molecule from the test split, rank all database molecules
and check if the ground-truth co-reactant(s) appear in the top-K.

Usage:
    # Evaluate V3 link predictor
    python scripts/eval_reactant_retrieval.py --method v3 \\
        --checkpoint model_reactions/checkpoints/cf_ncn_v2/hypergraph_link_v3_best.pt

    # Evaluate AIO directed hypergraph
    python scripts/eval_reactant_retrieval.py --method aio \\
        --checkpoint hypergraph/checkpoints/directed_predictor_best.pt

    # Evaluate fingerprint baseline
    python scripts/eval_reactant_retrieval.py --method fingerprint

    # Evaluate all methods
    python scripts/eval_reactant_retrieval.py --method all

    # Quick test with fewer queries
    python scripts/eval_reactant_retrieval.py --method v3 --max-queries 100
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Metrics
# =============================================================================

def recall_at_k(ranked_list: List[int], ground_truth: set, k: int) -> float:
    """Fraction of ground truth items found in top-k."""
    if not ground_truth:
        return 0.0
    found = sum(1 for item in ranked_list[:k] if item in ground_truth)
    return found / len(ground_truth)


def mrr(ranked_list: List[int], ground_truth: set) -> float:
    """Mean Reciprocal Rank: 1/rank of first correct item."""
    for i, item in enumerate(ranked_list):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_list: List[int], ground_truth: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # +2 because rank starts at 1, log2(1)=0
    # Ideal DCG: all relevant items at top
    n_rel = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(rankings: List[Tuple[List[int], set]],
                        k_values: List[int] = [10, 50, 100]) -> Dict:
    """Compute all metrics over a list of (ranked_list, ground_truth) pairs."""
    results = {}
    mrr_values = []

    for k in k_values:
        recall_values = []
        ndcg_values = []
        for ranked, gt in rankings:
            recall_values.append(recall_at_k(ranked, gt, k))
            ndcg_values.append(ndcg_at_k(ranked, gt, k))
        results[f"Recall@{k}"] = float(np.mean(recall_values))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_values))

    for ranked, gt in rankings:
        mrr_values.append(mrr(ranked, gt))
    results["MRR"] = float(np.mean(mrr_values))
    results["n_queries"] = len(rankings)

    return results


# =============================================================================
# Data Loading
# =============================================================================

def load_test_reactions(data_dir: str = "Data/uspto") -> List[Dict]:
    """Load test reactions and extract query-mol -> ground-truth co-reactants."""
    import pandas as pd
    from rdkit import Chem

    test_csv = os.path.join(data_dir, "test.csv")
    df = pd.read_csv(test_csv)

    reactions = []
    for _, row in df.iterrows():
        rxn = str(row.get("rxn_smiles", ""))
        if ">>" not in rxn:
            continue
        parts = rxn.split(">>")
        reactants_raw = parts[0].split(".")
        products_raw = parts[1].split(".") if len(parts) > 1 else []

        # Canonicalize
        reactants = []
        for r in reactants_raw:
            mol = Chem.MolFromSmiles(r)
            if mol is not None:
                reactants.append(Chem.MolToSmiles(mol))

        products = []
        for p in products_raw:
            mol = Chem.MolFromSmiles(p)
            if mol is not None:
                products.append(Chem.MolToSmiles(mol))

        if len(reactants) >= 2:
            reactions.append({
                "reactants": reactants,
                "products": products,
                "rxn_class": int(row.get("rxn_class", row.get("class", 0))),
            })

    return reactions


def build_query_pairs(reactions: List[Dict]) -> List[Tuple[str, set]]:
    """Build (query_smiles, {ground_truth_co_reactant_smiles}) pairs.

    For each reaction with reactants [A, B, ...], create queries:
      - query=A, gt={B, ...}
      - query=B, gt={A, ...}
    """
    # Aggregate across all reactions so a molecule can have multiple GT partners
    query_to_gt = defaultdict(set)
    for rxn in reactions:
        reactants = rxn["reactants"]
        for i, r in enumerate(reactants):
            co_reactants = set(reactants[j] for j in range(len(reactants)) if j != i)
            query_to_gt[r].update(co_reactants)

    return [(q, gt) for q, gt in query_to_gt.items()]


# =============================================================================
# Method Implementations
# =============================================================================

def evaluate_fingerprint(data_dir: str, query_pairs: List[Tuple[str, set]],
                         max_queries: int = 0) -> Dict:
    """Evaluate fingerprint Tanimoto baseline."""
    from model_reactions.reactant_predictor import (
        ReactionDatabase, FingerprintReactantPredictor, ReactantPredictorConfig
    )

    print("Loading database...")
    cache_path = os.path.join(data_dir, "reaction_db_cache.pkl")
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    print("Building fingerprint index...")
    config = ReactantPredictorConfig(fp_length=2048, fp_radius=2)
    fp_pred = FingerprintReactantPredictor(config, strategy="direct")
    fp_pred.db = db
    fp_pred._compute_fingerprints()

    smiles_to_idx = db.smiles_to_idx
    n_mols = len(db.reactant_smiles)

    if max_queries > 0:
        query_pairs = query_pairs[:max_queries]

    print(f"Evaluating {len(query_pairs)} queries...")
    rankings = []

    for i, (query_smi, gt_smiles) in enumerate(query_pairs):
        if i > 0 and i % 500 == 0:
            print(f"  {i}/{len(query_pairs)}")

        query_idx = smiles_to_idx.get(query_smi)
        if query_idx is None:
            continue

        gt_indices = set()
        for gs in gt_smiles:
            idx = smiles_to_idx.get(gs)
            if idx is not None:
                gt_indices.add(idx)
        if not gt_indices:
            continue

        # Compute Tanimoto similarity to all molecules
        fp_q = fp_pred.fp_matrix[query_idx]
        fp_q_norm = fp_q / (np.linalg.norm(fp_q) + 1e-8)

        sims = fp_pred.fp_matrix_norm @ fp_q_norm
        sims[query_idx] = -1.0  # exclude self

        ranked = np.argsort(sims)[::-1].tolist()
        rankings.append((ranked, gt_indices))

    return compute_all_metrics(rankings)


def evaluate_v3(data_dir: str, checkpoint_path: str, query_pairs: List[Tuple[str, set]],
                max_queries: int = 0) -> Dict:
    """Evaluate V3 link predictor for co-reactant retrieval."""
    import torch
    from model_reactions.reactant_predictor import ReactionDatabase
    from model_reactions.link_prediction.hypergraph_link_predictor_v3 import (
        HypergraphLinkConfigV3, HypergraphLinkPredictorV3,
        smiles_to_rich_graph,
    )

    # Patch for pickle compat
    sys.modules["__main__"].HypergraphLinkConfigV3 = HypergraphLinkConfigV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load database
    print("Loading database...")
    cache_path = os.path.join(data_dir, "reaction_db_cache.pkl")
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    smiles_to_idx = db.smiles_to_idx
    all_smiles = db.reactant_smiles
    n_mols = len(all_smiles)

    # Load model
    print(f"Loading V3 checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = HypergraphLinkPredictorV3(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Precompute embeddings for all database molecules
    print(f"Encoding {n_mols} database molecules...")
    emb_dim = config.mol_embedding_dim
    all_embeddings = torch.zeros(n_mols, emb_dim, device=device)

    batch_size = 256
    with torch.no_grad():
        for start in range(0, n_mols, batch_size):
            end = min(start + batch_size, n_mols)
            batch_smi = all_smiles[start:end]

            atom_parts, edge_parts, bond_parts, batch_parts = [], [], [], []
            offset = 0
            valid_indices = []

            for j, smi in enumerate(batch_smi):
                graph = smiles_to_rich_graph(smi)
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

            af_t = torch.cat(atom_parts, dim=0).to(device)
            ei_t = (torch.cat(edge_parts, dim=0).t().contiguous().to(device)
                    if edge_parts else torch.zeros(2, 0, dtype=torch.long, device=device))
            bf_t = torch.cat(bond_parts, dim=0).to(device)
            b_t = torch.cat(batch_parts, dim=0).to(device)

            _, link_emb = model.encode_molecule(af_t, ei_t, bf_t, b_t)
            for k, idx in enumerate(valid_indices):
                all_embeddings[idx] = link_emb[k]

            if start > 0 and start % 5000 == 0:
                print(f"  Encoded {start}/{n_mols}")

    # Normalize embeddings
    norms = all_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    all_embeddings = all_embeddings / norms

    # Evaluate retrieval
    if max_queries > 0:
        query_pairs = query_pairs[:max_queries]

    print(f"Evaluating {len(query_pairs)} queries...")
    rankings = []

    with torch.no_grad():
        for i, (query_smi, gt_smiles) in enumerate(query_pairs):
            if i > 0 and i % 500 == 0:
                print(f"  {i}/{len(query_pairs)}")

            query_idx = smiles_to_idx.get(query_smi)
            if query_idx is None:
                continue

            gt_indices = set()
            for gs in gt_smiles:
                idx = smiles_to_idx.get(gs)
                if idx is not None:
                    gt_indices.add(idx)
            if not gt_indices:
                continue

            # Score query vs all database
            query_emb = all_embeddings[query_idx].unsqueeze(0)  # (1, dim)
            # Simple dot product scoring (without NCN/topo features)
            scores = (all_embeddings @ query_emb.squeeze()).cpu().numpy()
            scores[query_idx] = -float("inf")

            ranked = np.argsort(scores)[::-1].tolist()
            rankings.append((ranked, gt_indices))

    return compute_all_metrics(rankings)


def evaluate_aio(data_dir: str, checkpoint_path: str, query_pairs: List[Tuple[str, set]],
                 max_queries: int = 0) -> Dict:
    """Evaluate AIO (DirectedHypergraphNet) Stage 1 for co-reactant retrieval."""
    import torch
    from model_reactions.reactant_predictor import ReactionDatabase

    sys.path.insert(0, str(Path(__file__).parent.parent / "hypergraph"))
    from hypergraph.hypergraph_neighbor_predictor import (
        HypergraphConfig, DirectedHypergraphNet, smiles_to_graph_v2, ATOM_FEAT_DIM, EDGE_FEAT_DIM
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load database
    print("Loading database...")
    cache_path = os.path.join(data_dir, "reaction_db_cache.pkl")
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    smiles_to_idx = db.smiles_to_idx
    all_smiles = db.reactant_smiles
    n_mols = len(all_smiles)

    # Load AIO model
    print(f"Loading AIO checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = HypergraphConfig.medium()
    model = DirectedHypergraphNet(config).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    emb_dim = config.mol_embedding_dim

    # Precompute all molecule embeddings
    print(f"Encoding {n_mols} database molecules...")
    all_embeddings = torch.zeros(n_mols, emb_dim, device=device)
    batch_size = 256

    with torch.no_grad():
        for start in range(0, n_mols, batch_size):
            end = min(start + batch_size, n_mols)
            batch_smi = all_smiles[start:end]

            af_parts, ei_parts, ef_parts, b_parts = [], [], [], []
            offset = 0
            valid_indices = []

            for j, smi in enumerate(batch_smi):
                graph = smiles_to_graph_v2(smi)
                if graph is None:
                    continue
                af = graph["atom_features"]
                ei = graph["edge_index"]
                ef = graph["edge_features"]
                n = af.size(0)
                af_parts.append(af)
                if ei.numel() > 0:
                    ei_parts.append(ei + offset)
                ef_parts.append(ef)
                b_parts.append(torch.full((n,), len(valid_indices), dtype=torch.long))
                offset += n
                valid_indices.append(start + j)

            if not valid_indices:
                continue

            af_t = torch.cat(af_parts, dim=0).to(device)
            ei_t = (torch.cat(ei_parts, dim=1).to(device)
                    if ei_parts else torch.zeros(2, 0, dtype=torch.long, device=device))
            ef_t = (torch.cat(ef_parts, dim=0).to(device)
                    if ef_parts else torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float, device=device))
            b_t = torch.cat(b_parts, dim=0).to(device)

            emb = model.encode_molecule(af_t, ei_t, ef_t, b_t)
            for k, idx in enumerate(valid_indices):
                all_embeddings[idx] = emb[k]

            if start > 0 and start % 5000 == 0:
                print(f"  Encoded {start}/{n_mols}")

    # For AIO Stage 1: predict co-reactant embedding, then search in database
    # The co_reactant_predictor maps mol_h -> predicted co-reactant embedding
    # We need to project embeddings through input_proj + role_embedding + co_reactant_predictor

    print("Computing co-reactant prediction embeddings...")
    # For each query mol, compute: input_proj(emb) + role_tail -> co_reactant_predictor -> predicted_emb
    # Then search in raw embeddings (or projected embeddings)

    # Normalize raw embeddings for cosine similarity search
    norms = all_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    all_embeddings_norm = all_embeddings / norms

    if max_queries > 0:
        query_pairs = query_pairs[:max_queries]

    print(f"Evaluating {len(query_pairs)} queries (AIO Stage 1)...")
    rankings = []

    with torch.no_grad():
        for i, (query_smi, gt_smiles) in enumerate(query_pairs):
            if i > 0 and i % 500 == 0:
                print(f"  {i}/{len(query_pairs)}")

            query_idx = smiles_to_idx.get(query_smi)
            if query_idx is None:
                continue

            gt_indices = set()
            for gs in gt_smiles:
                idx = smiles_to_idx.get(gs)
                if idx is not None:
                    gt_indices.add(idx)
            if not gt_indices:
                continue

            # Use co-reactant predictor to get predicted co-reactant embedding
            query_emb = all_embeddings[query_idx].unsqueeze(0)  # (1, emb_dim)
            mol_h = model._proj_with_role(query_emb, model.ROLE_TAIL)
            pred_co_emb = model.co_reactant_predictor(mol_h)  # (1, emb_dim)

            # Normalize
            pred_co_emb = pred_co_emb / pred_co_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Search in raw embeddings
            scores = (all_embeddings_norm @ pred_co_emb.squeeze()).cpu().numpy()
            scores[query_idx] = -float("inf")

            ranked = np.argsort(scores)[::-1].tolist()
            rankings.append((ranked, gt_indices))

    return compute_all_metrics(rankings)


def evaluate_aio_v3_rerank(data_dir: str, aio_checkpoint: str, v3_checkpoint: str,
                            query_pairs: List[Tuple[str, set]],
                            max_queries: int = 0,
                            rerank_topn: int = 100,
                            fusion_alpha: float = 0.7) -> Dict:
    """Hybrid retrieval: AIO Stage 1 initial retrieval + V3 full re-ranking.

    1. AIO co_reactant_predictor -> retrieve top-N candidates (fast embedding search)
    2. V3 compute_pair_score with NCN + CF-NCN features -> re-rank top-N

    fusion_alpha: weight for AIO scores in score fusion.
                  0.0 = pure V3 re-rank, 1.0 = pure AIO, 0.5 = equal blend.
    """
    import torch
    from model_reactions.reactant_predictor import ReactionDatabase
    from model_reactions.link_prediction.hypergraph_link_predictor_v3 import (
        HypergraphLinkConfigV3, HypergraphLinkPredictorV3,
        smiles_to_rich_graph, build_adjacency, get_common_neighbors,
        _prepare_cf_cn_batch,
    )
    from hypergraph.hypergraph_neighbor_predictor import (
        HypergraphConfig, DirectedHypergraphNet, smiles_to_graph_v2, EDGE_FEAT_DIM
    )

    sys.modules["__main__"].HypergraphLinkConfigV3 = HypergraphLinkConfigV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load database
    print("Loading database...")
    cache_path = os.path.join(data_dir, "reaction_db_cache.pkl")
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    smiles_to_idx = db.smiles_to_idx
    all_smiles = db.reactant_smiles
    n_mols = len(all_smiles)

    # Build adjacency from train edges for NCN/CF-NCN
    import pandas as pd
    from rdkit import Chem
    print("Building train adjacency for NCN/CF-NCN features...")
    train_csv = os.path.join(data_dir, "train.csv")
    train_edges = []
    df_train = pd.read_csv(train_csv)
    for _, row in df_train.iterrows():
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

    adj = build_adjacency(train_edges)
    print(f"  {len(train_edges)} train edges, adj covers {len(adj)} molecules")

    # Load kNN graph
    knn_path = os.path.join("Data/precomputed",
                            "knn_50k_k200.pkl" if "50k" in data_dir or data_dir.endswith("uspto")
                            else "knn_full_k200.pkl")
    knn_graph = {}
    if os.path.exists(knn_path):
        print(f"Loading kNN graph from {knn_path}...")
        with open(knn_path, "rb") as f:
            knn_data = pickle.load(f)
        knn_graph = knn_data["knn_graph"]
        print(f"  {len(knn_graph)} molecules in kNN graph")
    else:
        print(f"  WARNING: kNN graph not found at {knn_path}, CF-NCN features will be empty")

    # Load AIO model
    print(f"Loading AIO model from {aio_checkpoint}...")
    aio_ckpt = torch.load(aio_checkpoint, map_location=device, weights_only=False)
    aio_config = HypergraphConfig.medium()
    aio_model = DirectedHypergraphNet(aio_config).to(device)
    if "model_state_dict" in aio_ckpt:
        aio_model.load_state_dict(aio_ckpt["model_state_dict"], strict=False)
    aio_model.eval()

    # Load V3 model
    print(f"Loading V3 model from {v3_checkpoint}...")
    v3_ckpt = torch.load(v3_checkpoint, map_location=device, weights_only=False)
    v3_config = v3_ckpt["config"]
    v3_model = HypergraphLinkPredictorV3(v3_config).to(device)
    v3_model.load_state_dict(v3_ckpt["model_state_dict"], strict=False)
    v3_model.eval()

    aio_emb_dim = aio_config.mol_embedding_dim
    v3_emb_dim = v3_config.mol_embedding_dim

    # Precompute AIO embeddings for all database molecules
    print(f"Encoding {n_mols} molecules with AIO...")
    aio_embeddings = torch.zeros(n_mols, aio_emb_dim, device=device)
    batch_size = 256

    with torch.no_grad():
        for start in range(0, n_mols, batch_size):
            end = min(start + batch_size, n_mols)
            batch_smi = all_smiles[start:end]

            af_parts, ei_parts, ef_parts, b_parts = [], [], [], []
            offset = 0
            valid_indices = []

            for j, smi in enumerate(batch_smi):
                graph = smiles_to_graph_v2(smi)
                if graph is None:
                    continue
                af = graph["atom_features"]
                ei = graph["edge_index"]
                ef = graph["edge_features"]
                n = af.size(0)
                af_parts.append(af)
                if ei.numel() > 0:
                    ei_parts.append(ei + offset)
                ef_parts.append(ef)
                b_parts.append(torch.full((n,), len(valid_indices), dtype=torch.long))
                offset += n
                valid_indices.append(start + j)

            if not valid_indices:
                continue

            af_t = torch.cat(af_parts, dim=0).to(device)
            ei_t = (torch.cat(ei_parts, dim=1).to(device)
                    if ei_parts else torch.zeros(2, 0, dtype=torch.long, device=device))
            ef_t = (torch.cat(ef_parts, dim=0).to(device)
                    if ef_parts else torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float, device=device))
            b_t = torch.cat(b_parts, dim=0).to(device)

            emb = aio_model.encode_molecule(af_t, ei_t, ef_t, b_t)
            for k, idx in enumerate(valid_indices):
                aio_embeddings[idx] = emb[k]

            if start > 0 and start % 5000 == 0:
                print(f"  AIO encoded {start}/{n_mols}")

    # Normalize AIO embeddings for cosine search
    aio_norms = aio_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    aio_embeddings_norm = aio_embeddings / aio_norms

    # Precompute V3 link embeddings for all database molecules
    print(f"Encoding {n_mols} molecules with V3...")
    v3_link_embeddings = torch.zeros(n_mols, v3_emb_dim, device=device)

    with torch.no_grad():
        for start in range(0, n_mols, batch_size):
            end = min(start + batch_size, n_mols)
            batch_smi = all_smiles[start:end]

            atom_parts, edge_parts, bond_parts, batch_parts = [], [], [], []
            offset = 0
            valid_indices = []

            for j, smi in enumerate(batch_smi):
                graph = smiles_to_rich_graph(smi)
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

            af_t = torch.cat(atom_parts, dim=0).to(device)
            ei_t = (torch.cat(edge_parts, dim=0).t().contiguous().to(device)
                    if edge_parts else torch.zeros(2, 0, dtype=torch.long, device=device))
            bf_t = torch.cat(bond_parts, dim=0).to(device)
            b_t = torch.cat(batch_parts, dim=0).to(device)

            _, link_emb = v3_model.encode_molecule(af_t, ei_t, bf_t, b_t)
            for k, idx in enumerate(valid_indices):
                v3_link_embeddings[idx] = link_emb[k]

            if start > 0 and start % 5000 == 0:
                print(f"  V3 encoded {start}/{n_mols}")

    # Determine V3 model capabilities
    max_cn = getattr(v3_config, 'max_cn', 32)
    max_bridge = getattr(v3_config, 'max_bridge', 16)
    use_ncn = getattr(v3_config, 'use_ncn', False)
    use_cf_ncn = getattr(v3_config, 'use_cf_ncn', False)
    cf_ncn_use_count = getattr(v3_config, 'cf_ncn_use_count', False)
    print(f"  V3 features: NCN={use_ncn}, CF-NCN={use_cf_ncn}, count={cf_ncn_use_count}")

    # Evaluate hybrid retrieval
    if max_queries > 0:
        query_pairs = query_pairs[:max_queries]

    print(f"Evaluating {len(query_pairs)} queries (AIO top-{rerank_topn} + V3 full re-rank, alpha={fusion_alpha})...")
    rankings = []

    with torch.no_grad():
        for i, (query_smi, gt_smiles) in enumerate(query_pairs):
            if i > 0 and i % 500 == 0:
                print(f"  {i}/{len(query_pairs)}")

            query_idx = smiles_to_idx.get(query_smi)
            if query_idx is None:
                continue

            gt_indices = set()
            for gs in gt_smiles:
                idx = smiles_to_idx.get(gs)
                if idx is not None:
                    gt_indices.add(idx)
            if not gt_indices:
                continue

            # Step 1: AIO initial retrieval (top-N)
            query_emb = aio_embeddings[query_idx].unsqueeze(0)
            mol_h = aio_model._proj_with_role(query_emb, aio_model.ROLE_TAIL)
            pred_co_emb = aio_model.co_reactant_predictor(mol_h)
            pred_co_emb = pred_co_emb / pred_co_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

            aio_scores = (aio_embeddings_norm @ pred_co_emb.squeeze()).cpu().numpy()
            aio_scores[query_idx] = -float("inf")

            # Get top-N candidate indices from AIO
            top_n_indices = np.argsort(aio_scores)[::-1][:rerank_topn].tolist()

            # Step 2: V3 full scoring with NCN + CF-NCN features
            N = len(top_n_indices)
            idx_a = torch.tensor([query_idx] * N, dtype=torch.long)
            idx_b = torch.tensor(top_n_indices, dtype=torch.long)

            emb_a = v3_link_embeddings[idx_a.tolist()]  # (N, dim)
            emb_b = v3_link_embeddings[top_n_indices]  # (N, dim)

            # NCN features
            cn_emb = None
            if use_ncn and v3_model.ncn_aggregator is not None:
                cn_parts = []
                cn_mask_parts = []
                for j in range(N):
                    a = query_idx
                    b = top_n_indices[j]
                    cn_list = get_common_neighbors(adj, a, b, max_cn)
                    cn_embs_j = torch.zeros(max_cn, v3_emb_dim, device=device)
                    cn_mask_j = torch.zeros(max_cn, dtype=torch.bool, device=device)
                    for ci, cn_idx in enumerate(cn_list):
                        if 0 <= cn_idx < v3_link_embeddings.shape[0]:
                            cn_embs_j[ci] = v3_link_embeddings[cn_idx]
                            cn_mask_j[ci] = True
                    cn_parts.append(cn_embs_j)
                    cn_mask_parts.append(cn_mask_j)
                cn_embs_batch = torch.stack(cn_parts, dim=0)  # (N, max_cn, dim)
                cn_mask_batch = torch.stack(cn_mask_parts, dim=0)  # (N, max_cn)
                cn_emb = v3_model.ncn_aggregator(cn_embs_batch, cn_mask_batch)  # (N, dim)

            # CF-NCN features
            cf_bridge_emb = None
            cf_bridge_count = None
            if use_cf_ncn and v3_model.cf_ncn_aggregator is not None and knn_graph:
                bridge_embs, bridge_mask, bridge_weights = _prepare_cf_cn_batch(
                    idx_a, idx_b, knn_graph, adj, v3_link_embeddings, max_bridge, device)
                cf_bridge_result = v3_model.cf_ncn_aggregator(
                    bridge_embs, bridge_mask, bridge_weights,
                    max_bridge, emb_a, emb_b)  # returns (agg, count_or_None)
                cf_bridge_emb = cf_bridge_result[0]  # (N, dim)
                if cf_ncn_use_count and cf_bridge_result[1] is not None:
                    cf_bridge_count = cf_bridge_result[1]  # (N, 1)

            # Compute full pair scores
            v3_scores = v3_model.compute_pair_score(
                emb_a, emb_b, cn_emb=cn_emb,
                cf_bridge_emb=cf_bridge_emb,
                cf_bridge_count=cf_bridge_count
            ).cpu().numpy()

            # Score fusion
            if fusion_alpha > 0 and fusion_alpha < 1.0:
                aio_cand_scores = np.array([aio_scores[j] for j in top_n_indices])
                def _minmax(x):
                    mn, mx = x.min(), x.max()
                    return (x - mn) / (mx - mn + 1e-8)
                aio_norm = _minmax(aio_cand_scores)
                v3_norm = _minmax(v3_scores)
                combined = fusion_alpha * aio_norm + (1 - fusion_alpha) * v3_norm
            elif fusion_alpha >= 1.0:
                combined = np.array([aio_scores[j] for j in top_n_indices])
            else:
                combined = v3_scores

            reranked_order = np.argsort(combined)[::-1]
            ranked = [top_n_indices[j] for j in reranked_order]

            # Append remaining molecules (not in top-N) in AIO order
            top_n_set = set(top_n_indices)
            remaining = [idx for idx in np.argsort(aio_scores)[::-1].tolist()
                         if idx not in top_n_set]
            ranked.extend(remaining)

            rankings.append((ranked, gt_indices))

    return compute_all_metrics(rankings)


# =============================================================================
# Report
# =============================================================================

def print_results(results: Dict[str, Dict], output_path: Optional[str] = None):
    """Print results as formatted table."""
    lines = []
    lines.append("# Co-Reactant Retrieval Evaluation")
    lines.append("")

    # Summary table
    k_values = [10, 50, 100]
    header = "| Method | MRR |"
    for k in k_values:
        header += f" Recall@{k} | NDCG@{k} |"
    header += " N queries |"
    lines.append(header)

    sep = "|:-------|------:|"
    for k in k_values:
        sep += "------:|------:|"
    sep += "------:|"
    lines.append(sep)

    for method, metrics in sorted(results.items()):
        row = f"| {method} | {metrics['MRR']:.4f} |"
        for k in k_values:
            row += f" {metrics.get(f'Recall@{k}', 0):.4f} | {metrics.get(f'NDCG@{k}', 0):.4f} |"
        row += f" {metrics.get('n_queries', 0)} |"
        lines.append(row)

    lines.append("")

    text = "\n".join(lines)
    print(text)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text + "\n")
        print(f"\nSaved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate co-reactant retrieval")
    p.add_argument("--method", type=str, default="all",
                   choices=["v3", "aio", "fingerprint", "aio_v3_rerank", "all"],
                   help="Method to evaluate")
    p.add_argument("--data-dir", type=str, default="Data/uspto")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint path")
    p.add_argument("--v3-checkpoint", type=str,
                   default="model_reactions/checkpoints/cf_ncn_v2/hypergraph_link_v3_best.pt",
                   help="V3 checkpoint (when --method all)")
    p.add_argument("--aio-checkpoint", type=str,
                   default="hypergraph/checkpoints/directed_predictor_best.pt",
                   help="AIO checkpoint (when --method all)")
    p.add_argument("--max-queries", type=int, default=0,
                   help="Max queries to evaluate (0 = all)")
    p.add_argument("--rerank-topn", type=int, default=100,
                   help="Number of AIO candidates to re-rank with V3 (for aio_v3_rerank)")
    p.add_argument("--fusion-alpha", type=float, default=0.0,
                   help="Score fusion weight: 0=pure V3, 1=pure AIO, 0.5=equal blend")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output markdown file")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading test reactions...")
    reactions = load_test_reactions(args.data_dir)
    query_pairs = build_query_pairs(reactions)
    print(f"  {len(reactions)} test reactions -> {len(query_pairs)} unique query molecules")

    results = {}

    if args.method in ("fingerprint", "all"):
        print("\n" + "=" * 60)
        print("Evaluating: Fingerprint (Tanimoto)")
        print("=" * 60)
        t0 = time.time()
        results["Fingerprint"] = evaluate_fingerprint(
            args.data_dir, query_pairs, args.max_queries)
        print(f"  Time: {time.time() - t0:.1f}s")
        print(f"  MRR={results['Fingerprint']['MRR']:.4f}, "
              f"Recall@10={results['Fingerprint']['Recall@10']:.4f}")

    if args.method in ("v3", "all"):
        ckpt = args.checkpoint if args.method == "v3" else args.v3_checkpoint
        if ckpt and os.path.exists(ckpt):
            print("\n" + "=" * 60)
            print("Evaluating: V3 Link Predictor")
            print("=" * 60)
            t0 = time.time()
            results["V3 Link Pred"] = evaluate_v3(
                args.data_dir, ckpt, query_pairs, args.max_queries)
            print(f"  Time: {time.time() - t0:.1f}s")
            print(f"  MRR={results['V3 Link Pred']['MRR']:.4f}, "
                  f"Recall@10={results['V3 Link Pred']['Recall@10']:.4f}")
        else:
            print(f"Skipping V3: checkpoint not found at {ckpt}")

    if args.method in ("aio", "all"):
        ckpt = args.checkpoint if args.method == "aio" else args.aio_checkpoint
        if ckpt and os.path.exists(ckpt):
            print("\n" + "=" * 60)
            print("Evaluating: AIO Stage 1")
            print("=" * 60)
            t0 = time.time()
            results["AIO Stage 1"] = evaluate_aio(
                args.data_dir, ckpt, query_pairs, args.max_queries)
            print(f"  Time: {time.time() - t0:.1f}s")
            print(f"  MRR={results['AIO Stage 1']['MRR']:.4f}, "
                  f"Recall@10={results['AIO Stage 1']['Recall@10']:.4f}")
        else:
            print(f"Skipping AIO: checkpoint not found at {ckpt}")

    if args.method in ("aio_v3_rerank", "all"):
        aio_ckpt = args.aio_checkpoint
        v3_ckpt = args.v3_checkpoint
        if (aio_ckpt and os.path.exists(aio_ckpt) and
                v3_ckpt and os.path.exists(v3_ckpt)):
            print("\n" + "=" * 60)
            print(f"Evaluating: AIO + V3 Re-rank (top-{args.rerank_topn})")
            print("=" * 60)
            t0 = time.time()
            results[f"AIO+V3 Rerank@{args.rerank_topn}"] = evaluate_aio_v3_rerank(
                args.data_dir, aio_ckpt, v3_ckpt, query_pairs,
                args.max_queries, args.rerank_topn, args.fusion_alpha)
            key = f"AIO+V3 Rerank@{args.rerank_topn}"
            print(f"  Time: {time.time() - t0:.1f}s")
            print(f"  MRR={results[key]['MRR']:.4f}, "
                  f"Recall@10={results[key]['Recall@10']:.4f}")
        else:
            missing = []
            if not (aio_ckpt and os.path.exists(aio_ckpt)):
                missing.append(f"AIO: {aio_ckpt}")
            if not (v3_ckpt and os.path.exists(v3_ckpt)):
                missing.append(f"V3: {v3_ckpt}")
            print(f"Skipping AIO+V3 Rerank: checkpoints not found ({', '.join(missing)})")

    if results:
        print("\n" + "=" * 60)
        print_results(results, args.output)

    # Save raw results as pickle
    if args.output:
        pkl_path = args.output.replace(".md", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved raw results to {pkl_path}")


if __name__ == "__main__":
    main()
