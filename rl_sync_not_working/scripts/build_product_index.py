#!/usr/bin/env python
"""
Build product embedding index for AIO (DirectedHypergraphNet) model.

Extracts embeddings from the DirectedHypergraphNet encoder for all known
products (and optionally reactants), then builds a kNN index for fast retrieval.

Two index types:
  1. numpy: simple cosine similarity search (no extra dependencies)
  2. faiss: FAISS IVF index for large-scale retrieval (requires faiss-gpu/faiss-cpu)

Output:
  - Data/precomputed/aio_product_index.pkl: {embeddings, smiles, metadata}
  - Data/precomputed/aio_reactant_index.pkl: same for reactants

Usage:
    # Build with default AIO checkpoint
    python scripts/build_product_index.py

    # Build with specific checkpoint and FAISS
    python scripts/build_product_index.py \\
        --checkpoint hypergraph/checkpoints/directed_predictor_best.pt \\
        --index-type faiss --output-dir Data/precomputed

    # Build only reactant index (for co-reactant retrieval)
    python scripts/build_product_index.py --reactants-only

    # Limit to first N molecules (for testing)
    python scripts/build_product_index.py --max-mols 5000
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# =============================================================================
# Embedding Extraction
# =============================================================================

def extract_embeddings(model, smiles_list: List[str], device: torch.device,
                       batch_size: int = 256, use_v2_graph: bool = True,
                       verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings from DirectedHypergraphNet encoder.

    Args:
        model: DirectedHypergraphNet model (eval mode)
        smiles_list: list of SMILES to encode
        device: torch device
        batch_size: encoding batch size
        use_v2_graph: use smiles_to_graph_v2 (rich features, for AIO)

    Returns:
        embeddings: (N_valid, emb_dim) numpy array, L2-normalized
        valid_smiles: list of SMILES that were successfully encoded
    """
    from hypergraph.hypergraph_neighbor_predictor import (
        smiles_to_graph_v2, ATOM_FEAT_DIM, EDGE_FEAT_DIM
    )

    model.eval()
    all_embeddings = []
    valid_smiles = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            end = min(start + batch_size, len(smiles_list))
            batch = smiles_list[start:end]

            af_parts, ei_parts, ef_parts, b_parts = [], [], [], []
            offset = 0
            batch_valid = []

            for smi in batch:
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
                b_parts.append(torch.full((n,), len(batch_valid), dtype=torch.long))
                offset += n
                batch_valid.append(smi)

            if not batch_valid:
                continue

            af_t = torch.cat(af_parts, dim=0).to(device)
            ei_t = (torch.cat(ei_parts, dim=1).to(device)
                    if ei_parts else torch.zeros(2, 0, dtype=torch.long, device=device))
            ef_t = (torch.cat(ef_parts, dim=0).to(device)
                    if ef_parts else torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float, device=device))
            b_t = torch.cat(b_parts, dim=0).to(device)

            emb = model.encode_molecule(af_t, ei_t, ef_t, b_t)
            all_embeddings.append(emb.cpu().numpy())
            valid_smiles.extend(batch_valid)

            if verbose and start > 0 and start % 5000 == 0:
                print(f"  Encoded {start}/{len(smiles_list)} ({len(valid_smiles)} valid)")

    if not all_embeddings:
        return np.zeros((0, 0)), []

    embeddings = np.vstack(all_embeddings)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    return embeddings, valid_smiles


# =============================================================================
# Index Building
# =============================================================================

def build_numpy_index(embeddings: np.ndarray, smiles: List[str],
                      metadata: Optional[Dict] = None) -> Dict:
    """Build simple numpy-based index for cosine similarity search."""
    return {
        "embeddings": embeddings.astype(np.float32),
        "smiles": smiles,
        "n_mols": len(smiles),
        "emb_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "index_type": "numpy",
        "metadata": metadata or {},
    }


def build_faiss_index(embeddings: np.ndarray, smiles: List[str],
                      metadata: Optional[Dict] = None,
                      nlist: int = 100, nprobe: int = 10) -> Dict:
    """Build FAISS IVF index for fast approximate nearest neighbor search."""
    try:
        import faiss
    except ImportError:
        print("FAISS not available, falling back to numpy index")
        return build_numpy_index(embeddings, smiles, metadata)

    n, d = embeddings.shape
    emb_f32 = embeddings.astype(np.float32)

    if n < nlist * 10:
        # Too few vectors for IVF, use flat index
        index = faiss.IndexFlatIP(d)  # Inner product (cosine on L2-normalized vectors)
        index.add(emb_f32)
        print(f"  Built FAISS FlatIP index: {n} vectors, dim={d}")
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(emb_f32)
        index.add(emb_f32)
        index.nprobe = nprobe
        print(f"  Built FAISS IVF index: {n} vectors, dim={d}, nlist={nlist}, nprobe={nprobe}")

    return {
        "faiss_index": index,
        "smiles": smiles,
        "n_mols": len(smiles),
        "emb_dim": d,
        "index_type": "faiss",
        "metadata": metadata or {},
    }


# =============================================================================
# Search Interface
# =============================================================================

class ProductIndex:
    """Unified search interface for product/reactant embedding indices."""

    def __init__(self, index_path: str):
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.smiles = data["smiles"]
        self.n_mols = data["n_mols"]
        self.emb_dim = data["emb_dim"]
        self.index_type = data["index_type"]
        self.metadata = data.get("metadata", {})

        if self.index_type == "numpy":
            self.embeddings = data["embeddings"]
        elif self.index_type == "faiss":
            self.faiss_index = data["faiss_index"]
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: (emb_dim,) query vector (should be L2-normalized)
            top_k: number of results

        Returns:
            List of (smiles, score) tuples, sorted by score descending
        """
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.index_type == "numpy":
            scores = (self.embeddings @ query.squeeze()).astype(float)
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(self.smiles[i], float(scores[i])) for i in top_idx]
        else:
            scores, indices = self.faiss_index.search(query, top_k)
            results = []
            for j in range(top_k):
                idx = int(indices[0, j])
                if idx >= 0:
                    results.append((self.smiles[idx], float(scores[0, j])))
            return results

    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        queries = query_embeddings.astype(np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self.index_type == "numpy":
            scores = queries @ self.embeddings.T  # (Q, N)
            all_results = []
            for i in range(scores.shape[0]):
                top_idx = np.argsort(scores[i])[::-1][:top_k]
                all_results.append([(self.smiles[j], float(scores[i, j])) for j in top_idx])
            return all_results
        else:
            scores, indices = self.faiss_index.search(queries, top_k)
            all_results = []
            for i in range(scores.shape[0]):
                results = []
                for j in range(top_k):
                    idx = int(indices[i, j])
                    if idx >= 0:
                        results.append((self.smiles[idx], float(scores[i, j])))
                all_results.append(results)
            return all_results


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Product Retrieval Evaluation
# =============================================================================

def evaluate_product_retrieval(model, product_index: 'ProductIndex',
                               data_dir: str, device: torch.device,
                               max_queries: int = 0) -> Dict:
    """Evaluate product retrieval: reactants -> predict product embedding -> search index.

    For each test reaction with reactants [A, B] -> product P:
      1. Encode A with AIO model
      2. Use predict_neighbors to get predicted product embedding
      3. Search product index for nearest neighbors
      4. Check if ground-truth P is in top-K

    Returns dict with Recall@K, MRR, exact match rate, avg Tanimoto.
    """
    import pandas as pd
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from hypergraph.hypergraph_neighbor_predictor import smiles_to_graph_v2, EDGE_FEAT_DIM

    test_csv = os.path.join(data_dir, "test.csv")
    df = pd.read_csv(test_csv)

    # Build (query_reactant, {ground_truth_products}) pairs
    queries = []
    for _, row in df.iterrows():
        rxn = str(row.get("rxn_smiles", ""))
        if ">>" not in rxn:
            continue
        parts = rxn.split(">>")
        reactants = []
        for r in parts[0].split("."):
            mol = Chem.MolFromSmiles(r)
            if mol:
                reactants.append(Chem.MolToSmiles(mol))
        products = []
        if len(parts) > 1:
            for p in parts[1].split("."):
                mol = Chem.MolFromSmiles(p)
                if mol:
                    products.append(Chem.MolToSmiles(mol))
        if reactants and products:
            queries.append({"query": reactants[0], "gt_products": set(products)})

    if max_queries > 0:
        queries = queries[:max_queries]

    # Build SMILES -> index lookup for product index
    prod_smi_set = set(product_index.smiles)

    k_values = [1, 5, 10, 50, 100]
    recall_accum = {k: [] for k in k_values}
    mrr_values = []
    exact_matches = []
    best_tanimotos = []
    n_evaluated = 0

    model.eval()
    with torch.no_grad():
        for i, q in enumerate(queries):
            if i > 0 and i % 500 == 0:
                print(f"  Product retrieval eval: {i}/{len(queries)}")

            # Check if any GT product is in the index
            gt_in_index = q["gt_products"] & prod_smi_set
            if not gt_in_index:
                continue

            # Encode query reactant
            graph = smiles_to_graph_v2(q["query"])
            if graph is None:
                continue

            af = graph["atom_features"].to(device)
            ei = graph["edge_index"].to(device)
            ef = graph["edge_features"].to(device)
            batch_idx = torch.zeros(af.size(0), dtype=torch.long, device=device)

            # Predict product embedding via AIO model
            preds = model.predict_neighbors(af, ei, ef, batch_idx)
            pred_prod_emb = preds["product_emb"].cpu().numpy()[0]
            pred_prod_emb = pred_prod_emb / (np.linalg.norm(pred_prod_emb) + 1e-8)

            # Search product index
            max_k = max(k_values)
            results = product_index.search(pred_prod_emb, top_k=max_k)
            ranked_smiles = [smi for smi, _ in results]

            # Compute metrics
            n_evaluated += 1

            # Exact match
            exact = any(smi in q["gt_products"] for smi in ranked_smiles)
            exact_matches.append(1.0 if exact else 0.0)

            # MRR
            rr = 0.0
            for rank, smi in enumerate(ranked_smiles):
                if smi in gt_in_index:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_values.append(rr)

            # Recall@K
            for k in k_values:
                found = sum(1 for smi in ranked_smiles[:k] if smi in gt_in_index)
                recall_accum[k].append(found / len(gt_in_index))

            # Best Tanimoto between top-5 predictions and GT products
            best_sim = 0.0
            for pred_smi in ranked_smiles[:5]:
                for gt_smi in q["gt_products"]:
                    try:
                        mol_a = Chem.MolFromSmiles(pred_smi)
                        mol_b = Chem.MolFromSmiles(gt_smi)
                        if mol_a and mol_b:
                            fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
                            fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
                            sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
                            best_sim = max(best_sim, sim)
                    except Exception:
                        pass
            best_tanimotos.append(best_sim)

    metrics = {
        "n_evaluated": n_evaluated,
        "n_queries": len(queries),
        "MRR": float(np.mean(mrr_values)) if mrr_values else 0.0,
        "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "best_tanimoto_mean": float(np.mean(best_tanimotos)) if best_tanimotos else 0.0,
        "best_tanimoto_median": float(np.median(best_tanimotos)) if best_tanimotos else 0.0,
    }
    for k in k_values:
        if recall_accum[k]:
            metrics[f"Recall@{k}"] = float(np.mean(recall_accum[k]))
        else:
            metrics[f"Recall@{k}"] = 0.0

    return metrics


def print_retrieval_report(metrics: Dict) -> str:
    """Format retrieval evaluation as markdown."""
    lines = []
    lines.append("# AIO Product Retrieval Evaluation")
    lines.append("")
    lines.append(f"Evaluated {metrics['n_evaluated']} queries "
                 f"(from {metrics['n_queries']} test reactions)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|:-------|------:|")
    lines.append(f"| MRR | {metrics['MRR']:.4f} |")
    lines.append(f"| Exact Match | {metrics['exact_match_rate']:.4f} |")
    for k in [1, 5, 10, 50, 100]:
        key = f"Recall@{k}"
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.4f} |")
    lines.append(f"| Best Tanimoto (top-5, mean) | {metrics['best_tanimoto_mean']:.4f} |")
    lines.append(f"| Best Tanimoto (top-5, median) | {metrics['best_tanimoto_median']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Build product embedding index for AIO model")
    p.add_argument("--checkpoint", type=str,
                   default="hypergraph/checkpoints/directed_predictor_best.pt",
                   help="AIO model checkpoint")
    p.add_argument("--data-dir", type=str, default="Data/uspto",
                   help="USPTO data directory")
    p.add_argument("--output-dir", type=str, default="Data/precomputed",
                   help="Output directory for index files")
    p.add_argument("--index-type", type=str, default="numpy",
                   choices=["numpy", "faiss"],
                   help="Index type")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-mols", type=int, default=0,
                   help="Max molecules to index (0 = all)")
    p.add_argument("--reactants-only", action="store_true",
                   help="Only build reactant index")
    p.add_argument("--products-only", action="store_true",
                   help="Only build product index")
    p.add_argument("--eval-retrieval", action="store_true",
                   help="Evaluate product retrieval quality after building index")
    p.add_argument("--eval-max-queries", type=int, default=0,
                   help="Max queries for retrieval eval (0 = all)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip index building, only run evaluation on existing index")
    return p.parse_args()


def main():
    args = parse_args()

    from hypergraph.hypergraph_neighbor_predictor import (
        HypergraphConfig, DirectedHypergraphNet
    )
    import pandas as pd
    from rdkit import Chem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading AIO model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = HypergraphConfig.medium()
    model = DirectedHypergraphNet(config).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded (emb_dim={config.mol_embedding_dim})")

    # --eval-only: skip index building, load existing index and run eval
    if args.eval_only:
        prod_path = os.path.join(args.output_dir, "aio_product_index.pkl")
        if not os.path.exists(prod_path):
            print(f"ERROR: Product index not found at {prod_path}. Build it first.")
            sys.exit(1)
        print(f"Loading existing product index from {prod_path}...")
        pidx = ProductIndex(prod_path)
        print(f"  Loaded: {pidx.n_mols} products, dim={pidx.emb_dim}")

        print("\nRunning product retrieval evaluation...")
        metrics = evaluate_product_retrieval(
            model, pidx, args.data_dir, device, max_queries=args.eval_max_queries)
        report = print_retrieval_report(metrics)
        print(report)

        # Save report
        os.makedirs("docs", exist_ok=True)
        report_path = "docs/product_retrieval_benchmark.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved report to {report_path}")

        # Save raw metrics
        metrics_path = os.path.join(args.output_dir, "product_retrieval_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved metrics to {metrics_path}")
        return

    # Collect unique products and reactants from all splits
    print(f"Loading molecules from {args.data_dir}...")
    reactant_set = set()
    product_set = set()

    for split in ["train", "val", "test"]:
        csv_path = os.path.join(args.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rxn = str(row.get("rxn_smiles", ""))
            if ">>" not in rxn:
                continue
            parts = rxn.split(">>")
            for r in parts[0].split("."):
                mol = Chem.MolFromSmiles(r)
                if mol:
                    reactant_set.add(Chem.MolToSmiles(mol))
            if len(parts) > 1:
                for p in parts[1].split("."):
                    mol = Chem.MolFromSmiles(p)
                    if mol:
                        product_set.add(Chem.MolToSmiles(mol))

    print(f"  Unique reactants: {len(reactant_set)}, products: {len(product_set)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build product index
    if not args.reactants_only:
        product_list = sorted(product_set)
        if args.max_mols > 0:
            product_list = product_list[:args.max_mols]

        print(f"\nEncoding {len(product_list)} products...")
        t0 = time.time()
        prod_embs, prod_valid = extract_embeddings(
            model, product_list, device, batch_size=args.batch_size)
        print(f"  Encoded {len(prod_valid)} products in {time.time() - t0:.1f}s")

        if args.index_type == "faiss":
            prod_index = build_faiss_index(prod_embs, prod_valid,
                                           metadata={"source": args.checkpoint, "type": "product"})
        else:
            prod_index = build_numpy_index(prod_embs, prod_valid,
                                           metadata={"source": args.checkpoint, "type": "product"})

        prod_path = os.path.join(args.output_dir, "aio_product_index.pkl")
        with open(prod_path, "wb") as f:
            pickle.dump(prod_index, f)
        print(f"  Saved product index to {prod_path} ({os.path.getsize(prod_path) / 1e6:.1f} MB)")

    # Build reactant index
    if not args.products_only:
        reactant_list = sorted(reactant_set)
        if args.max_mols > 0:
            reactant_list = reactant_list[:args.max_mols]

        print(f"\nEncoding {len(reactant_list)} reactants...")
        t0 = time.time()
        react_embs, react_valid = extract_embeddings(
            model, reactant_list, device, batch_size=args.batch_size)
        print(f"  Encoded {len(react_valid)} reactants in {time.time() - t0:.1f}s")

        if args.index_type == "faiss":
            react_index = build_faiss_index(react_embs, react_valid,
                                            metadata={"source": args.checkpoint, "type": "reactant"})
        else:
            react_index = build_numpy_index(react_embs, react_valid,
                                            metadata={"source": args.checkpoint, "type": "reactant"})

        react_path = os.path.join(args.output_dir, "aio_reactant_index.pkl")
        with open(react_path, "wb") as f:
            pickle.dump(react_index, f)
        print(f"  Saved reactant index to {react_path} ({os.path.getsize(react_path) / 1e6:.1f} MB)")

    # Optionally run product retrieval evaluation after building
    if args.eval_retrieval and not args.reactants_only:
        prod_path = os.path.join(args.output_dir, "aio_product_index.pkl")
        print(f"\nLoading product index for evaluation...")
        pidx = ProductIndex(prod_path)

        print("Running product retrieval evaluation...")
        metrics = evaluate_product_retrieval(
            model, pidx, args.data_dir, device, max_queries=args.eval_max_queries)
        report = print_retrieval_report(metrics)
        print(report)

        os.makedirs("docs", exist_ok=True)
        report_path = "docs/product_retrieval_benchmark.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved report to {report_path}")

        metrics_path = os.path.join(args.output_dir, "product_retrieval_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved metrics to {metrics_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
