#!/usr/bin/env python
"""
End-to-end Pipeline Benchmark: 2-step vs AIO.

Compares:
  1. 2-step pipeline: Model 1 (V3 or Hypergraph link pred) -> co-reactants -> Model 2 (T5v2) -> products
  2. AIO pipeline: DirectedHypergraphNet -> product embedding -> search product index

Metrics:
  - Quality: Recall@K, exact match, Tanimoto similarity to ground truth
  - Speed: latency per molecule, throughput
  - Diversity: structural diversity of candidate products

Usage:
    # Full benchmark (needs GPU + T5v2 model)
    python scripts/benchmark_pipeline.py --n-queries 200

    # AIO only (faster, no T5v2 needed)
    python scripts/benchmark_pipeline.py --method aio --n-queries 500

    # 2-step only
    python scripts/benchmark_pipeline.py --method 2step --n-queries 100

    # Output report
    python scripts/benchmark_pipeline.py -o docs/benchmark_report.md
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
# Ground Truth Loading
# =============================================================================

def load_test_ground_truth(data_dir: str = "Data/uspto") -> List[Dict]:
    """Load test reactions with ground truth products and co-reactants."""
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

        reactants = []
        for r in reactants_raw:
            mol = Chem.MolFromSmiles(r)
            if mol:
                reactants.append(Chem.MolToSmiles(mol))
        products = []
        for p in products_raw:
            mol = Chem.MolFromSmiles(p)
            if mol:
                products.append(Chem.MolToSmiles(mol))

        if len(reactants) >= 2 and len(products) >= 1:
            # Use first reactant as query, rest as ground truth co-reactants
            reactions.append({
                "query": reactants[0],
                "co_reactants": reactants[1:],
                "products": products,
                "all_reactants": reactants,
                "rxn_class": int(row.get("rxn_class", row.get("class", 0))),
            })

    return reactions


# =============================================================================
# Metrics
# =============================================================================

def compute_tanimoto(smi_a: str, smi_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def compute_product_metrics(predicted_products: List[str],
                            gt_products: List[str]) -> Dict:
    """Compute product prediction quality metrics."""
    if not gt_products or not predicted_products:
        return {"exact_match": 0.0, "best_tanimoto": 0.0, "avg_tanimoto": 0.0}

    gt_set = set(gt_products)

    # Exact match
    exact = any(p in gt_set for p in predicted_products)

    # Best Tanimoto similarity between any predicted and any GT product
    best_sim = 0.0
    all_sims = []
    for pred in predicted_products:
        for gt in gt_products:
            sim = compute_tanimoto(pred, gt)
            best_sim = max(best_sim, sim)
            all_sims.append(sim)

    return {
        "exact_match": 1.0 if exact else 0.0,
        "best_tanimoto": best_sim,
        "avg_tanimoto": float(np.mean(all_sims)) if all_sims else 0.0,
    }


def compute_diversity(smiles_list: List[str]) -> float:
    """Compute structural diversity: 1 - mean pairwise Tanimoto."""
    if len(smiles_list) < 2:
        return 0.0

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if len(fps) < 2:
        return 0.0

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    return 1.0 - float(np.mean(sims))


# =============================================================================
# Pipeline Runners
# =============================================================================

def run_2step_pipeline(queries: List[Dict], data_dir: str,
                       reactant_method: str = "hypergraph",
                       top_k: int = 5) -> Tuple[List[Dict], Dict]:
    """Run 2-step pipeline: Model 1 + Model 2."""
    from model_reactions.reaction_predictor import ReactionPredictor
    from model_reactions.config import ReactionPredictorConfig, ReactantPredictorConfig

    config = ReactionPredictorConfig(
        reactant_method=reactant_method,
        reactant_top_k=top_k,
        reactant_config=ReactantPredictorConfig(data_dir=data_dir),
    )
    config.product_config.num_beams = 1
    config.product_config.num_return_sequences = 1

    predictor = ReactionPredictor(config=config)
    predictor.load()

    results = []
    latencies = []

    for i, q in enumerate(queries):
        t0 = time.time()
        co_reactants, products, scores = predictor.get_valid_actions(q["query"], top_k=top_k)
        latency = time.time() - t0
        latencies.append(latency)

        metrics = compute_product_metrics(products, q["products"])
        diversity = compute_diversity(products)

        results.append({
            "query": q["query"],
            "gt_products": q["products"],
            "gt_co_reactants": q["co_reactants"],
            "predicted_products": products,
            "predicted_co_reactants": co_reactants,
            "scores": scores.tolist() if len(scores) > 0 else [],
            "latency": latency,
            "n_products": len(products),
            **metrics,
            "diversity": diversity,
        })

        if (i + 1) % 50 == 0:
            print(f"  2-step: {i + 1}/{len(queries)} "
                  f"(exact_match={np.mean([r['exact_match'] for r in results]):.3f}, "
                  f"avg_latency={np.mean(latencies):.3f}s)")

    timing = {
        "avg_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "throughput": len(queries) / sum(latencies) if sum(latencies) > 0 else 0,
    }

    return results, timing


def run_aio_pipeline(queries: List[Dict], data_dir: str,
                     checkpoint_path: str, product_index_path: str,
                     top_k: int = 5) -> Tuple[List[Dict], Dict]:
    """Run AIO pipeline: DirectedHypergraphNet -> product index search."""
    import torch
    from hypergraph.hypergraph_neighbor_predictor import (
        HypergraphConfig, DirectedHypergraphNet, smiles_to_graph_v2, EDGE_FEAT_DIM
    )
    from scripts.build_product_index import ProductIndex

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"  Loading AIO model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = HypergraphConfig.medium()
    model = DirectedHypergraphNet(config).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load product index
    print(f"  Loading product index from {product_index_path}...")
    product_idx = ProductIndex(product_index_path)
    print(f"  Index: {product_idx.n_mols} products, dim={product_idx.emb_dim}")

    results = []
    latencies = []

    with torch.no_grad():
        for i, q in enumerate(queries):
            t0 = time.time()

            # Encode query molecule
            graph = smiles_to_graph_v2(q["query"])
            if graph is None:
                latencies.append(time.time() - t0)
                results.append({
                    "query": q["query"],
                    "gt_products": q["products"],
                    "gt_co_reactants": q["co_reactants"],
                    "predicted_products": [],
                    "n_products": 0,
                    "exact_match": 0.0,
                    "best_tanimoto": 0.0,
                    "avg_tanimoto": 0.0,
                    "diversity": 0.0,
                    "latency": latencies[-1],
                })
                continue

            af = graph["atom_features"].to(device)
            ei = graph["edge_index"].to(device)
            ef = graph["edge_features"].to(device)
            batch_idx = torch.zeros(af.size(0), dtype=torch.long, device=device)

            # Predict product embedding
            preds = model.predict_neighbors(af, ei, ef, batch_idx)
            pred_prod_emb = preds["product_emb"].cpu().numpy()[0]
            pred_prod_emb = pred_prod_emb / (np.linalg.norm(pred_prod_emb) + 1e-8)

            # Search product index
            search_results = product_idx.search(pred_prod_emb, top_k=top_k)
            predicted_products = [smi for smi, _ in search_results]

            latency = time.time() - t0
            latencies.append(latency)

            metrics = compute_product_metrics(predicted_products, q["products"])
            diversity = compute_diversity(predicted_products)

            results.append({
                "query": q["query"],
                "gt_products": q["products"],
                "gt_co_reactants": q["co_reactants"],
                "predicted_products": predicted_products,
                "scores": [s for _, s in search_results],
                "latency": latency,
                "n_products": len(predicted_products),
                **metrics,
                "diversity": diversity,
            })

            if (i + 1) % 100 == 0:
                print(f"  AIO: {i + 1}/{len(queries)} "
                      f"(exact_match={np.mean([r['exact_match'] for r in results]):.3f}, "
                      f"avg_latency={np.mean(latencies):.3f}s)")

    timing = {
        "avg_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "throughput": len(queries) / sum(latencies) if sum(latencies) > 0 else 0,
    }

    return results, timing


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_results: Dict[str, Tuple[List[Dict], Dict]],
                    n_queries: int) -> str:
    """Generate markdown benchmark report."""
    lines = []
    lines.append("# End-to-End Pipeline Benchmark")
    lines.append("")
    lines.append(f"Test queries: {n_queries} reactions from USPTO test split")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + " | ".join(all_results.keys()) + " |")
    sep = "|:-------|" + "|".join("------:" for _ in all_results) + "|"
    lines.append(sep)

    metrics_to_show = [
        ("Exact Match", "exact_match", ".3f"),
        ("Best Tanimoto", "best_tanimoto", ".3f"),
        ("Avg Tanimoto", "avg_tanimoto", ".3f"),
        ("Avg N Products", "n_products", ".1f"),
        ("Product Diversity", "diversity", ".3f"),
    ]

    for label, key, fmt in metrics_to_show:
        row = f"| {label} |"
        for name, (results, _) in all_results.items():
            val = np.mean([r[key] for r in results])
            row += f" {val:{fmt}} |"
        lines.append(row)

    # Timing
    lines.append("")
    timing_metrics = [
        ("Avg Latency (s)", "avg_latency", ".4f"),
        ("Median Latency (s)", "median_latency", ".4f"),
        ("P95 Latency (s)", "p95_latency", ".4f"),
        ("Throughput (mol/s)", "throughput", ".1f"),
    ]

    for label, key, fmt in timing_metrics:
        row = f"| {label} |"
        for name, (_, timing) in all_results.items():
            val = timing[key]
            row += f" {val:{fmt}} |"
        lines.append(row)

    lines.append("")

    # Per-method details
    for name, (results, timing) in all_results.items():
        lines.append(f"## {name}")
        lines.append("")

        exact_matches = [r["exact_match"] for r in results]
        best_tanimotos = [r["best_tanimoto"] for r in results]
        n_prods = [r["n_products"] for r in results]

        lines.append(f"- Exact match rate: {np.mean(exact_matches):.3f}")
        lines.append(f"- Best Tanimoto: mean={np.mean(best_tanimotos):.3f}, "
                      f"median={np.median(best_tanimotos):.3f}")
        lines.append(f"- Avg products per query: {np.mean(n_prods):.1f}")
        lines.append(f"- Product diversity: {np.mean([r['diversity'] for r in results]):.3f}")
        lines.append(f"- Timing: avg={timing['avg_latency']:.4f}s, "
                      f"median={timing['median_latency']:.4f}s, "
                      f"throughput={timing['throughput']:.1f} mol/s")

        # Tanimoto distribution
        lines.append("")
        for threshold in [0.5, 0.7, 0.9, 1.0]:
            count = sum(1 for t in best_tanimotos if t >= threshold)
            lines.append(f"  - Best Tanimoto >= {threshold}: "
                          f"{count}/{len(results)} ({count / len(results) * 100:.1f}%)")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    p.add_argument("--method", type=str, default="all",
                   choices=["2step", "aio", "all"])
    p.add_argument("--data-dir", type=str, default="Data/uspto")
    p.add_argument("--n-queries", type=int, default=200,
                   help="Number of test queries")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of candidates per query")
    p.add_argument("--reactant-method", type=str, default="hypergraph",
                   choices=["hypergraph", "fingerprint"])
    p.add_argument("--aio-checkpoint", type=str,
                   default="hypergraph/checkpoints/directed_predictor_best.pt")
    p.add_argument("--product-index", type=str,
                   default="Data/precomputed/aio_product_index.pkl")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading test ground truth...")
    reactions = load_test_ground_truth(args.data_dir)
    print(f"  {len(reactions)} test reactions")

    # Sample queries
    rng = np.random.RandomState(args.seed)
    n = min(args.n_queries, len(reactions))
    indices = rng.choice(len(reactions), size=n, replace=False)
    queries = [reactions[i] for i in indices]
    print(f"  Sampled {n} queries")

    all_results = {}

    if args.method in ("2step", "all"):
        print("\n" + "=" * 60)
        print("Running 2-step Pipeline")
        print("=" * 60)
        results_2step, timing_2step = run_2step_pipeline(
            queries, args.data_dir,
            reactant_method=args.reactant_method,
            top_k=args.top_k,
        )
        all_results["2-Step"] = (results_2step, timing_2step)

    if args.method in ("aio", "all"):
        if os.path.exists(args.aio_checkpoint) and os.path.exists(args.product_index):
            print("\n" + "=" * 60)
            print("Running AIO Pipeline")
            print("=" * 60)
            results_aio, timing_aio = run_aio_pipeline(
                queries, args.data_dir,
                checkpoint_path=args.aio_checkpoint,
                product_index_path=args.product_index,
                top_k=args.top_k,
            )
            all_results["AIO"] = (results_aio, timing_aio)
        else:
            print(f"Skipping AIO: checkpoint or index not found")
            if not os.path.exists(args.aio_checkpoint):
                print(f"  Missing: {args.aio_checkpoint}")
            if not os.path.exists(args.product_index):
                print(f"  Missing: {args.product_index}")

    if not all_results:
        print("No results to report.")
        return

    # Generate report
    report = generate_report(all_results, n)
    print("\n" + report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved report to {args.output}")

    # Save raw results
    if args.output:
        pkl_path = args.output.replace(".md", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Saved raw results to {pkl_path}")


if __name__ == "__main__":
    main()
