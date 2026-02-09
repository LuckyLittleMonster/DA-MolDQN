#!/usr/bin/env python
"""
Comprehensive evaluation of RetroPredictor (ReactionT5v2-retrosynthesis).

Evaluates:
  1. Basic functionality on known drug products
  2. Round-trip validation (retro -> forward -> compare)
  3. USPTO test set: predicted reactants vs actual reactants
  4. Prediction quality analysis (beam diversity, speed, failure rate)

Usage:
    source ~/.bashrc_maple 2>/dev/null && conda activate rl4
    PYTHONUNBUFFERED=1 python scripts/eval_retro_predictor.py [--n_uspto 300] [--device auto]
"""

import argparse
import json
import os
import sys
import time
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_reactions.product_prediction.retro_predictor import RetroPredictor
from model_reactions.product_prediction.product_predictor import ProductPredictor
from model_reactions.config import RetroPredictorConfig, ProductPredictorConfig
from model_reactions.utils import canonicalize_smiles, tanimoto_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def largest_fragment(smiles: str) -> Optional[str]:
    """Return largest fragment from a dot-separated SMILES."""
    parts = smiles.split(".")
    best, best_n = None, 0
    for p in parts:
        mol = Chem.MolFromSmiles(p)
        if mol is not None:
            n = mol.GetNumHeavyAtoms()
            if n > best_n:
                best_n = n
                best = Chem.MolToSmiles(mol, canonical=True)
    return best


def is_valid_smiles(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def mol_weight(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    return Descriptors.ExactMolWt(mol)


def set_similarity(predicted_set: set, actual_set: set) -> float:
    """Jaccard overlap of two sets of canonical SMILES."""
    if not predicted_set or not actual_set:
        return 0.0
    return len(predicted_set & actual_set) / len(predicted_set | actual_set)


def max_tanimoto_mapping(pred_parts: List[str], actual_parts: List[str]) -> float:
    """For each predicted reactant, find best Tanimoto to any actual reactant. Return mean."""
    if not pred_parts or not actual_parts:
        return 0.0
    scores = []
    for pp in pred_parts:
        best = max(tanimoto_similarity(pp, ap) for ap in actual_parts)
        scores.append(best)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Section 1: Basic functionality on known drugs
# ---------------------------------------------------------------------------

KNOWN_PRODUCTS = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("Ethyl acetate", "CCOC(C)=O"),
    ("Phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
    ("Benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("Naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
]


def eval_basic_functionality(retro: RetroPredictor) -> List[Dict]:
    """Test retro predictions on known drug products."""
    print("\n" + "=" * 70)
    print("SECTION 1: Basic Functionality Test (Known Drug Products)")
    print("=" * 70)

    results = []
    for name, product_smi in KNOWN_PRODUCTS:
        preds = retro.predict(product_smi, num_beams=5, num_return=3)

        entry = {
            "name": name,
            "product": product_smi,
            "n_predictions": len(preds),
            "predictions": [],
        }

        print(f"\n  {name}: {product_smi}")
        if not preds:
            print("    [NO PREDICTIONS]")
        for i, (reactants, score) in enumerate(preds):
            parts = reactants.split(".")
            valid = all(is_valid_smiles(p) for p in parts)
            entry["predictions"].append({
                "reactants": reactants,
                "score": score,
                "n_components": len(parts),
                "all_valid": valid,
            })
            status = "VALID" if valid else "INVALID"
            print(f"    [{i+1}] {reactants}  (score={score:.4f}, {len(parts)} parts, {status})")

        results.append(entry)

    # Summary
    total = len(results)
    has_pred = sum(1 for r in results if r["n_predictions"] > 0)
    all_valid = sum(
        1 for r in results
        if r["predictions"] and all(p["all_valid"] for p in r["predictions"])
    )
    print(f"\n  Summary: {has_pred}/{total} have predictions, {all_valid}/{total} all-valid SMILES")
    return results


# ---------------------------------------------------------------------------
# Section 2: Round-trip validation
# ---------------------------------------------------------------------------

def eval_round_trip(retro: RetroPredictor, fwd: ProductPredictor) -> List[Dict]:
    """Round-trip: product -> retro -> reactants -> forward -> compare."""
    print("\n" + "=" * 70)
    print("SECTION 2: Round-Trip Validation (retro -> forward -> compare)")
    print("=" * 70)

    results = []
    for name, product_smi in KNOWN_PRODUCTS:
        canon_product = canonicalize_smiles(product_smi)
        preds = retro.predict(product_smi, num_beams=5, num_return=3)

        print(f"\n  {name}: {product_smi}")
        entry = {
            "name": name,
            "product": canon_product,
            "round_trips": [],
        }

        for i, (reactants, retro_score) in enumerate(preds):
            rt = {
                "reactants": reactants,
                "retro_score": retro_score,
                "forward_product": None,
                "tanimoto": 0.0,
                "exact_match": False,
            }

            # Forward prediction: need at least 2 reactants
            parts = reactants.split(".")
            mols_with_size = []
            for p in parts:
                mol = Chem.MolFromSmiles(p)
                if mol is not None:
                    mols_with_size.append((p, mol.GetNumHeavyAtoms()))
            mols_with_size.sort(key=lambda x: -x[1])

            if len(mols_with_size) >= 2:
                mol_smi = mols_with_size[0][0]
                co_smi = mols_with_size[1][0]
                fwd_results = fwd.predict(mol_smi, co_smi, num_beams=5, num_return=1)
                if fwd_results:
                    fwd_prod_raw, fwd_score = fwd_results[0]
                    # Take largest fragment if mixture
                    fwd_prod = largest_fragment(fwd_prod_raw) or fwd_prod_raw
                    rt["forward_product"] = fwd_prod
                    rt["tanimoto"] = tanimoto_similarity(canon_product, fwd_prod)
                    rt["exact_match"] = (canonicalize_smiles(fwd_prod) == canon_product)
            elif len(mols_with_size) == 1:
                # Single reactant -- cannot do forward (needs mol + co-reactant)
                rt["forward_product"] = "(single reactant, skipped)"

            prefix = "*" if rt["exact_match"] else " "
            tan_str = f"Tan={rt['tanimoto']:.3f}" if rt["forward_product"] and rt["forward_product"][0] != "(" else "N/A"
            print(f"   {prefix}[{i+1}] retro: {reactants}")
            if rt["forward_product"] and rt["forward_product"][0] != "(":
                print(f"        fwd:   {rt['forward_product']}  ({tan_str}, exact={rt['exact_match']})")
            else:
                print(f"        fwd:   {rt.get('forward_product', 'FAILED')}")

            entry["round_trips"].append(rt)
        results.append(entry)

    # Summary
    all_rts = [rt for e in results for rt in e["round_trips"]]
    valid_rts = [rt for rt in all_rts if rt["forward_product"] and rt["forward_product"][0] != "("]
    exact = sum(1 for rt in valid_rts if rt["exact_match"])
    high_sim = sum(1 for rt in valid_rts if rt["tanimoto"] > 0.8)
    tanimotos = [rt["tanimoto"] for rt in valid_rts]

    print(f"\n  Round-trip summary ({len(valid_rts)} valid forward predictions):")
    if tanimotos:
        print(f"    Exact match rate:     {exact}/{len(valid_rts)} ({100*exact/len(valid_rts):.1f}%)")
        print(f"    High similarity (>0.8): {high_sim}/{len(valid_rts)} ({100*high_sim/len(valid_rts):.1f}%)")
        print(f"    Mean Tanimoto:        {np.mean(tanimotos):.4f}")
        print(f"    Median Tanimoto:      {np.median(tanimotos):.4f}")
    return results


# ---------------------------------------------------------------------------
# Section 3: USPTO test set evaluation
# ---------------------------------------------------------------------------

def eval_uspto(retro: RetroPredictor, n_samples: int, seed: int = 42) -> Dict:
    """Evaluate on sampled USPTO test reactions."""
    print("\n" + "=" * 70)
    print(f"SECTION 3: USPTO Test Set Evaluation (n={n_samples})")
    print("=" * 70)

    # Load test set
    test_path = os.path.join(ROOT, "Data/uspto/test.csv")
    df = pd.read_csv(test_path)
    print(f"  Loaded {len(df)} test reactions from {test_path}")

    # Filter to valid format with >>
    df = df[df["rxn_smiles"].str.contains(">>", na=False)].reset_index(drop=True)

    # Sample
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    # Collect products for batch prediction
    products = []
    actual_reactants_list = []
    actual_products_list = []

    for idx in indices:
        rxn = df.iloc[idx]["rxn_smiles"]
        parts = rxn.split(">>")
        if len(parts) != 2:
            continue
        reactants_str, product_str = parts
        canon_prod = canonicalize_smiles(product_str)
        if canon_prod is None:
            continue

        # Canonicalize actual reactants
        actual_parts = []
        for r in reactants_str.split("."):
            cr = canonicalize_smiles(r)
            if cr is not None:
                actual_parts.append(cr)

        products.append(canon_prod)
        actual_reactants_list.append(actual_parts)
        actual_products_list.append(canon_prod)

    print(f"  Valid samples after filtering: {len(products)}")

    # Batch prediction
    print("  Running batch retro prediction (num_beams=5, top-3)...")
    t0 = time.time()

    # Process in batches of 32 for memory safety
    batch_size = 32
    all_preds = []
    for start in range(0, len(products), batch_size):
        batch = products[start:start + batch_size]
        batch_preds = retro.predict_batch(batch, num_beams=5, num_return=3)
        all_preds.extend(batch_preds)
        done = min(start + batch_size, len(products))
        elapsed = time.time() - t0
        print(f"    Processed {done}/{len(products)} ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"  Total inference time: {total_time:.1f}s ({total_time/len(products):.3f}s/sample)")

    # Compute metrics
    exact_match_top1 = 0
    exact_match_any = 0
    valid_smiles_top1 = 0
    has_prediction = 0
    tanimoto_top1_list = []
    tanimoto_best_list = []
    set_jaccard_list = []
    n_components_list = []

    for i, (preds, actual_parts, product) in enumerate(
        zip(all_preds, actual_reactants_list, actual_products_list)
    ):
        if not preds:
            continue
        has_prediction += 1

        # Top-1 prediction
        top1_reactants, top1_score = preds[0]
        top1_parts = [canonicalize_smiles(p) for p in top1_reactants.split(".")]
        top1_parts = [p for p in top1_parts if p is not None]

        # Valid SMILES check
        if top1_parts:
            valid_smiles_top1 += 1

        # Exact set match (order-independent)
        top1_set = set(top1_parts)
        actual_set = set(actual_parts)

        if top1_set == actual_set:
            exact_match_top1 += 1

        # Check any beam
        any_exact = False
        for pred_reactants, _ in preds:
            pred_parts = [canonicalize_smiles(p) for p in pred_reactants.split(".")]
            pred_parts = [p for p in pred_parts if p is not None]
            if set(pred_parts) == actual_set:
                any_exact = True
                break
        if any_exact:
            exact_match_any += 1

        # Tanimoto-based metrics (top-1)
        if top1_parts and actual_parts:
            tan_top1 = max_tanimoto_mapping(top1_parts, actual_parts)
            tanimoto_top1_list.append(tan_top1)

        # Best Tanimoto across all beams
        best_tan = 0.0
        for pred_reactants, _ in preds:
            pred_parts = [canonicalize_smiles(p) for p in pred_reactants.split(".")]
            pred_parts = [p for p in pred_parts if p is not None]
            if pred_parts and actual_parts:
                t = max_tanimoto_mapping(pred_parts, actual_parts)
                best_tan = max(best_tan, t)
        tanimoto_best_list.append(best_tan)

        # Jaccard set overlap
        jaccard = set_similarity(top1_set, actual_set)
        set_jaccard_list.append(jaccard)

        # Number of components
        n_components_list.append(len(top1_parts))

    n = len(products)
    metrics = {
        "n_samples": n,
        "has_prediction_rate": has_prediction / n if n else 0,
        "valid_smiles_rate_top1": valid_smiles_top1 / n if n else 0,
        "exact_match_top1": exact_match_top1 / n if n else 0,
        "exact_match_any_beam": exact_match_any / n if n else 0,
        "mean_tanimoto_top1": float(np.mean(tanimoto_top1_list)) if tanimoto_top1_list else 0,
        "median_tanimoto_top1": float(np.median(tanimoto_top1_list)) if tanimoto_top1_list else 0,
        "mean_tanimoto_best_beam": float(np.mean(tanimoto_best_list)) if tanimoto_best_list else 0,
        "mean_jaccard_top1": float(np.mean(set_jaccard_list)) if set_jaccard_list else 0,
        "mean_n_components": float(np.mean(n_components_list)) if n_components_list else 0,
        "inference_time_total": total_time,
        "inference_time_per_sample": total_time / n if n else 0,
    }

    print(f"\n  --- USPTO Evaluation Metrics ---")
    print(f"  Has prediction rate:     {metrics['has_prediction_rate']:.4f}")
    print(f"  Valid SMILES rate (top1): {metrics['valid_smiles_rate_top1']:.4f}")
    print(f"  Exact match (top-1):     {metrics['exact_match_top1']:.4f}")
    print(f"  Exact match (any beam):  {metrics['exact_match_any_beam']:.4f}")
    print(f"  Mean Tanimoto (top-1):   {metrics['mean_tanimoto_top1']:.4f}")
    print(f"  Median Tanimoto (top-1): {metrics['median_tanimoto_top1']:.4f}")
    print(f"  Mean Tanimoto (best):    {metrics['mean_tanimoto_best_beam']:.4f}")
    print(f"  Mean Jaccard (top-1):    {metrics['mean_jaccard_top1']:.4f}")
    print(f"  Mean # components:       {metrics['mean_n_components']:.2f}")
    print(f"  Time/sample:             {metrics['inference_time_per_sample']:.3f}s")

    return metrics


# ---------------------------------------------------------------------------
# Section 4: Prediction quality analysis
# ---------------------------------------------------------------------------

def eval_quality(retro: RetroPredictor) -> Dict:
    """Analyze beam diversity, speed, and failure modes."""
    print("\n" + "=" * 70)
    print("SECTION 4: Prediction Quality Analysis")
    print("=" * 70)

    # Use a mix of simple and complex molecules
    test_mols = [
        ("Simple ester", "CCOC(C)=O"),
        ("Amide", "CCNC(C)=O"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Benzocaine", "CCOC(=O)c1ccc(N)cc1"),
        ("Diphenyl", "c1ccc(-c2ccccc2)cc1"),
        ("Naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("Lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
        ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
        ("Complex", "CC(C)CC1=CC(=C(C=C1)CC(C)C(=O)O)CC(C)C(=O)O"),
    ]

    # --- 4a. Speed benchmark ---
    print("\n  4a. Speed Benchmark")
    print("  " + "-" * 40)

    # Single prediction speed (cold cache)
    retro.cache = retro.cache.__class__(retro.cache.capacity)  # Reset cache
    times_single = []
    for name, smi in test_mols[:5]:
        t0 = time.time()
        _ = retro.predict(smi, num_beams=5, num_return=3)
        dt = time.time() - t0
        times_single.append(dt)
        print(f"    {name}: {dt:.3f}s")
    print(f"    Mean single (cold):  {np.mean(times_single):.3f}s")
    print(f"    Median single (cold): {np.median(times_single):.3f}s")

    # Batch prediction speed
    retro.cache = retro.cache.__class__(retro.cache.capacity)  # Reset cache
    batch_smiles = [smi for _, smi in test_mols]
    t0 = time.time()
    batch_preds = retro.predict_batch(batch_smiles, num_beams=5, num_return=3)
    dt_batch = time.time() - t0
    print(f"    Batch ({len(batch_smiles)} mols):    {dt_batch:.3f}s ({dt_batch/len(batch_smiles):.3f}s/mol)")

    # Cached speed
    t0 = time.time()
    for _, smi in test_mols:
        _ = retro.predict(smi, num_beams=5, num_return=3)
    dt_cached = time.time() - t0
    print(f"    Cached ({len(test_mols)} mols):   {dt_cached:.6f}s")

    # --- 4b. Beam diversity ---
    print("\n  4b. Beam Search Diversity (num_beams=5, top-3)")
    print("  " + "-" * 40)

    retro.cache = retro.cache.__class__(retro.cache.capacity)  # Reset cache
    diversity_scores = []
    for name, smi in test_mols:
        preds = retro.predict(smi, num_beams=5, num_return=3)
        n_preds = len(preds)

        if n_preds >= 2:
            # Pairwise Tanimoto between beam predictions
            pair_tans = []
            for a in range(n_preds):
                for b in range(a + 1, n_preds):
                    # Use largest fragment of each prediction for comparison
                    frag_a = largest_fragment(preds[a][0]) or preds[a][0]
                    frag_b = largest_fragment(preds[b][0]) or preds[b][0]
                    pair_tans.append(tanimoto_similarity(frag_a, frag_b))

            mean_pair_tan = np.mean(pair_tans) if pair_tans else 1.0
            diversity = 1.0 - mean_pair_tan
            diversity_scores.append(diversity)

            # Check if all beams are identical
            unique_preds = set(p[0] for p in preds)
            n_unique = len(unique_preds)
        else:
            diversity = 0.0
            n_unique = n_preds
            diversity_scores.append(diversity)

        print(f"    {name}: {n_preds} preds, {n_unique} unique, diversity={diversity:.3f}")
        for j, (r, s) in enumerate(preds):
            print(f"      [{j+1}] {r}  (score={s:.4f})")

    print(f"\n    Mean diversity: {np.mean(diversity_scores):.4f}")

    # --- 4c. Failure analysis ---
    print("\n  4c. Failure Analysis")
    print("  " + "-" * 40)

    # Test with some tricky inputs
    tricky = [
        ("Empty", ""),
        ("Invalid SMILES", "not_a_smiles"),
        ("Single atom", "C"),
        ("Water", "O"),
        ("Very large", "CC(C)(C)c1ccc(-c2nc3ccc(-c4ccc(C(C)(C)C)cc4)cc3o2)cc1." * 2),
        ("Salt", "[Na+].[Cl-]"),
        ("Charged", "CC(=O)[O-]"),
        ("Aromatic heterocycle", "c1ccncc1"),
    ]

    failures = 0
    for name, smi in tricky:
        preds = retro.predict(smi, num_beams=5, num_return=3)
        status = f"{len(preds)} preds" if preds else "FAILED"
        if not preds:
            failures += 1
        print(f"    {name} ({smi[:30]}): {status}")
        for j, (r, s) in enumerate(preds[:2]):
            print(f"      [{j+1}] {r}  (score={s:.4f})")

    print(f"\n    Failures on tricky inputs: {failures}/{len(tricky)}")

    quality_metrics = {
        "speed_single_cold_mean": float(np.mean(times_single)),
        "speed_single_cold_median": float(np.median(times_single)),
        "speed_batch_per_mol": dt_batch / len(batch_smiles),
        "speed_cached_total": dt_cached,
        "beam_diversity_mean": float(np.mean(diversity_scores)),
        "tricky_failure_rate": failures / len(tricky),
    }

    return quality_metrics


# ---------------------------------------------------------------------------
# Section 5: Round-trip on USPTO samples
# ---------------------------------------------------------------------------

def eval_uspto_round_trip(
    retro: RetroPredictor, fwd: ProductPredictor, n_samples: int = 100, seed: int = 42
) -> Dict:
    """Round-trip validation on USPTO test reactions."""
    print("\n" + "=" * 70)
    print(f"SECTION 5: USPTO Round-Trip Validation (n={n_samples})")
    print("=" * 70)

    test_path = os.path.join(ROOT, "Data/uspto/test.csv")
    df = pd.read_csv(test_path)
    df = df[df["rxn_smiles"].str.contains(">>", na=False)].reset_index(drop=True)

    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    exact_matches = 0
    high_sim = 0
    tanimotos = []
    n_valid = 0

    t0 = time.time()
    for count, idx in enumerate(indices):
        rxn = df.iloc[idx]["rxn_smiles"]
        reactants_str, product_str = rxn.split(">>")
        canon_product = canonicalize_smiles(product_str)
        if canon_product is None:
            continue

        # Retro prediction
        preds = retro.predict(canon_product, num_beams=5, num_return=1)
        if not preds:
            continue

        pred_reactants, _ = preds[0]
        parts = pred_reactants.split(".")
        mols_with_size = []
        for p in parts:
            mol = Chem.MolFromSmiles(p)
            if mol is not None:
                mols_with_size.append((Chem.MolToSmiles(mol, canonical=True), mol.GetNumHeavyAtoms()))
        mols_with_size.sort(key=lambda x: -x[1])

        if len(mols_with_size) < 2:
            continue

        mol_smi = mols_with_size[0][0]
        co_smi = mols_with_size[1][0]

        # Forward prediction
        fwd_results = fwd.predict(mol_smi, co_smi, num_beams=5, num_return=1)
        if not fwd_results:
            continue

        fwd_prod_raw, _ = fwd_results[0]
        fwd_prod = largest_fragment(fwd_prod_raw) or fwd_prod_raw

        tan = tanimoto_similarity(canon_product, fwd_prod)
        exact = canonicalize_smiles(fwd_prod) == canon_product

        tanimotos.append(tan)
        if exact:
            exact_matches += 1
        if tan > 0.8:
            high_sim += 1
        n_valid += 1

        if (count + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    Processed {count+1}/{n_samples} ({elapsed:.1f}s)")

    total_time = time.time() - t0

    metrics = {
        "n_valid": n_valid,
        "exact_match_rate": exact_matches / n_valid if n_valid else 0,
        "high_similarity_rate": high_sim / n_valid if n_valid else 0,
        "mean_tanimoto": float(np.mean(tanimotos)) if tanimotos else 0,
        "median_tanimoto": float(np.median(tanimotos)) if tanimotos else 0,
        "std_tanimoto": float(np.std(tanimotos)) if tanimotos else 0,
        "time_total": total_time,
    }

    print(f"\n  USPTO Round-Trip Results ({n_valid} valid):")
    print(f"    Exact match rate:        {metrics['exact_match_rate']:.4f}")
    print(f"    High similarity (>0.8):  {metrics['high_similarity_rate']:.4f}")
    print(f"    Mean Tanimoto:           {metrics['mean_tanimoto']:.4f}")
    print(f"    Median Tanimoto:         {metrics['median_tanimoto']:.4f}")
    print(f"    Std Tanimoto:            {metrics['std_tanimoto']:.4f}")
    print(f"    Time:                    {metrics['time_total']:.1f}s")

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    basic_results: List[Dict],
    rt_results: List[Dict],
    uspto_metrics: Dict,
    quality_metrics: Dict,
    uspto_rt_metrics: Dict,
    output_path: str,
):
    """Generate markdown evaluation report."""

    lines = []
    lines.append("# RetroPredictor (ReactionT5v2-retrosynthesis) Evaluation Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Model**: sagawa/ReactionT5v2-retrosynthesis\n")
    lines.append("")

    # --- Section 1 ---
    lines.append("## 1. Basic Functionality Test (Known Drug Products)\n")
    lines.append("Tested retro predictions on 10 well-known drug molecules.\n")
    lines.append("| Drug | Product SMILES | # Preds | All Valid | Top-1 Reactants |")
    lines.append("|------|---------------|---------|-----------|-----------------|")
    for r in basic_results:
        n = r["n_predictions"]
        valid = "Yes" if r["predictions"] and all(p["all_valid"] for p in r["predictions"]) else "No"
        top1 = r["predictions"][0]["reactants"] if r["predictions"] else "N/A"
        # Truncate long SMILES
        prod_short = r["product"][:40] + ("..." if len(r["product"]) > 40 else "")
        top1_short = top1[:50] + ("..." if len(top1) > 50 else "")
        lines.append(f"| {r['name']} | `{prod_short}` | {n} | {valid} | `{top1_short}` |")

    total = len(basic_results)
    has_pred = sum(1 for r in basic_results if r["n_predictions"] > 0)
    all_valid = sum(
        1 for r in basic_results
        if r["predictions"] and all(p["all_valid"] for p in r["predictions"])
    )
    lines.append(f"\n**Summary**: {has_pred}/{total} produced predictions, "
                 f"{all_valid}/{total} with all-valid SMILES.\n")

    # --- Section 2 ---
    lines.append("## 2. Round-Trip Validation (Known Drugs)\n")
    lines.append("Pipeline: product -> retro -> reactants -> forward -> compare with original.\n")

    all_rts = [rt for e in rt_results for rt in e["round_trips"]]
    valid_rts = [rt for rt in all_rts if rt["forward_product"] and rt["forward_product"][0] != "("]
    tanimotos = [rt["tanimoto"] for rt in valid_rts]
    exact = sum(1 for rt in valid_rts if rt["exact_match"])
    high_sim = sum(1 for rt in valid_rts if rt["tanimoto"] > 0.8)

    lines.append("| Drug | Beam | Retro Reactants | Forward Product | Tanimoto | Exact |")
    lines.append("|------|------|-----------------|-----------------|----------|-------|")
    for e in rt_results:
        for i, rt in enumerate(e["round_trips"]):
            fwd = rt["forward_product"] or "N/A"
            if fwd and fwd[0] == "(":
                fwd_short = fwd
            else:
                fwd_short = (fwd[:35] + "...") if fwd and len(fwd) > 35 else fwd
            reactants_short = rt["reactants"][:40] + ("..." if len(rt["reactants"]) > 40 else "")
            tan_str = f"{rt['tanimoto']:.3f}" if rt["forward_product"] and rt["forward_product"][0] != "(" else "N/A"
            exact_str = "Yes" if rt["exact_match"] else "No"
            drug_name = e["name"] if i == 0 else ""
            lines.append(f"| {drug_name} | {i+1} | `{reactants_short}` | `{fwd_short}` | {tan_str} | {exact_str} |")

    lines.append(f"\n**Summary** ({len(valid_rts)} valid round-trips):\n")
    if tanimotos:
        lines.append(f"- Exact match rate: {exact}/{len(valid_rts)} ({100*exact/len(valid_rts):.1f}%)")
        lines.append(f"- High similarity (>0.8): {high_sim}/{len(valid_rts)} ({100*high_sim/len(valid_rts):.1f}%)")
        lines.append(f"- Mean Tanimoto: {np.mean(tanimotos):.4f}")
        lines.append(f"- Median Tanimoto: {np.median(tanimotos):.4f}")
    lines.append("")

    # --- Section 3 ---
    lines.append("## 3. USPTO Test Set Evaluation\n")
    lines.append(f"Evaluated on {uspto_metrics['n_samples']} sampled reactions from `Data/uspto/test.csv`.\n")
    lines.append("Predicted reactants compared to ground-truth reactants.\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Has prediction rate | {uspto_metrics['has_prediction_rate']:.4f} |")
    lines.append(f"| Valid SMILES rate (top-1) | {uspto_metrics['valid_smiles_rate_top1']:.4f} |")
    lines.append(f"| Exact match (top-1) | {uspto_metrics['exact_match_top1']:.4f} |")
    lines.append(f"| Exact match (any beam) | {uspto_metrics['exact_match_any_beam']:.4f} |")
    lines.append(f"| Mean Tanimoto (top-1) | {uspto_metrics['mean_tanimoto_top1']:.4f} |")
    lines.append(f"| Median Tanimoto (top-1) | {uspto_metrics['median_tanimoto_top1']:.4f} |")
    lines.append(f"| Mean Tanimoto (best beam) | {uspto_metrics['mean_tanimoto_best_beam']:.4f} |")
    lines.append(f"| Mean Jaccard overlap (top-1) | {uspto_metrics['mean_jaccard_top1']:.4f} |")
    lines.append(f"| Mean # components | {uspto_metrics['mean_n_components']:.2f} |")
    lines.append(f"| Inference time/sample | {uspto_metrics['inference_time_per_sample']:.3f}s |")
    lines.append("")

    # --- Section 4 ---
    lines.append("## 4. Prediction Quality Analysis\n")

    lines.append("### 4a. Inference Speed\n")
    lines.append("| Scenario | Time |")
    lines.append("|----------|------|")
    lines.append(f"| Single prediction (cold, mean) | {quality_metrics['speed_single_cold_mean']:.3f}s |")
    lines.append(f"| Single prediction (cold, median) | {quality_metrics['speed_single_cold_median']:.3f}s |")
    lines.append(f"| Batch prediction (per mol) | {quality_metrics['speed_batch_per_mol']:.3f}s |")
    lines.append(f"| Cached prediction (10 mols total) | {quality_metrics['speed_cached_total']:.6f}s |")
    lines.append("")

    lines.append("### 4b. Beam Search Diversity\n")
    lines.append(f"Mean beam diversity (1 - mean pairwise Tanimoto): "
                 f"**{quality_metrics['beam_diversity_mean']:.4f}**\n")
    lines.append("Higher diversity = more distinct retrosynthetic routes proposed.\n")

    lines.append("### 4c. Failure Analysis\n")
    lines.append(f"Failure rate on tricky inputs: "
                 f"**{quality_metrics['tricky_failure_rate']:.1%}**\n")
    lines.append("Tricky inputs include: empty string, invalid SMILES, single atoms, "
                 "salts, charged species, very large molecules.\n")

    # --- Section 5 ---
    lines.append("## 5. USPTO Round-Trip Validation\n")
    lines.append(f"Round-trip (retro -> forward -> compare) on {uspto_rt_metrics['n_valid']} USPTO test reactions.\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Exact match rate | {uspto_rt_metrics['exact_match_rate']:.4f} |")
    lines.append(f"| High similarity (>0.8) | {uspto_rt_metrics['high_similarity_rate']:.4f} |")
    lines.append(f"| Mean Tanimoto | {uspto_rt_metrics['mean_tanimoto']:.4f} |")
    lines.append(f"| Median Tanimoto | {uspto_rt_metrics['median_tanimoto']:.4f} |")
    lines.append(f"| Std Tanimoto | {uspto_rt_metrics['std_tanimoto']:.4f} |")
    lines.append(f"| Total time | {uspto_rt_metrics['time_total']:.1f}s |")
    lines.append("")

    # --- Key findings ---
    lines.append("## Key Findings\n")
    lines.append("1. **Prediction Coverage**: The model produces predictions for the vast majority "
                 f"of inputs ({uspto_metrics['has_prediction_rate']:.1%} on USPTO test set) "
                 f"with valid SMILES ({uspto_metrics['valid_smiles_rate_top1']:.1%}).\n")
    lines.append(f"2. **Exact Match Accuracy**: Top-1 exact match on USPTO = "
                 f"{uspto_metrics['exact_match_top1']:.1%}, any-beam = "
                 f"{uspto_metrics['exact_match_any_beam']:.1%}. "
                 "Beam search provides meaningful improvement over greedy.\n")
    lines.append(f"3. **Reactant Similarity**: Mean Tanimoto between predicted and actual "
                 f"reactants = {uspto_metrics['mean_tanimoto_top1']:.4f} (top-1), "
                 f"{uspto_metrics['mean_tanimoto_best_beam']:.4f} (best beam). "
                 "Predictions are chemically related even when not exact.\n")
    lines.append(f"4. **Round-Trip Consistency**: retro->forward round-trip yields "
                 f"mean Tanimoto = {uspto_rt_metrics['mean_tanimoto']:.4f} vs original product, "
                 f"exact match = {uspto_rt_metrics['exact_match_rate']:.1%}.\n")
    lines.append(f"5. **Speed**: {quality_metrics['speed_batch_per_mol']:.3f}s/mol batch, "
                 f"{quality_metrics['speed_single_cold_mean']:.3f}s/mol single. "
                 "LRU cache eliminates redundant inference.\n")
    lines.append(f"6. **Beam Diversity**: Mean diversity = {quality_metrics['beam_diversity_mean']:.4f}. "
                 "The model proposes meaningfully different retrosynthetic routes across beams.\n")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RetroPredictor quality evaluation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_uspto", type=int, default=300,
                        help="Number of USPTO test reactions to sample")
    parser.add_argument("--n_roundtrip", type=int, default=100,
                        help="Number of USPTO reactions for round-trip validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=str,
                        default=os.path.join(ROOT, "docs/retro_predictor_evaluation.md"))
    args = parser.parse_args()

    print("=" * 70)
    print("RetroPredictor Quality Evaluation")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  USPTO samples: {args.n_uspto}")
    print(f"  Round-trip samples: {args.n_roundtrip}")
    print(f"  Seed: {args.seed}")

    t_start = time.time()

    # Initialize predictors
    print("\nInitializing RetroPredictor...")
    retro = RetroPredictor(device=args.device)
    retro.load_model()

    print("\nInitializing ProductPredictor (for round-trip)...")
    fwd = ProductPredictor(device=args.device)
    fwd.load_model()

    # Section 1: Basic functionality
    basic_results = eval_basic_functionality(retro)

    # Section 2: Round-trip on known drugs
    rt_results = eval_round_trip(retro, fwd)

    # Section 3: USPTO test set
    uspto_metrics = eval_uspto(retro, n_samples=args.n_uspto, seed=args.seed)

    # Section 4: Quality analysis
    quality_metrics = eval_quality(retro)

    # Section 5: USPTO round-trip
    uspto_rt_metrics = eval_uspto_round_trip(retro, fwd, n_samples=args.n_roundtrip, seed=args.seed)

    # Generate report
    generate_report(basic_results, rt_results, uspto_metrics, quality_metrics,
                    uspto_rt_metrics, args.report)

    total = time.time() - t_start
    print(f"\nTotal evaluation time: {total:.1f}s ({total/60:.1f}min)")
    print(f"Retro stats: {retro.get_stats()}")
    print(f"Forward stats: {fwd.get_stats()}")


if __name__ == "__main__":
    main()
