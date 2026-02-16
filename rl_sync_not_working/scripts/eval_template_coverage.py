#!/usr/bin/env python
"""Evaluate template coverage of USPTO 50K reactions.

For each reaction in USPTO 50K, checks whether any of the 71 SMARTS
reaction templates (from RxnFlow/SynFlowNet) can reproduce it.

Match levels:
  1. Exact match: template product canonical SMILES == actual product SMILES
  2. Near match: Tanimoto similarity >= threshold (0.9, 0.8)
  3. Substructure match: reactant(s) match template pattern (reaction type match)

Usage:
    python scripts/eval_template_coverage.py [--sample N] [--workers W]
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction

# Suppress RDKit warnings (templates generate many mapping warnings)
RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "template" / "data" / "templates.txt"
DATA_DIR = PROJECT_ROOT / "Data" / "uspto"


# ============================================================
# Template loading (standalone, avoids importing template module)
# ============================================================

def load_all_templates(path: str) -> list[dict]:
    """Load templates and return list of dicts with metadata.

    Each dict: {index, smarts, num_reactants, num_products, rxn}
    """
    templates = []
    with open(path) as f:
        for idx, line in enumerate(f):
            smarts = line.strip()
            if not smarts:
                continue
            try:
                rxn = ReactionFromSmarts(smarts)
                ChemicalReaction.Initialize(rxn)
                templates.append({
                    "index": idx,
                    "smarts": smarts,
                    "num_reactants": rxn.GetNumReactantTemplates(),
                    "num_products": rxn.GetNumProductTemplates(),
                    "rxn": rxn,
                    "patterns": [rxn.GetReactantTemplate(i)
                                 for i in range(rxn.GetNumReactantTemplates())],
                })
            except Exception as e:
                print(f"  WARNING: Template {idx} failed to load: {e}")
    return templates


# ============================================================
# Fingerprint / similarity helpers
# ============================================================

def canonical(smi: str) -> str | None:
    """Return canonical SMILES or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def mol_fingerprint(mol):
    """Morgan radius-2 fingerprint (2048 bits)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(fp1, fp2) -> float:
    """Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def run_template_forward(rxn, reactant_mols, num_products):
    """Run forward reaction, return set of canonical product SMILES."""
    try:
        product_sets = rxn.RunReactants(tuple(reactant_mols), 10)
    except Exception:
        return set()

    products = set()
    for ps in product_sets:
        if len(ps) != num_products:
            continue
        for mol in ps:
            try:
                mol = Chem.RemoveHs(mol, updateExplicitCount=True)
                smi = Chem.MolToSmiles(mol)
                # Clean up common artifacts
                smi = smi.replace("[C]", "C").replace("[N]", "N").replace("[CH]", "C")
                can = canonical(smi)
                if can:
                    products.add(can)
            except Exception:
                continue
    return products


# ============================================================
# Per-reaction matching
# ============================================================

def match_reaction(rxn_smiles: str, templates: list[dict]) -> dict:
    """Check if any template matches a given reaction.

    Returns dict with match info.
    """
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return {"status": "parse_error", "rxn_smiles": rxn_smiles}

    reactant_str, product_str = parts

    # Parse product
    product_can = canonical(product_str)
    if product_can is None:
        return {"status": "invalid_product", "rxn_smiles": rxn_smiles}

    product_mol = Chem.MolFromSmiles(product_can)
    if product_mol is None:
        return {"status": "invalid_product", "rxn_smiles": rxn_smiles}
    product_fp = mol_fingerprint(product_mol)

    # Parse reactants -- handle the `.` ambiguity by trying to parse
    # individual SMILES fragments
    reactant_smiles_list = reactant_str.split(".")
    # Reconstruct valid molecules from fragments
    # Strategy: greedily merge fragments that don't parse alone
    reactant_mols = []
    reactant_cans = []
    i = 0
    while i < len(reactant_smiles_list):
        frag = reactant_smiles_list[i]
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            reactant_mols.append(mol)
            reactant_cans.append(Chem.MolToSmiles(mol))
            i += 1
        else:
            # Try merging with next fragment (might be a salt like Na.Cl)
            merged = False
            for j in range(i + 1, min(i + 4, len(reactant_smiles_list) + 1)):
                combined = ".".join(reactant_smiles_list[i:j])
                mol = Chem.MolFromSmiles(combined)
                if mol is not None:
                    reactant_mols.append(mol)
                    reactant_cans.append(Chem.MolToSmiles(mol))
                    i = j
                    merged = True
                    break
            if not merged:
                i += 1  # skip unparseable fragment

    if not reactant_mols:
        return {"status": "invalid_reactants", "rxn_smiles": rxn_smiles}

    num_reactants = len(reactant_mols)

    # Track best match
    best_match = {
        "status": "no_match",
        "rxn_smiles": rxn_smiles,
        "num_reactants": num_reactants,
        "exact_match": False,
        "best_tanimoto": 0.0,
        "best_template_idx": -1,
        "substructure_match_templates": [],  # templates where reactants match pattern
    }

    for tmpl in templates:
        tmpl_idx = tmpl["index"]
        tmpl_n_react = tmpl["num_reactants"]
        tmpl_n_prod = tmpl["num_products"]
        rxn_obj = tmpl["rxn"]
        patterns = tmpl["patterns"]

        # --- Substructure match: do reactants match the template pattern? ---
        if tmpl_n_react == 1 and num_reactants >= 1:
            # Try each reactant as the uni-molecular input
            for r_mol in reactant_mols:
                if r_mol.HasSubstructMatch(patterns[0]):
                    best_match["substructure_match_templates"].append(tmpl_idx)
                    # Try forward reaction
                    pred_products = run_template_forward(rxn_obj, [r_mol], tmpl_n_prod)
                    if product_can in pred_products:
                        best_match["exact_match"] = True
                        best_match["best_tanimoto"] = 1.0
                        best_match["best_template_idx"] = tmpl_idx
                        best_match["status"] = "exact_match"
                        return best_match  # exact match found, done

                    # Check tanimoto of predicted products vs actual
                    for pred_smi in pred_products:
                        pred_mol = Chem.MolFromSmiles(pred_smi)
                        if pred_mol is None:
                            continue
                        pred_fp = mol_fingerprint(pred_mol)
                        sim = tanimoto(product_fp, pred_fp)
                        if sim > best_match["best_tanimoto"]:
                            best_match["best_tanimoto"] = sim
                            best_match["best_template_idx"] = tmpl_idx
                    break  # only need to find one matching reactant for substructure

        elif tmpl_n_react == 2 and num_reactants >= 2:
            # Try all pairs of reactants in both orders
            matched_sub = False
            for i_r in range(num_reactants):
                for j_r in range(num_reactants):
                    if i_r == j_r:
                        continue
                    r0, r1 = reactant_mols[i_r], reactant_mols[j_r]
                    if r0.HasSubstructMatch(patterns[0]) and r1.HasSubstructMatch(patterns[1]):
                        if not matched_sub:
                            best_match["substructure_match_templates"].append(tmpl_idx)
                            matched_sub = True

                        pred_products = run_template_forward(rxn_obj, [r0, r1], tmpl_n_prod)
                        if product_can in pred_products:
                            best_match["exact_match"] = True
                            best_match["best_tanimoto"] = 1.0
                            best_match["best_template_idx"] = tmpl_idx
                            best_match["status"] = "exact_match"
                            return best_match

                        for pred_smi in pred_products:
                            pred_mol = Chem.MolFromSmiles(pred_smi)
                            if pred_mol is None:
                                continue
                            pred_fp = mol_fingerprint(pred_mol)
                            sim = tanimoto(product_fp, pred_fp)
                            if sim > best_match["best_tanimoto"]:
                                best_match["best_tanimoto"] = sim
                                best_match["best_template_idx"] = tmpl_idx

        elif tmpl_n_react == 2 and num_reactants == 1:
            # Bi-molecular template but uni-molecular reaction: skip
            pass

    # Classify final status
    if best_match["best_tanimoto"] >= 0.9:
        best_match["status"] = "near_match_0.9"
    elif best_match["best_tanimoto"] >= 0.8:
        best_match["status"] = "near_match_0.8"
    elif best_match["substructure_match_templates"]:
        best_match["status"] = "substructure_only"
    else:
        best_match["status"] = "no_match"

    # Deduplicate substructure match templates
    best_match["substructure_match_templates"] = sorted(
        set(best_match["substructure_match_templates"]))

    return best_match


# ============================================================
# Worker function for multiprocessing
# ============================================================

def _worker_init():
    """Worker initializer: suppress RDKit warnings."""
    RDLogger.logger().setLevel(RDLogger.ERROR)


def _worker_match(args):
    """Worker function: match a batch of reactions."""
    rxn_batch, template_data = args
    # Reconstruct templates from serialized data (can't pickle rxn objects)
    templates = []
    for td in template_data:
        try:
            rxn = ReactionFromSmarts(td["smarts"])
            ChemicalReaction.Initialize(rxn)
            templates.append({
                "index": td["index"],
                "smarts": td["smarts"],
                "num_reactants": rxn.GetNumReactantTemplates(),
                "num_products": rxn.GetNumProductTemplates(),
                "rxn": rxn,
                "patterns": [rxn.GetReactantTemplate(i)
                             for i in range(rxn.GetNumReactantTemplates())],
            })
        except Exception:
            pass

    results = []
    for rxn_smi in rxn_batch:
        results.append(match_reaction(rxn_smi, templates))
    return results


# ============================================================
# Main
# ============================================================

def load_reactions(split: str) -> list[str]:
    """Load reaction SMILES from a split CSV."""
    path = DATA_DIR / f"{split}.csv"
    reactions = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reactions.append(row["rxn_smiles"])
    return reactions


def main():
    parser = argparse.ArgumentParser(description="Evaluate template coverage of USPTO 50K")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N reactions per split (0 = all)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Reactions per worker batch")
    parser.add_argument("--report", type=str,
                        default=str(PROJECT_ROOT / "docs" / "template_coverage_report.md"),
                        help="Output report path")
    args = parser.parse_args()

    print("=" * 70)
    print("USPTO 50K Template Coverage Evaluation")
    print("=" * 70)

    # Load templates
    t0 = time.perf_counter()
    templates = load_all_templates(str(TEMPLATE_PATH))
    n_uni = sum(1 for t in templates if t["num_reactants"] == 1)
    n_bi = sum(1 for t in templates if t["num_reactants"] == 2)
    print(f"\nTemplates loaded: {len(templates)} ({n_uni} uni, {n_bi} bi) in "
          f"{time.perf_counter() - t0:.2f}s")

    # Serialize template data for workers (rxn objects can't be pickled)
    template_data = [{"index": t["index"], "smarts": t["smarts"]}
                     for t in templates]

    # Load reactions
    all_results = {}  # split -> list of match results
    split_reactions = {}

    for split in ["train", "val", "test"]:
        reactions = load_reactions(split)
        if args.sample > 0:
            import random
            random.seed(42)
            reactions = random.sample(reactions, min(args.sample, len(reactions)))
        split_reactions[split] = reactions
        print(f"\n{split}: {len(reactions)} reactions")

    total_reactions = sum(len(v) for v in split_reactions.values())
    print(f"\nTotal: {total_reactions} reactions to evaluate")

    # Process all reactions with multiprocessing
    t_start = time.perf_counter()

    for split in ["train", "val", "test"]:
        reactions = split_reactions[split]
        results = []

        # Create batches
        batches = []
        for i in range(0, len(reactions), args.batch_size):
            batch = reactions[i:i + args.batch_size]
            batches.append((batch, template_data))

        if args.workers > 1 and len(reactions) > 100:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_worker_init,
            ) as executor:
                futures = {executor.submit(_worker_match, b): i
                           for i, b in enumerate(batches)}
                batch_results = [None] * len(batches)
                done_count = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    batch_results[idx] = future.result()
                    done_count += 1
                    if done_count % 10 == 0 or done_count == len(batches):
                        elapsed = time.perf_counter() - t_start
                        print(f"\r  {split}: {done_count}/{len(batches)} batches "
                              f"({elapsed:.0f}s)", end="", flush=True)

                for br in batch_results:
                    results.extend(br)
        else:
            # Serial fallback
            for i, (batch, _) in enumerate(batches):
                for rxn_smi in batch:
                    results.append(match_reaction(rxn_smi, templates))
                if (i + 1) % 5 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"\r  {split}: {i + 1}/{len(batches)} batches "
                          f"({elapsed:.0f}s)", end="", flush=True)

        print()  # newline after progress
        all_results[split] = results

    total_time = time.perf_counter() - t_start
    print(f"\nTotal evaluation time: {total_time:.1f}s "
          f"({total_reactions / total_time:.0f} rxn/s)")

    # ============================================================
    # Aggregate statistics
    # ============================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Per-split and overall stats
    report_lines = []
    report_lines.append("# Template Coverage Report: USPTO 50K")
    report_lines.append("")
    report_lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**Templates**: {len(templates)} ({n_uni} uni-molecular, {n_bi} bi-molecular)")
    report_lines.append(f"**Template source**: RxnFlow/SynFlowNet (`template/data/templates.txt`)")
    sample_note = f" (sampled {args.sample}/split)" if args.sample > 0 else ""
    report_lines.append(f"**Reactions evaluated**: {total_reactions}{sample_note}")
    report_lines.append(f"**Evaluation time**: {total_time:.1f}s")
    report_lines.append("")

    # Overall summary
    overall_status = Counter()
    overall_sub = 0
    overall_tanimotos = []
    template_usage = Counter()  # template_idx -> count of exact/near matches
    template_sub_usage = Counter()  # template_idx -> count of substructure matches
    uncovered_classes = Counter()

    for split in ["train", "val", "test"]:
        for r in all_results[split]:
            overall_status[r["status"]] += 1
            if r.get("best_tanimoto", 0) > 0:
                overall_tanimotos.append(r["best_tanimoto"])
            if r.get("substructure_match_templates"):
                overall_sub += 1
                for tidx in r["substructure_match_templates"]:
                    template_sub_usage[tidx] += 1
            if r["status"] == "exact_match":
                template_usage[r["best_template_idx"]] += 1
            elif r["status"] in ("near_match_0.9", "near_match_0.8"):
                template_usage[r["best_template_idx"]] += 1

    exact = overall_status.get("exact_match", 0)
    near_09 = overall_status.get("near_match_0.9", 0)
    near_08 = overall_status.get("near_match_0.8", 0)
    sub_only = overall_status.get("substructure_only", 0)
    no_match = overall_status.get("no_match", 0)
    errors = (overall_status.get("parse_error", 0) +
              overall_status.get("invalid_product", 0) +
              overall_status.get("invalid_reactants", 0))

    report_lines.append("## Overall Coverage")
    report_lines.append("")
    report_lines.append("| Metric | Count | Percentage |")
    report_lines.append("|--------|------:|------------|")
    report_lines.append(f"| Exact match (SMILES identical) | {exact} | {100*exact/total_reactions:.1f}% |")
    report_lines.append(f"| Near match (Tanimoto >= 0.9) | {near_09} | {100*near_09/total_reactions:.1f}% |")
    report_lines.append(f"| Near match (Tanimoto >= 0.8) | {near_08} | {100*near_08/total_reactions:.1f}% |")
    report_lines.append(f"| Exact + Near (>= 0.9) | {exact + near_09} | {100*(exact + near_09)/total_reactions:.1f}% |")
    report_lines.append(f"| Exact + Near (>= 0.8) | {exact + near_09 + near_08} | {100*(exact + near_09 + near_08)/total_reactions:.1f}% |")
    report_lines.append(f"| Substructure match only | {sub_only} | {100*sub_only/total_reactions:.1f}% |")
    report_lines.append(f"| Any reactant pattern match | {overall_sub} | {100*overall_sub/total_reactions:.1f}% |")
    report_lines.append(f"| No match at all | {no_match} | {100*no_match/total_reactions:.1f}% |")
    report_lines.append(f"| Parse/validity errors | {errors} | {100*errors/total_reactions:.1f}% |")
    report_lines.append(f"| **Total** | **{total_reactions}** | **100%** |")
    report_lines.append("")

    # Print summary to stdout
    print(f"\n  Exact match:             {exact:6d} ({100*exact/total_reactions:.1f}%)")
    print(f"  Near match (Tan >= 0.9): {near_09:6d} ({100*near_09/total_reactions:.1f}%)")
    print(f"  Near match (Tan >= 0.8): {near_08:6d} ({100*near_08/total_reactions:.1f}%)")
    print(f"  Substructure match only: {sub_only:6d} ({100*sub_only/total_reactions:.1f}%)")
    print(f"  Any pattern match:       {overall_sub:6d} ({100*overall_sub/total_reactions:.1f}%)")
    print(f"  No match:                {no_match:6d} ({100*no_match/total_reactions:.1f}%)")
    print(f"  Errors:                  {errors:6d} ({100*errors/total_reactions:.1f}%)")

    # Per-split breakdown
    report_lines.append("## Per-Split Coverage")
    report_lines.append("")
    report_lines.append("| Split | Total | Exact | Near>=0.9 | Near>=0.8 | Substructure | No match |")
    report_lines.append("|-------|------:|------:|----------:|----------:|-------------:|---------:|")

    for split in ["train", "val", "test"]:
        results = all_results[split]
        n = len(results)
        s = Counter(r["status"] for r in results)
        ex = s.get("exact_match", 0)
        n9 = s.get("near_match_0.9", 0)
        n8 = s.get("near_match_0.8", 0)
        sb = s.get("substructure_only", 0)
        nm = s.get("no_match", 0)
        report_lines.append(
            f"| {split} | {n} | {ex} ({100*ex/n:.1f}%) | {n9} ({100*n9/n:.1f}%) | "
            f"{n8} ({100*n8/n:.1f}%) | {sb} ({100*sb/n:.1f}%) | {nm} ({100*nm/n:.1f}%) |")
        print(f"\n  [{split}] exact={ex}, near0.9={n9}, near0.8={n8}, sub={sb}, nomatch={nm}")

    report_lines.append("")

    # Tanimoto distribution for non-exact matches
    if overall_tanimotos:
        report_lines.append("## Tanimoto Similarity Distribution (best match per reaction)")
        report_lines.append("")
        import numpy as np
        tans = np.array(overall_tanimotos)
        report_lines.append(f"- Mean: {tans.mean():.3f}")
        report_lines.append(f"- Median: {np.median(tans):.3f}")
        report_lines.append(f"- Std: {tans.std():.3f}")
        report_lines.append(f"- Min: {tans.min():.3f}, Max: {tans.max():.3f}")
        report_lines.append("")

        # Histogram buckets
        report_lines.append("| Range | Count | Percentage |")
        report_lines.append("|-------|------:|------------|")
        buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0),
                   (1.0, 1.01)]
        for lo, hi in buckets:
            count = int(np.sum((tans >= lo) & (tans < hi)))
            label = f"[{lo:.1f}, {hi:.1f})" if hi <= 1.0 else "1.0 (exact)"
            report_lines.append(f"| {label} | {count} | {100*count/len(tans):.1f}% |")
        report_lines.append("")

    # Template usage distribution
    report_lines.append("## Template Usage Distribution")
    report_lines.append("")
    report_lines.append("### Templates by exact/near match count (top 20)")
    report_lines.append("")
    report_lines.append("| Template Index | Matches | Type | First 80 chars of SMARTS |")
    report_lines.append("|---------------:|--------:|------|--------------------------|")

    # Read original template lines for display
    with open(TEMPLATE_PATH) as f:
        raw_templates = [l.strip() for l in f if l.strip()]

    for tidx, count in template_usage.most_common(20):
        tmpl = next((t for t in templates if t["index"] == tidx), None)
        ttype = "uni" if tmpl and tmpl["num_reactants"] == 1 else "bi"
        smarts_preview = raw_templates[tidx][:80] if tidx < len(raw_templates) else "?"
        report_lines.append(f"| {tidx} | {count} | {ttype} | `{smarts_preview}` |")
    report_lines.append("")

    # Templates with zero matches
    matched_templates = set(template_usage.keys())
    all_template_indices = set(t["index"] for t in templates)
    unused = all_template_indices - matched_templates
    report_lines.append(f"**Templates with 0 exact/near matches**: {len(unused)} / {len(templates)}")
    report_lines.append("")

    # Substructure match distribution
    report_lines.append("### Templates by substructure match count (top 20)")
    report_lines.append("")
    report_lines.append("| Template Index | Substructure Matches | Type |")
    report_lines.append("|---------------:|---------------------:|------|")
    for tidx, count in template_sub_usage.most_common(20):
        tmpl = next((t for t in templates if t["index"] == tidx), None)
        ttype = "uni" if tmpl and tmpl["num_reactants"] == 1 else "bi"
        report_lines.append(f"| {tidx} | {count} | {ttype} |")
    report_lines.append("")

    # Uncovered reaction analysis
    report_lines.append("## Uncovered Reaction Analysis")
    report_lines.append("")

    # Analyze uncovered reactions by number of reactants
    uncovered_by_nreact = Counter()
    uncovered_examples = []
    for split in ["train", "val", "test"]:
        for r in all_results[split]:
            if r["status"] == "no_match":
                n_react = r.get("num_reactants", -1)
                uncovered_by_nreact[n_react] += 1
                if len(uncovered_examples) < 20:
                    uncovered_examples.append(r["rxn_smiles"])

    report_lines.append("### By number of reactants")
    report_lines.append("")
    report_lines.append("| # Reactants | Count |")
    report_lines.append("|------------:|------:|")
    for n_react, count in sorted(uncovered_by_nreact.items()):
        report_lines.append(f"| {n_react} | {count} |")
    report_lines.append("")

    # Show some example uncovered reactions
    if uncovered_examples:
        report_lines.append("### Example uncovered reactions (first 10)")
        report_lines.append("")
        report_lines.append("```")
        for ex in uncovered_examples[:10]:
            report_lines.append(ex)
        report_lines.append("```")
        report_lines.append("")

    # Reaction class distribution for covered vs uncovered
    report_lines.append("## Coverage by Reaction Class")
    report_lines.append("")

    # Load class info from CSVs
    class_covered = Counter()
    class_total = Counter()
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.csv"
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rxn_to_class = {}
            for row in reader:
                rxn_to_class[row["rxn_smiles"]] = row.get("class", row.get("rxn_class", "unknown"))

        for r in all_results[split]:
            rxn_smi = r["rxn_smiles"]
            cls = rxn_to_class.get(rxn_smi, "unknown")
            class_total[cls] += 1
            if r["status"] in ("exact_match", "near_match_0.9", "near_match_0.8"):
                class_covered[cls] += 1

    report_lines.append("| Reaction Class | Total | Covered (exact+near) | Coverage % |")
    report_lines.append("|---------------:|------:|---------------------:|-----------:|")
    for cls in sorted(class_total.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        total = class_total[cls]
        covered = class_covered.get(cls, 0)
        pct = 100 * covered / total if total > 0 else 0
        report_lines.append(f"| {cls} | {total} | {covered} | {pct:.1f}% |")
    report_lines.append("")

    # Conclusion
    report_lines.append("## Conclusion")
    report_lines.append("")
    coverage_pct = 100 * (exact + near_09 + near_08) / total_reactions
    pattern_pct = 100 * overall_sub / total_reactions
    report_lines.append(
        f"The 71 RxnFlow/SynFlowNet templates cover **{coverage_pct:.1f}%** of USPTO 50K "
        f"reactions (exact + near match with Tanimoto >= 0.8). "
        f"Reactant pattern matching (substructure) reaches **{pattern_pct:.1f}%**, "
        f"indicating that while templates recognize more reactants, they often produce "
        f"different products than the ground truth.")
    report_lines.append("")
    report_lines.append(
        f"This coverage rate represents the theoretical upper bound for template-based "
        f"molecular optimization in DA-MolDQN using these 71 templates.")

    # Write report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
