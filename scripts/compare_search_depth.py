"""Compare ReaSyn output molecules across different search configurations.

Measures: overlap of product SMILES, score distributions, molecular diversity.

Usage:
    cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn
    conda run -n reasyn --live-stream python ../../scripts/compare_search_depth.py
"""

import sys
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REASYN_ROOT = pathlib.Path(__file__).resolve().parent.parent / "refs" / "ReaSyn"
sys.path.insert(0, str(REASYN_ROOT))

from rl.reasyn.chem.fpindex import FingerprintIndex
from rl.reasyn.chem.matrix import ReactantReactionMatrix
from rl.reasyn.chem.mol import Molecule
from rl.reasyn.models.reasyn import ReaSyn
from rl.reasyn.sampler.sampler_fast import FastSampler
from rl.reasyn.utils.sample_utils import TimeLimit


def load_models(model_dir, device="cuda"):
    ar_path = model_dir / "nv-reasyn-ar-166m-v2.ckpt"
    eb_path = model_dir / "nv-reasyn-eb-174m-v2.ckpt"
    models = []
    config = None
    for ckpt_path in [ar_path, eb_path]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ReaSyn(config.model).to(device)
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        models.append(model)
    fpindex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return models, fpindex, rxn_matrix


def run_config(name, mol, models, fpindex, rxn_matrix, ex, sw, steps):
    """Run one configuration, return (name, time, DataFrame)."""
    sampler = FastSampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix,
        mol=mol, model=models,
        factor=sw, max_active_states=ex,
        use_fp16=True,
        max_branch_states=ex // 2,
        skip_editflow=True,
        rxn_product_limit=1,
    )
    tl = TimeLimit(120)
    t0 = time.perf_counter()
    sampler.evolve(
        gpu_lock=None, time_limit=tl,
        max_evolve_steps=steps, num_cycles=1,
        num_editflow_samples=10, num_editflow_steps=50,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    df = sampler.get_dataframe()
    return name, elapsed, df


def analyze_overlap(ref_df, test_df, ref_name, test_name):
    """Compute overlap metrics between two result sets."""
    if ref_df.empty or test_df.empty:
        return {}

    ref_smiles = set(ref_df['smiles'].tolist())
    test_smiles = set(test_df['smiles'].tolist())

    overlap = ref_smiles & test_smiles
    jaccard = len(overlap) / len(ref_smiles | test_smiles) if (ref_smiles | test_smiles) else 0

    # Score-based comparison: for overlapping molecules, how do scores compare?
    # For non-overlapping: compare score distributions
    ref_scores = ref_df.set_index('smiles')['score']
    test_scores = test_df.set_index('smiles')['score']

    # Top-K overlap: are the BEST molecules the same?
    for k in [5, 10, 20]:
        ref_topk = set(ref_df.nlargest(k, 'score')['smiles'].tolist())
        test_topk = set(test_df.nlargest(k, 'score')['smiles'].tolist())
        topk_overlap = len(ref_topk & test_topk)
        print(f"    Top-{k} overlap: {topk_overlap}/{k} "
              f"({100*topk_overlap/k:.0f}%)")

    print(f"    Total overlap: {len(overlap)}/{len(ref_smiles)} ref, "
          f"{len(overlap)}/{len(test_smiles)} test "
          f"(Jaccard={jaccard:.3f})")

    # Score distribution comparison
    print(f"    Ref  scores: max={ref_df['score'].max():.3f}, "
          f"mean={ref_df['score'].mean():.3f}, "
          f"top5={ref_df['score'].nlargest(5).mean():.3f}, "
          f"n={len(ref_df)}")
    print(f"    Test scores: max={test_df['score'].max():.3f}, "
          f"mean={test_df['score'].mean():.3f}, "
          f"top5={test_df['score'].nlargest(5).mean():.3f}, "
          f"n={len(test_df)}")

    # Number of synthesis steps in products
    if 'num_steps' in ref_df.columns and 'num_steps' in test_df.columns:
        print(f"    Ref  steps: mean={ref_df['num_steps'].mean():.1f}, "
              f"max={ref_df['num_steps'].max()}")
        print(f"    Test steps: mean={test_df['num_steps'].mean():.1f}, "
              f"max={test_df['num_steps'].max()}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mols", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per config to measure stochastic variance")
    args = parser.parse_args()

    model_dir = REASYN_ROOT / "data" / "trained_model"
    zinc_path = REASYN_ROOT / "data" / "zinc_first64.txt"

    print("Loading models...")
    models, fpindex, rxn_matrix = load_models(model_dir)

    with open(zinc_path) as f:
        lines = f.read().strip().split("\n")
    if lines[0].upper() == "SMILES":
        lines = lines[1:]
    mols = [Molecule(s.strip()) for s in lines[:args.num_mols]]

    # Configurations to compare
    configs = [
        # (name, exhaustiveness, search_width, evolve_steps)
        ("full",    64, 8, 8),   # Full mode (reference)
        ("L1a",     16, 8, 4),   # Reduced: 16 states, 4 steps
        ("L1b",     16, 8, 2),   # Reduced: 16 states, 2 steps
        ("L1c",     16, 4, 2),   # Reduced: 16 states, sw=4, 2 steps
        ("L2",       8, 4, 1),   # Minimal: 8 states, sw=4, 1 step
    ]

    print(f"Testing {len(mols)} molecules × {len(configs)} configs × {args.runs} runs\n")

    # Aggregated results
    all_times = {c[0]: [] for c in configs}
    all_n_results = {c[0]: [] for c in configs}
    all_max_scores = {c[0]: [] for c in configs}
    all_mean_scores = {c[0]: [] for c in configs}
    all_top5_scores = {c[0]: [] for c in configs}

    # Per-molecule overlap with "full" (across runs)
    overlap_top5 = {c[0]: [] for c in configs}
    overlap_top10 = {c[0]: [] for c in configs}
    overlap_jaccard = {c[0]: [] for c in configs}

    for mol_i, mol in enumerate(mols):
        print(f"{'='*70}")
        print(f"Molecule {mol_i}: {mol.csmiles}")
        print(f"{'='*70}")

        # Run each config multiple times
        run_results = {c[0]: [] for c in configs}  # name → list of DataFrames

        for run_i in range(args.runs):
            for name, ex, sw, steps in configs:
                _, elapsed, df = run_config(name, mol, models, fpindex, rxn_matrix, ex, sw, steps)
                run_results[name].append(df)
                all_times[name].append(elapsed)
                n = len(df)
                all_n_results[name].append(n)
                all_max_scores[name].append(df['score'].max() if n > 0 else 0)
                all_mean_scores[name].append(df['score'].mean() if n > 0 else 0)
                all_top5_scores[name].append(df['score'].nlargest(5).mean() if n >= 5 else (df['score'].max() if n > 0 else 0))

                if run_i == 0:
                    print(f"  {name:>6s} (ex={ex}, sw={sw}, steps={steps}): "
                          f"{elapsed:.2f}s, {n} results, "
                          f"max={df['score'].max():.3f}" if n > 0 else f"  {name:>6s}: {elapsed:.2f}s, 0 results")

        # Cross-run overlap analysis: compare each non-full config to full
        print(f"\n  Overlap analysis (vs full, averaged over {args.runs} runs):")

        full_dfs = run_results["full"]
        for name, ex, sw, steps in configs:
            if name == "full":
                # Self-overlap: measure stochastic variance of full mode
                self_overlaps_top5 = []
                self_overlaps_top10 = []
                self_jaccards = []
                for i in range(len(full_dfs)):
                    for j in range(i+1, len(full_dfs)):
                        if full_dfs[i].empty or full_dfs[j].empty:
                            continue
                        s_i = set(full_dfs[i]['smiles'])
                        s_j = set(full_dfs[j]['smiles'])
                        jac = len(s_i & s_j) / len(s_i | s_j) if (s_i | s_j) else 0
                        self_jaccards.append(jac)

                        top5_i = set(full_dfs[i].nlargest(5, 'score')['smiles'])
                        top5_j = set(full_dfs[j].nlargest(5, 'score')['smiles'])
                        self_overlaps_top5.append(len(top5_i & top5_j) / 5)

                        top10_i = set(full_dfs[i].nlargest(10, 'score')['smiles'])
                        top10_j = set(full_dfs[j].nlargest(10, 'score')['smiles'])
                        self_overlaps_top10.append(len(top10_i & top10_j) / 10)

                if self_jaccards:
                    avg_j = np.mean(self_jaccards)
                    avg_t5 = np.mean(self_overlaps_top5)
                    avg_t10 = np.mean(self_overlaps_top10)
                    print(f"    full vs full (stochastic variance): "
                          f"top5={avg_t5:.0%}, top10={avg_t10:.0%}, "
                          f"jaccard={avg_j:.3f}")
                    overlap_top5[name].append(avg_t5)
                    overlap_top10[name].append(avg_t10)
                    overlap_jaccard[name].append(avg_j)
                continue

            test_dfs = run_results[name]
            mol_top5s = []
            mol_top10s = []
            mol_jacs = []
            for full_df in full_dfs:
                for test_df in test_dfs:
                    if full_df.empty or test_df.empty:
                        continue
                    s_full = set(full_df['smiles'])
                    s_test = set(test_df['smiles'])
                    jac = len(s_full & s_test) / len(s_full | s_test) if (s_full | s_test) else 0
                    mol_jacs.append(jac)

                    top5_f = set(full_df.nlargest(5, 'score')['smiles'])
                    top5_t = set(test_df.nlargest(5, 'score')['smiles'])
                    mol_top5s.append(len(top5_f & top5_t) / 5)

                    top10_f = set(full_df.nlargest(10, 'score')['smiles'])
                    top10_t = set(test_df.nlargest(10, 'score')['smiles'])
                    mol_top10s.append(len(top10_f & top10_t) / 10)

            if mol_jacs:
                avg_j = np.mean(mol_jacs)
                avg_t5 = np.mean(mol_top5s)
                avg_t10 = np.mean(mol_top10s)
                print(f"    {name:>6s} vs full: "
                      f"top5={avg_t5:.0%}, top10={avg_t10:.0%}, "
                      f"jaccard={avg_j:.3f}")
                overlap_top5[name].append(avg_t5)
                overlap_top10[name].append(avg_t10)
                overlap_jaccard[name].append(avg_j)

    # ═══════════════════════════════════════════════════════════════════
    # Grand summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("GRAND SUMMARY (averaged across all molecules and runs)")
    print(f"{'='*70}")

    print(f"\n{'Config':>6s} | {'Time':>6s} | {'Speedup':>7s} | {'#Res':>5s} | "
          f"{'MaxScr':>7s} | {'MeanScr':>7s} | {'Top5Scr':>7s} | "
          f"{'OvTop5':>6s} | {'OvTop10':>7s} | {'Jaccard':>7s}")
    print("-" * 95)

    full_time = np.mean(all_times["full"])

    for name, ex, sw, steps in configs:
        t = np.mean(all_times[name])
        spd = full_time / t if t > 0 else 0
        nres = np.mean(all_n_results[name])
        maxs = np.mean(all_max_scores[name])
        means = np.mean(all_mean_scores[name])
        top5s = np.mean(all_top5_scores[name])
        ov_t5 = np.mean(overlap_top5[name]) if overlap_top5[name] else 0
        ov_t10 = np.mean(overlap_top10[name]) if overlap_top10[name] else 0
        jac = np.mean(overlap_jaccard[name]) if overlap_jaccard[name] else 0

        label = f"{name}" if name == "full" else name
        ov_label = "self" if name == "full" else "vs full"

        print(f"{label:>6s} | {t:5.2f}s | {spd:6.1f}x | {nres:5.0f} | "
              f"{maxs:7.3f} | {means:7.3f} | {top5s:7.3f} | "
              f"{ov_t5:5.0%} | {ov_t10:6.0%} | {jac:7.3f}  ({ov_label})")

    print(f"\nNote: 'full vs full' shows stochastic variance — even identical configs")
    print(f"      produce different molecules due to random sampling (temperature=0.1).")
    print(f"      This is the BASELINE for interpreting overlap numbers.")


if __name__ == "__main__":
    main()
