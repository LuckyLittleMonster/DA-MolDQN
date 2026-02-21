#!/usr/bin/env python
"""Benchmark ADMET-AI prediction speed: FastADMETModel vs ADMETModel.

Usage:
    conda run -n rl4 --live-stream python scripts/benchmark_admet.py
"""

import time
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU to avoid Lightning/CUDA issues

import numpy as np
import pandas as pd

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# Test SMILES — diverse set of drug-like molecules (64)
TEST_SMILES_64 = [
    "c1ccc(N)cc1", "c1ccc(O)cc1", "CCO", "CC(=O)O",
    "c1ccccc1", "CC(C)N", "c1ccc(C=O)cc1", "CCCC",
    "CC(=O)Nc1ccc(O)cc1",   # acetaminophen
    "CC(O)c1ccccc1",
    "c1ccc2[nH]ccc2c1",
    "CC(=O)OC1CC(N(C)C)C(OC2CC(C)(OC)C(=O)C(C)O2)C(C)(O)C1OC1CC(C)C(=O)C(C)O1",
    "O=C(O)c1ccccc1O",      # salicylic acid
    "CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2O",  # testosterone
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Clc1ccc(cc1)C(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",  # DDT
    "CC1=CC(=O)C(CC(C)CCCC(C)CCCC(C)CCCC(C)C)C(C)=C1O",
    "c1ccc2c(c1)ccc1ccccc12",  # phenanthrene
    "OC(=O)CCCCC(=O)O",
    "Nc1ccc(N)cc1", "CCN(CC)CC", "CCCCCCCC", "c1ccncc1",
    "CC(C)(C)c1ccc(O)cc1", "OC(=O)c1ccccc1", "Nc1ccccc1N",
    "CC(=O)c1ccc(Br)cc1", "c1ccc(-c2ccccc2)cc1", "OCCN1CCOCC1",
    "CC(C)Cc1ccc(CC(C)O)cc1", "c1ccc2ncccc2c1", "OC(=O)c1ccncc1",
    "Cc1cc(O)c(C)cc1N", "CCOc1ccccc1OCC", "c1cccc(Nc2ccccc2)c1",
    "CC(=O)Nc1ccc(OC)cc1", "CCOC(=O)c1ccccc1O", "Nc1cccc2ccccc12",
    "CC1CCCC(=O)C1", "c1ccc(CO)cc1", "Oc1cccc(O)c1O",
    "CC(=O)c1ccc(F)cc1", "CCN(c1ccccc1)c1ccccc1", "CC(=O)NCC(=O)O",
    "c1ccc(S)cc1", "CC(N)c1ccccc1", "OC(=O)CCc1ccccc1",
    "c1ccc(Oc2ccccc2)cc1", "CCc1ccc(CC)cc1", "c1cccc(C(=O)O)c1N",
    "CC(=O)c1cccc(O)c1", "Clc1cccc(Cl)c1", "CCOC(=O)CC(=O)OCC",
    "c1ccc(-c2ccccn2)cc1", "Nc1ccc(Cl)cc1", "CC(O)CC(=O)O",
    "c1ccc(C(=O)c2ccccc2)cc1", "CCOc1ccc(N)cc1", "OCC(O)CO",
    "c1cnc2ccccc2n1", "CC(=O)c1ccc(N)cc1", "CCCCN(CCCC)CCCC",
    "c1ccc(CC(=O)O)cc1", "Oc1ccc(O)c(O)c1", "c1cccc(CCO)c1",
    "CC(C)c1ccccc1", "CCOC(=O)c1ccccc1N",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "c1ccc(NC(=O)c2ccccc2)cc1",
    "CCCCC(=O)OCC", "Oc1ccccc1O",
    "CC(=O)c1ccc(OC)cc1", "c1ccc(CCCO)cc1",
]


def timed_runs(fn, n_warmup=2, n_runs=5):
    """Run function with warmup and return median time."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return np.median(times), result


def benchmark_original(smiles_list, n_warmup=2, n_runs=5):
    """Benchmark original ADMETModel."""
    from admet_ai import ADMETModel

    print("Loading ADMETModel (original)...")
    t0 = time.perf_counter()
    model = ADMETModel(include_physchem=True, num_workers=0)
    t_init = time.perf_counter() - t0
    print(f"  Init: {t_init:.3f}s")

    t_single, res_single = timed_runs(
        lambda: model.predict(smiles_list[0]), n_warmup, n_runs
    )
    t_batch, res_batch = timed_runs(
        lambda: model.predict(smiles_list), n_warmup, n_runs
    )

    return {
        "init": t_init,
        "single": t_single,
        "batch": t_batch,
        "result_single": res_single,
        "result_batch": res_batch,
        "n_mols": len(smiles_list),
    }


def benchmark_fast(smiles_list, n_warmup=2, n_runs=5):
    """Benchmark FastADMETModel."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reward.admet.model import FastADMETModel

    print("Loading FastADMETModel...")
    t0 = time.perf_counter()
    model = FastADMETModel(include_physchem=True, drugbank_percentiles=True, device="cpu")
    t_init = time.perf_counter() - t0
    print(f"  Init: {t_init:.3f}s")

    # Single mol (no cache)
    def single_no_cache():
        model.clear_cache()
        return model.predict(smiles_list[0])

    t_single, res_single = timed_runs(single_no_cache, n_warmup, n_runs)

    # Batch (no cache)
    def batch_no_cache():
        model.clear_cache()
        return model.predict(smiles_list)

    t_batch, res_batch = timed_runs(batch_no_cache, n_warmup, n_runs)

    # Batch with cache (second call)
    def batch_cached():
        model.clear_cache()
        model.predict(smiles_list)  # populate cache
        return model.predict(smiles_list)  # cache hit

    t_cached, _ = timed_runs(batch_cached, n_warmup, n_runs)

    # Properties only (fastest path)
    def props_only():
        model.clear_cache()
        return model.predict_properties(smiles_list)

    t_props, _ = timed_runs(props_only, n_warmup, n_runs)

    return {
        "init": t_init,
        "single": t_single,
        "batch": t_batch,
        "batch_cached": t_cached,
        "props_only": t_props,
        "result_single": res_single,
        "result_batch": res_batch,
        "n_mols": len(smiles_list),
    }


def compare_correctness(fast_result, orig_result, label):
    """Compare prediction correctness between Fast and Original models."""
    print(f"\n{'='*60}")
    print(f"Correctness Check: {label}")
    print(f"{'='*60}")

    if isinstance(fast_result, dict) and isinstance(orig_result, dict):
        all_keys = set(fast_result.keys()) & set(orig_result.keys())
        fast_only = set(fast_result.keys()) - set(orig_result.keys())
        orig_only = set(orig_result.keys()) - set(fast_result.keys())

        if fast_only:
            print(f"  Keys only in Fast: {len(fast_only)}")
        if orig_only:
            print(f"  Keys only in Original: {len(orig_only)}")

        max_diff = 0
        max_diff_key = ""
        diffs = []
        for k in sorted(all_keys):
            fv, ov = fast_result[k], orig_result[k]
            if isinstance(fv, (int, float)) and isinstance(ov, (int, float)):
                diff = abs(fv - ov)
                diffs.append(diff)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_key = k

        print(f"  Compared {len(diffs)} numeric properties")
        print(f"  Max absolute difference: {max_diff:.6e} (property: {max_diff_key})")
        print(f"  Mean absolute difference: {np.mean(diffs):.6e}")
        print(f"  All ADMET preds < 1e-5: {all(d < 1e-5 for d in diffs if 'percentile' not in max_diff_key)}")

    elif isinstance(fast_result, pd.DataFrame) and isinstance(orig_result, pd.DataFrame):
        common_cols = sorted(set(fast_result.columns) & set(orig_result.columns))
        fast_only = set(fast_result.columns) - set(orig_result.columns)
        orig_only = set(orig_result.columns) - set(fast_result.columns)

        print(f"  Fast columns: {len(fast_result.columns)}, Original columns: {len(orig_result.columns)}")
        print(f"  Common columns: {len(common_cols)}")
        if fast_only:
            print(f"  Fast-only columns ({len(fast_only)}): {sorted(fast_only)[:5]}...")
        if orig_only:
            print(f"  Original-only columns ({len(orig_only)}): {sorted(orig_only)[:5]}...")

        admet_cols = [c for c in common_cols if "percentile" not in c]
        pctile_cols = [c for c in common_cols if "percentile" in c]

        if admet_cols:
            fast_vals = fast_result[admet_cols].values
            orig_vals = orig_result[admet_cols].values
            diffs = np.abs(fast_vals - orig_vals)
            max_idx = np.unravel_index(np.nanargmax(diffs), diffs.shape)
            print(f"\n  ADMET + physchem properties ({len(admet_cols)} cols):")
            print(f"    Max abs diff: {np.nanmax(diffs):.6e} (col: {admet_cols[max_idx[1]]})")
            print(f"    Mean abs diff: {np.nanmean(diffs):.6e}")
            print(f"    All < 1e-5: {np.nanmax(diffs) < 1e-5}")

        if pctile_cols:
            fast_vals = fast_result[pctile_cols].values
            orig_vals = orig_result[pctile_cols].values
            diffs = np.abs(fast_vals - orig_vals)
            print(f"\n  DrugBank percentiles ({len(pctile_cols)} cols):")
            print(f"    Max abs diff: {np.nanmax(diffs):.2f}%")
            print(f"    Mean abs diff: {np.nanmean(diffs):.2f}%")
            print(f"    All < 1%: {np.nanmax(diffs) < 1.0}")


def main():
    smiles = TEST_SMILES_64
    print(f"=== ADMET-AI Benchmark ({len(smiles)} molecules) ===\n")

    fast = benchmark_fast(smiles)
    print()
    orig = benchmark_original(smiles)

    # Timing results
    print(f"\n{'='*65}")
    print(f"TIMING RESULTS ({fast['n_mols']} molecules)")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'Original':>12} {'Fast':>12} {'Speedup':>10}")
    print(f"{'-'*65}")
    print(f"{'Init (s)':<25} {orig['init']:>12.3f} {fast['init']:>12.3f} {orig['init']/fast['init']:>9.1f}x")
    print(f"{'Single mol (ms)':<25} {orig['single']*1000:>12.1f} {fast['single']*1000:>12.1f} {orig['single']/fast['single']:>9.1f}x")
    print(f"{'Batch {0} (ms)'.format(fast['n_mols']):<25} {orig['batch']*1000:>12.1f} {fast['batch']*1000:>12.1f} {orig['batch']/fast['batch']:>9.1f}x")
    print(f"{'Batch cached (ms)':<25} {'N/A':>12} {fast['batch_cached']*1000:>12.1f} {'':>10}")
    print(f"{'Props only (ms)':<25} {'N/A':>12} {fast['props_only']*1000:>12.1f} {'':>10}")
    print(f"{'Per-mol batch (ms)':<25} {orig['batch']*1000/orig['n_mols']:>12.2f} {fast['batch']*1000/fast['n_mols']:>12.2f} {orig['batch']/fast['batch']:>9.1f}x")

    # Correctness
    compare_correctness(fast["result_single"], orig["result_single"], "Single Molecule")
    compare_correctness(fast["result_batch"], orig["result_batch"], f"Batch ({fast['n_mols']} molecules)")

    print(f"\nCache info: {fast.get('cache_info', 'N/A')}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
