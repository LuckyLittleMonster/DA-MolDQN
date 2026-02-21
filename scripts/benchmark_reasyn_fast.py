"""Benchmark: Sampler vs FastSampler on ReaSyn.

Compares speed, GPU memory, and result quality on a small set of molecules.

Usage:
    cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn
    conda run -n reasyn --live-stream python ../../scripts/benchmark_reasyn_fast.py
"""

import sys
import os
import time
import pathlib
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Ensure reasyn is importable
REASYN_ROOT = pathlib.Path(__file__).resolve().parent.parent / "refs" / "ReaSyn"
sys.path.insert(0, str(REASYN_ROOT))

from rl.reasyn.chem.fpindex import FingerprintIndex
from rl.reasyn.chem.matrix import ReactantReactionMatrix
from rl.reasyn.chem.mol import Molecule
from rl.reasyn.models.reasyn import ReaSyn
from rl.reasyn.sampler.sampler import Sampler
from rl.reasyn.sampler.sampler_fast import FastSampler
from rl.reasyn.utils.sample_utils import TimeLimit


def load_models_and_data(model_dir: pathlib.Path, device: str = "cuda"):
    """Load AR + EditFlow models and chemistry data."""
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


def run_single(
    SamplerClass,
    mol: Molecule,
    models,
    fpindex,
    rxn_matrix,
    exhaustiveness: int = 64,
    search_width: int = 8,
    max_evolve_steps: int = 8,
    num_cycles: int = 1,
    num_editflow_samples: int = 10,
    num_editflow_steps: int = 50,
    time_limit: int = 60,
):
    """Run one sampling pass and return (df, elapsed, peak_mem)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    sampler = SamplerClass(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        mol=mol,
        model=models,
        factor=search_width,
        max_active_states=exhaustiveness,
    )
    tl = TimeLimit(time_limit)
    t0 = time.perf_counter()
    sampler.evolve(
        gpu_lock=None,
        time_limit=tl,
        max_evolve_steps=max_evolve_steps,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    df = sampler.get_dataframe()
    return df, elapsed, peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mols", type=int, default=4,
                        help="Number of molecules to test")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per molecule per sampler (for variance)")
    parser.add_argument("--exhaustiveness", type=int, default=64,
                        help="max_active_states (beam width)")
    parser.add_argument("--search_width", type=int, default=8,
                        help="branching factor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--time_limit", type=int, default=120,
                        help="seconds per molecule")
    args = parser.parse_args()

    model_dir = REASYN_ROOT / "data" / "trained_model"
    zinc_path = REASYN_ROOT / "data" / "zinc_first64.txt"

    print("Loading models and data...")
    models, fpindex, rxn_matrix = load_models_and_data(model_dir, args.device)

    # Read molecules
    with open(zinc_path) as f:
        lines = f.read().strip().split("\n")
    # Skip header if present
    if lines[0].upper() == "SMILES":
        lines = lines[1:]
    mols = [Molecule(s.strip()) for s in lines[:args.num_mols]]
    print(f"Testing {len(mols)} molecules, exhaustiveness={args.exhaustiveness}, "
          f"search_width={args.search_width}, runs={args.runs}")

    samplers = [
        ("Sampler", Sampler),
        ("FastSampler", FastSampler),
    ]

    results = []
    for mol_i, mol in enumerate(mols):
        print(f"\n{'='*60}")
        print(f"Molecule {mol_i}: {mol.csmiles}")
        print(f"{'='*60}")

        for name, cls in samplers:
            for run_i in range(args.runs):
                try:
                    df, elapsed, peak_mem = run_single(
                        cls, mol, models, fpindex, rxn_matrix,
                        exhaustiveness=args.exhaustiveness,
                        search_width=args.search_width,
                        time_limit=args.time_limit,
                    )
                    n_results = len(df)
                    max_score = df["score"].max() if n_results > 0 else 0.0
                    mean_score = df["score"].mean() if n_results > 0 else 0.0

                    print(f"  {name:>12s} run {run_i}: "
                          f"{elapsed:6.1f}s | {peak_mem:7.0f} MB | "
                          f"{n_results:3d} results | "
                          f"max={max_score:.3f} mean={mean_score:.3f}")

                    results.append({
                        "mol_idx": mol_i,
                        "smiles": mol.csmiles,
                        "sampler": name,
                        "run": run_i,
                        "time_s": elapsed,
                        "peak_mem_mb": peak_mem,
                        "n_results": n_results,
                        "max_score": max_score,
                        "mean_score": mean_score,
                    })
                except Exception as e:
                    print(f"  {name:>12s} run {run_i}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    df_results = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name in ["Sampler", "FastSampler"]:
        sub = df_results[df_results["sampler"] == name]
        if sub.empty:
            continue
        print(f"\n{name}:")
        print(f"  Avg time:      {sub['time_s'].mean():.1f}s "
              f"(std={sub['time_s'].std():.1f})")
        print(f"  Avg peak mem:  {sub['peak_mem_mb'].mean():.0f} MB")
        print(f"  Avg n_results: {sub['n_results'].mean():.0f}")
        print(f"  Avg max_score: {sub['max_score'].mean():.3f}")
        print(f"  Avg mean_score:{sub['mean_score'].mean():.3f}")

    if len(df_results) > 0:
        sampler_time = df_results[df_results["sampler"] == "Sampler"]["time_s"].mean()
        fast_time = df_results[df_results["sampler"] == "FastSampler"]["time_s"].mean()
        if fast_time > 0:
            print(f"\nSpeedup: {sampler_time / fast_time:.2f}x")


if __name__ == "__main__":
    main()
