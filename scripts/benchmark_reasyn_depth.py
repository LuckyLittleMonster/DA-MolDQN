"""Benchmark ReaSyn configurations for synthesis depth vs speed.

Compare L1a (current RL), Medium (8x2), Full (8x4+editflow) configs.
"""
import os
import sys
import time
import pickle

import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REASYN_ROOT = os.path.join(PROJECT, 'refs', 'ReaSyn')
sys.path.insert(0, REASYN_ROOT)

from rl.reasyn.models.reasyn import ReaSyn
from rl.reasyn.chem.fpindex import FingerprintIndex
from rl.reasyn.chem.matrix import ReactantReactionMatrix
from rl.reasyn.chem.mol import Molecule
from rl.reasyn.sampler.sampler_fast import FastSampler
from rl.reasyn.sampler.sampler import Sampler
from rl.reasyn.utils.sample_utils import TimeLimit

# Patch sklearn compatibility (old pickle uses ManhattanDistance,
# new sklearn has ManhattanDistance64)
import sklearn.metrics._dist_metrics as _dm
if not hasattr(_dm, 'ManhattanDistance'):
    _dm.ManhattanDistance = _dm.ManhattanDistance64


def load_models(model_dir, device="cuda", fp16=False):
    model_files = [
        "nv-reasyn-ar-166m-v2.ckpt",
        "nv-reasyn-eb-174m-v2.ckpt",
    ]
    models = []
    reasyn_config = None
    for f in model_files:
        p = os.path.join(model_dir, f)
        ckpt = torch.load(p, map_location="cpu")
        reasyn_config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        m = ReaSyn(reasyn_config.model)
        m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        if fp16:
            m = m.half()
        m = m.to(device)
        m.eval()
        models.append(m)

    _fpindex_path = os.path.join(REASYN_ROOT, reasyn_config.chem.fpindex)
    _rxn_matrix_path = os.path.join(REASYN_ROOT, reasyn_config.chem.rxn_matrix)
    fpindex = pickle.load(open(_fpindex_path, "rb"))
    rxn_matrix = pickle.load(open(_rxn_matrix_path, "rb"))
    return models, fpindex, rxn_matrix


CONFIGS = {
    "L1a": {
        "sampler_cls": FastSampler,
        "sampler_kwargs": {
            "use_fp16": True,
            "max_branch_states": 8,
            "skip_editflow": True,
            "rxn_product_limit": 1,
        },
        "evolve_kwargs": {
            "max_evolve_steps": 4,
            "num_cycles": 1,
            "num_editflow_samples": 10,
            "num_editflow_steps": 50,
        },
        "factor": 8,
        "max_active_states": 16,
    },
    "Med_8x2": {
        "sampler_cls": FastSampler,
        "sampler_kwargs": {
            "use_fp16": True,
            "max_branch_states": 8,
            "skip_editflow": True,
            "rxn_product_limit": 1,
        },
        "evolve_kwargs": {
            "max_evolve_steps": 8,
            "num_cycles": 2,
            "num_editflow_samples": 10,
            "num_editflow_steps": 50,
        },
        "factor": 8,
        "max_active_states": 16,
    },
    "Med_8x3": {
        "sampler_cls": FastSampler,
        "sampler_kwargs": {
            "use_fp16": True,
            "max_branch_states": 8,
            "skip_editflow": True,
            "rxn_product_limit": 1,
        },
        "evolve_kwargs": {
            "max_evolve_steps": 8,
            "num_cycles": 3,
            "num_editflow_samples": 10,
            "num_editflow_steps": 50,
        },
        "factor": 8,
        "max_active_states": 16,
    },
    "Full_8x4_ef": {
        "sampler_cls": Sampler,  # Full mode uses base Sampler for editflow
        "sampler_kwargs": {},   # base Sampler has no extra kwargs
        "evolve_kwargs": {
            "max_evolve_steps": 8,
            "num_cycles": 4,
            "num_editflow_samples": 10,
            "num_editflow_steps": 100,
        },
        "factor": 16,
        "max_active_states": 256,
    },
}


def benchmark_config(name, config, mol, models, fpindex, rxn_matrix):
    """Run one config on one molecule, return stats."""
    SamplerCls = config["sampler_cls"]
    sampler = SamplerCls(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        mol=mol,
        model=models,
        factor=config["factor"],
        max_active_states=config["max_active_states"],
        **config["sampler_kwargs"],
    )
    tl = TimeLimit(120)
    t0 = time.perf_counter()
    try:
        sampler.evolve(gpu_lock=None, time_limit=tl, **config["evolve_kwargs"])
    except Exception as e:
        print(f"  ERROR in {name}: {e}")
        return None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    df = sampler.get_dataframe()
    if len(df) == 0:
        return {"config": name, "time": elapsed, "n_products": 0,
                "step_dist": {}, "top5_sim": 0.0, "top5_steps": []}

    # Synthesis depth distribution
    step_counts = df['num_steps'].value_counts().to_dict()
    df_sorted = df.sort_values('score', ascending=False)
    top5 = df_sorted.head(5)
    top5_sim = top5['score'].mean()
    top5_steps = top5['num_steps'].tolist()
    top5_synth = top5['synthesis'].tolist()

    return {
        "config": name,
        "time": elapsed,
        "n_products": len(df),
        "step_dist": step_counts,
        "top5_sim": top5_sim,
        "top5_steps": top5_steps,
        "top5_synth": top5_synth,
    }


def main():
    model_dir = os.path.join(PROJECT, "refs", "ReaSyn", "data", "trained_model")
    print("Loading models...", flush=True)
    models, fpindex, rxn_matrix = load_models(model_dir)
    print("Models loaded.", flush=True)

    # Test molecules (diverse drug-like)
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F",  # celecoxib
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
        "O=C(O)c1ccc(cc1)C(=O)c2ccccc2",    # benzophenone-4-carboxylic acid
    ]

    print(f"\nBenchmarking {len(CONFIGS)} configs × {len(test_smiles)} molecules")
    print("=" * 80)

    results = {}
    for config_name, config in CONFIGS.items():
        print(f"\n--- {config_name} ---")
        config_results = []
        for smi in test_smiles:
            mol = Molecule(smi)
            print(f"  {smi[:40]}...", end=" ", flush=True)
            r = benchmark_config(config_name, config, mol, models, fpindex, rxn_matrix)
            if r:
                print(f"{r['time']:.1f}s, {r['n_products']} prods, "
                      f"steps={r['step_dist']}, top5_sim={r['top5_sim']:.3f}")
                config_results.append(r)
            else:
                print("FAILED")
        results[config_name] = config_results

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<15s} | {'Time(s)':>8s} | {'Products':>8s} | "
          f"{'Top5 Sim':>8s} | {'Top5 Steps':>20s} | {'Max Steps':>9s}")
    print("-" * 80)
    for config_name, rs in results.items():
        if not rs:
            continue
        avg_time = np.mean([r['time'] for r in rs])
        avg_prods = np.mean([r['n_products'] for r in rs])
        avg_sim = np.mean([r['top5_sim'] for r in rs])
        all_steps = []
        for r in rs:
            all_steps.extend(r['top5_steps'])
        max_step = max(all_steps) if all_steps else 0
        step_str = f"avg={np.mean(all_steps):.1f}" if all_steps else "N/A"
        print(f"{config_name:<15s} | {avg_time:>8.1f} | {avg_prods:>8.0f} | "
              f"{avg_sim:>8.3f} | {step_str:>20s} | {max_step:>9d}")

    # Show sample synthesis routes from Full mode
    if "Full_8x4_ef" in results:
        print("\n--- Sample multi-step synthesis routes (Full mode) ---")
        for r in results["Full_8x4_ef"]:
            if r.get("top5_synth"):
                for j, syn in enumerate(r["top5_synth"][:2]):
                    steps = r["top5_steps"][j] if j < len(r["top5_steps"]) else "?"
                    print(f"  [{steps} steps] {syn[:120]}")


if __name__ == "__main__":
    main()
