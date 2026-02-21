"""Benchmark RL-style 5-step molecular optimization with multiple workers.

Simulates RL episodes: each molecule goes through 5 steps of ReaSyn (L1a config).
At each step, pick the highest-score candidate as the next molecule.
Tests different worker counts to find optimal throughput.

Usage:
    cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn
    conda run -n reasyn --live-stream python ../../scripts/benchmark_rl_workers.py
    conda run -n reasyn --live-stream python ../../scripts/benchmark_rl_workers.py --workers 1 4 8 16 32 64
"""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
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


# L1a config: ex=16, sw=8, 4 evolve steps
L1A_SAMPLER_KWARGS = {
    "use_fp16": True,
    "max_branch_states": 8,   # ex//2
    "skip_editflow": True,
    "rxn_product_limit": 1,
}
L1A_EVOLVE_KWARGS = {
    "max_evolve_steps": 4,
    "num_cycles": 1,
    "num_editflow_samples": 10,
    "num_editflow_steps": 50,
}
L1A_EX = 16
L1A_SW = 8


def run_episode(mol, models, fpindex, rxn_matrix, n_steps=5):
    """Run one RL episode: n_steps of ReaSyn optimization.

    Returns dict with per-step timings and final molecule info.
    """
    current_mol = mol
    step_times = []
    step_results = []

    for step_i in range(n_steps):
        t0 = time.perf_counter()
        sampler = FastSampler(
            fpindex=fpindex, rxn_matrix=rxn_matrix,
            mol=current_mol, model=models,
            factor=L1A_SW, max_active_states=L1A_EX,
            **L1A_SAMPLER_KWARGS,
        )
        tl = TimeLimit(30)  # 30s timeout per step
        sampler.evolve(gpu_lock=None, time_limit=tl, **L1A_EVOLVE_KWARGS)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        df = sampler.get_dataframe()
        n = len(df)
        if n > 0:
            best = df.loc[df['score'].idxmax()]
            best_smiles = best['smiles']
            best_score = best['score']
            # Move to best candidate for next step
            current_mol = Molecule(best_smiles)
        else:
            best_smiles = current_mol.csmiles
            best_score = 0.0

        step_times.append(elapsed)
        step_results.append({
            'step': step_i,
            'time_s': elapsed,
            'n_candidates': n,
            'best_score': best_score,
            'best_smiles': best_smiles,
        })

    return {
        'init_smiles': mol.csmiles,
        'final_smiles': current_mol.csmiles,
        'total_time': sum(step_times),
        'step_times': step_times,
        'step_results': step_results,
    }


class RLWorker(mp.Process):
    """Worker that runs RL episodes (5 steps each) for assigned molecules."""

    def __init__(self, model_paths, task_queue, result_queue, n_steps):
        super().__init__(daemon=True)
        self.model_paths = model_paths
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.n_steps = n_steps

    def run(self):
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        device = "cuda"

        models = []
        config = None
        for p in self.model_paths:
            ckpt = torch.load(p, map_location="cpu")
            config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
            m = ReaSyn(config.model)
            m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
            m = m.half().to(device)  # fp16 weights to save GPU memory (~1.2GB vs 2.4GB)
            m.eval()
            models.append(m)
        fpindex = pickle.load(open(config.chem.fpindex, "rb"))
        rxn_matrix = pickle.load(open(config.chem.rxn_matrix, "rb"))

        while True:
            item = self.task_queue.get()
            if item is None:
                self.task_queue.task_done()
                break
            mol_idx, mol = item
            try:
                result = run_episode(mol, models, fpindex, rxn_matrix, self.n_steps)
                result['mol_idx'] = mol_idx
                self.result_queue.put(result)
            except Exception as e:
                print(f"  Worker {self.name} mol {mol_idx} ERROR: {e}")
                self.result_queue.put({
                    'mol_idx': mol_idx,
                    'init_smiles': mol.csmiles,
                    'final_smiles': mol.csmiles,
                    'total_time': 0,
                    'step_times': [0] * self.n_steps,
                    'step_results': [],
                    'error': str(e),
                })
            self.task_queue.task_done()


def run_with_workers(mols, model_paths, num_workers, n_steps):
    """Run all molecules through worker pool. Returns (total_time, results)."""
    task_queue = mp.JoinableQueue()
    result_queue = mp.Queue()

    workers = []
    for _ in range(num_workers):
        w = RLWorker(model_paths, task_queue, result_queue, n_steps)
        w.start()
        workers.append(w)

    t_start = time.perf_counter()
    for i, mol in enumerate(mols):
        task_queue.put((i, mol))
    for _ in workers:
        task_queue.put(None)

    task_queue.join()
    total_time = time.perf_counter() - t_start

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()

    results.sort(key=lambda x: x['mol_idx'])
    return total_time, results


def run_sequential(mols, model_paths, n_steps):
    """Run all molecules sequentially in current process."""
    device = "cuda"
    models = []
    config = None
    for p in model_paths:
        ckpt = torch.load(p, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        m = ReaSyn(config.model)
        m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        m = m.half().to(device)  # fp16 weights
        m.eval()
        models.append(m)
    fpindex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix = pickle.load(open(config.chem.rxn_matrix, "rb"))

    results = []
    t_start = time.perf_counter()
    for i, mol in enumerate(mols):
        result = run_episode(mol, models, fpindex, rxn_matrix, n_steps)
        result['mol_idx'] = i
        results.append(result)

        if (i + 1) % 8 == 0 or i == len(mols) - 1:
            avg_ep = np.mean([r['total_time'] for r in results])
            avg_step = np.mean([t for r in results for t in r['step_times']])
            print(f"  [{i+1}/{len(mols)}] avg_episode={avg_ep:.1f}s, avg_step={avg_step:.2f}s")

    total_time = time.perf_counter() - t_start
    return total_time, results


def print_results(label, total_time, results, n_mols, n_steps):
    """Print summary of benchmark results."""
    ep_times = [r['total_time'] for r in results]
    step_times = [t for r in results for t in r['step_times']]
    n_zero = sum(1 for r in results
                 for sr in r.get('step_results', [])
                 if sr['n_candidates'] == 0)
    total_steps = n_mols * n_steps
    candidates = [sr['n_candidates'] for r in results for sr in r.get('step_results', [])]

    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Throughput: {n_mols / total_time:.2f} mol/s "
          f"({total_steps / total_time:.1f} steps/s)")
    print(f"  Per-episode: avg={np.mean(ep_times):.2f}s, "
          f"min={np.min(ep_times):.2f}s, max={np.max(ep_times):.2f}s")
    print(f"  Per-step: avg={np.mean(step_times):.2f}s, "
          f"min={np.min(step_times):.2f}s, max={np.max(step_times):.2f}s")
    if candidates:
        print(f"  Candidates/step: avg={np.mean(candidates):.1f}, "
              f"zero-result steps: {n_zero}/{total_steps} ({100*n_zero/total_steps:.0f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mols", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    model_dir = REASYN_ROOT / "data" / "trained_model"
    zinc_path = REASYN_ROOT / "data" / "zinc_first64.txt"
    model_paths = [
        model_dir / "nv-reasyn-ar-166m-v2.ckpt",
        model_dir / "nv-reasyn-eb-174m-v2.ckpt",
    ]

    with open(zinc_path) as f:
        lines = f.read().strip().split("\n")
    if lines[0].upper() == "SMILES":
        lines = lines[1:]
    mols = [Molecule(s.strip()) for s in lines[:args.num_mols]]

    print(f"RL Benchmark: {len(mols)} mols × {args.n_steps} steps/episode")
    print(f"Config: L1a (ex={L1A_EX}, sw={L1A_SW}, "
          f"steps={L1A_EVOLVE_KWARGS['max_evolve_steps']})")
    print(f"Workers to test: {args.workers}\n")

    all_summaries = []

    for nw in args.workers:
        print(f"{'='*70}")
        torch.cuda.reset_peak_memory_stats()

        if nw <= 1:
            print(f"Sequential (1 process, no spawn)")
            total_time, results = run_sequential(mols, model_paths, args.n_steps)
        else:
            print(f"Parallel: {nw} workers (no gpu_lock)")
            total_time, results = run_with_workers(
                mols, model_paths, nw, args.n_steps,
            )

        print_results(f"{nw}w", total_time, results, len(mols), args.n_steps)

        ep_times = [r['total_time'] for r in results]
        step_times = [t for r in results for t in r['step_times']]
        candidates = [sr['n_candidates'] for r in results
                      for sr in r.get('step_results', [])]

        all_summaries.append({
            'workers': nw,
            'total_s': total_time,
            'throughput_mol': len(mols) / total_time,
            'throughput_step': len(mols) * args.n_steps / total_time,
            'avg_episode_s': np.mean(ep_times),
            'avg_step_s': np.mean(step_times),
            'avg_candidates': np.mean(candidates) if candidates else 0,
        })

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    seq_time = all_summaries[0]['total_s']

    print(f"\n{'Workers':>7s} | {'Total':>7s} | {'Speedup':>7s} | "
          f"{'mol/s':>6s} | {'step/s':>6s} | "
          f"{'Ep(s)':>6s} | {'Step(s)':>7s} | {'Cand':>5s}")
    print("-" * 75)

    for s in all_summaries:
        spd = seq_time / s['total_s'] if s['total_s'] > 0 else 0
        print(f"{s['workers']:>7d} | {s['total_s']:6.1f}s | {spd:6.2f}x | "
              f"{s['throughput_mol']:5.2f} | {s['throughput_step']:5.1f} | "
              f"{s['avg_episode_s']:6.2f} | {s['avg_step_s']:7.2f} | "
              f"{s['avg_candidates']:5.1f}")

    # Estimate optimal config
    best = min(all_summaries, key=lambda s: s['total_s'])
    print(f"\nBest: {best['workers']} workers, "
          f"{best['total_s']:.1f}s total, "
          f"{best['throughput_step']:.1f} steps/s")


if __name__ == "__main__":
    main()
