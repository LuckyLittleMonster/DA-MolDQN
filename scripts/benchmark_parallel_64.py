"""Benchmark parallel processing of 64 molecules on a single GH200.

Tests different worker counts to find optimal throughput.
Uses FastSampler with P3 optimizations (fp16, skip_editflow, plim1, fast_copy_state).

Usage:
    cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn
    conda run -n reasyn --live-stream python ../../scripts/benchmark_parallel_64.py
    conda run -n reasyn --live-stream python ../../scripts/benchmark_parallel_64.py --workers 1 2 4 8
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


# ── Worker for parallel mode ──────────────────────────────────────────

class BenchWorker(mp.Process):
    """Lightweight worker that loads models and processes molecules from a queue."""

    def __init__(
        self,
        model_paths, task_queue, result_queue,
        gpu_lock, sampler_kwargs, evolve_kwargs,
        no_lock=False,
    ):
        super().__init__(daemon=True)
        self.model_paths = model_paths
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gpu_lock = gpu_lock
        self.sampler_kwargs = sampler_kwargs
        self.evolve_kwargs = evolve_kwargs
        self.no_lock = no_lock

    def run(self):
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        device = "cuda"

        # Load models (each worker loads its own copy due to spawn)
        models = []
        config = None
        for p in self.model_paths:
            ckpt = torch.load(p, map_location="cpu")
            config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
            m = ReaSyn(config.model).to(device)
            m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
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
            t0 = time.perf_counter()
            try:
                sampler = FastSampler(
                    fpindex=fpindex, rxn_matrix=rxn_matrix,
                    mol=mol, model=models,
                    **self.sampler_kwargs,
                )
                tl = TimeLimit(120)
                lock = None if self.no_lock else self.gpu_lock
                sampler.evolve(gpu_lock=lock, time_limit=tl,
                               **self.evolve_kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                df = sampler.get_dataframe()
                n = len(df)
                max_score = df['score'].max() if n > 0 else 0.0
                mean_score = df['score'].mean() if n > 0 else 0.0
                self.result_queue.put((mol_idx, elapsed, n, max_score, mean_score))
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  Worker {self.name} mol {mol_idx} ERROR: {e}")
                self.result_queue.put((mol_idx, elapsed, 0, 0.0, 0.0))
            self.task_queue.task_done()


def run_parallel(mols, model_paths, num_workers, sampler_kwargs, evolve_kwargs,
                  no_lock=False):
    """Run all molecules through a pool of workers. Returns (total_time, results_list)."""
    task_queue = mp.JoinableQueue()
    result_queue = mp.Queue()
    gpu_lock = mp.Lock()

    # Start workers
    workers = []
    for _ in range(num_workers):
        w = BenchWorker(
            model_paths, task_queue, result_queue,
            gpu_lock, sampler_kwargs, evolve_kwargs,
            no_lock=no_lock,
        )
        w.start()
        workers.append(w)

    # Submit all molecules
    t_start = time.perf_counter()
    for i, mol in enumerate(mols):
        task_queue.put((i, mol))

    # Send poison pills
    for _ in workers:
        task_queue.put(None)

    # Wait for all tasks
    task_queue.join()
    total_time = time.perf_counter() - t_start

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    # Cleanup
    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()

    results.sort(key=lambda x: x[0])
    return total_time, results


def run_sequential(mols, model_paths, sampler_kwargs, evolve_kwargs):
    """Run all molecules sequentially in current process. Returns (total_time, results_list)."""
    device = "cuda"
    models = []
    config = None
    for p in model_paths:
        ckpt = torch.load(p, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        m = ReaSyn(config.model).to(device)
        m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        m.eval()
        models.append(m)
    fpindex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix = pickle.load(open(config.chem.rxn_matrix, "rb"))

    results = []
    t_start = time.perf_counter()
    for i, mol in enumerate(mols):
        t0 = time.perf_counter()
        sampler = FastSampler(
            fpindex=fpindex, rxn_matrix=rxn_matrix,
            mol=mol, model=models,
            **sampler_kwargs,
        )
        tl = TimeLimit(120)
        sampler.evolve(gpu_lock=None, time_limit=tl, **evolve_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        df = sampler.get_dataframe()
        n = len(df)
        max_score = df['score'].max() if n > 0 else 0.0
        mean_score = df['score'].mean() if n > 0 else 0.0
        results.append((i, elapsed, n, max_score, mean_score))

        if (i + 1) % 8 == 0 or i == len(mols) - 1:
            avg_t = np.mean([r[1] for r in results])
            print(f"  [{i+1}/{len(mols)}] avg {avg_t:.1f}s/mol, "
                  f"last={elapsed:.1f}s, #res={n}, max={max_score:.3f}")

    total_time = time.perf_counter() - t_start
    return total_time, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mols", type=int, default=64)
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--search_width", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Worker counts to test. 0 = sequential (no spawn)")
    parser.add_argument("--no_lock", action="store_true",
                        help="Disable gpu_lock (let CUDA time-share across workers)")
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
    print(f"Benchmark: {len(mols)} molecules, ex={args.exhaustiveness}, "
          f"sw={args.search_width}, workers={args.workers}")

    # P3 sampler settings
    sampler_kwargs = {
        "factor": args.search_width,
        "max_active_states": args.exhaustiveness,
        "use_fp16": True,
        "max_branch_states": args.exhaustiveness // 2,
        "skip_editflow": True,
        "rxn_product_limit": 1,
    }
    evolve_kwargs = {
        "max_evolve_steps": 8,
        "num_cycles": 1,
        "num_editflow_samples": 10,
        "num_editflow_steps": 50,
    }

    all_summaries = []

    for nw in args.workers:
        print(f"\n{'='*70}")
        torch.cuda.reset_peak_memory_stats()

        if nw == 0:
            print(f"Sequential (no spawn, 1 process)")
            total_time, results = run_sequential(
                mols, model_paths, sampler_kwargs, evolve_kwargs,
            )
        else:
            lock_str = "no_lock" if args.no_lock else "gpu_lock"
            print(f"Parallel: {nw} workers ({lock_str})")
            total_time, results = run_parallel(
                mols, model_paths, nw, sampler_kwargs, evolve_kwargs,
                no_lock=args.no_lock,
            )

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        # Stats
        times = [r[1] for r in results]
        n_results = [r[2] for r in results]
        max_scores = [r[3] for r in results]
        mean_scores = [r[4] for r in results]

        throughput = len(mols) / total_time if total_time > 0 else 0

        summary = {
            'workers': nw if nw > 0 else 'seq',
            'total_s': total_time,
            'throughput': throughput,
            'avg_mol_s': np.mean(times),
            'max_mol_s': np.max(times),
            'min_mol_s': np.min(times),
            'peak_mem_mb': peak_mem,
            'avg_results': np.mean(n_results),
            'avg_max_score': np.mean(max_scores),
            'avg_mean_score': np.mean(mean_scores),
        }
        all_summaries.append(summary)

        print(f"  Total: {total_time:.1f}s | Throughput: {throughput:.2f} mol/s")
        print(f"  Per-mol: avg={np.mean(times):.1f}s, min={np.min(times):.1f}s, "
              f"max={np.max(times):.1f}s")
        print(f"  Peak GPU mem: {peak_mem:.0f} MB")
        print(f"  Quality: avg_max_score={np.mean(max_scores):.3f}, "
              f"avg_mean_score={np.mean(mean_scores):.3f}, "
              f"avg_results={np.mean(n_results):.1f}")

    # Final summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    seq_time = all_summaries[0]['total_s']
    print(f"\n{'Workers':>8s} | {'Total':>7s} | {'Speedup':>8s} | {'Throughput':>11s} | "
          f"{'AvgMol':>7s} | {'PeakMem':>8s} | {'AvgMaxScr':>9s} | {'AvgRes':>6s}")
    print("-" * 85)
    for s in all_summaries:
        speedup = seq_time / s['total_s'] if s['total_s'] > 0 else 0
        print(f"{str(s['workers']):>8s} | {s['total_s']:6.1f}s | {speedup:7.2f}x | "
              f"{s['throughput']:8.2f} mol/s | {s['avg_mol_s']:6.1f}s | "
              f"{s['peak_mem_mb']:7.0f}MB | {s['avg_max_score']:9.3f} | "
              f"{s['avg_results']:6.1f}")


if __name__ == "__main__":
    main()
