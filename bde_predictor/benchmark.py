#!/usr/bin/env python
"""Benchmark PyTorch BDE predictor performance.

Usage:
    conda run -n rl4 python bde_predictor/benchmark.py
"""
import time
import torch
from bde_predictor.predict import BDEModel


def benchmark(n_mols=64, device='cpu', dtype=torch.float32):
    label = f"device={device}, dtype={str(dtype).split('.')[-1]}"
    print(f"\n--- {label}, n_mols={n_mols} ---")

    t0 = time.perf_counter()
    model = BDEModel('bde_predictor/weights/bde_db2_model3.npz', device=device, dtype=dtype)
    load_time = time.perf_counter() - t0
    print(f"Model load: {load_time:.3f}s")

    # Test molecules (antioxidant-like, varying sizes)
    smiles = [
        'c1ccc(O)cc1',                          # phenol (13 atoms+H)
        'Oc1ccc(O)cc1',                          # hydroquinone (14)
        'CC(C)(C)c1cc(O)cc(C(C)(C)C)c1O',       # BHT (38)
        'O=C(O)c1ccccc1O',                       # salicylic acid (16)
        'CC(=O)Oc1ccccc1O',                      # aspirin-like (19)
    ] * (n_mols // 5 + 1)
    smiles = smiles[:n_mols]

    # Warmup
    model.predict_oh_bde(smiles[:4])
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    bdes, valids = model.predict_oh_bde(smiles)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_valid = sum(valids)
    valid_bdes = [b for b, v in zip(bdes, valids) if v]
    print(f"Inference: {elapsed:.3f}s ({elapsed/n_mols*1000:.1f}ms/mol)")
    print(f"Valid: {n_valid}/{len(valids)}")
    if valid_bdes:
        print(f"BDE range: [{min(valid_bdes):.1f}, {max(valid_bdes):.1f}] kcal/mol")
    return elapsed


if __name__ == '__main__':
    for n in [16, 64, 256]:
        benchmark(n, 'cpu', torch.float32)

    if torch.cuda.is_available():
        for n in [16, 64, 256]:
            benchmark(n, 'cuda', torch.float32)
        for n in [16, 64, 256]:
            benchmark(n, 'cuda', torch.float16)
