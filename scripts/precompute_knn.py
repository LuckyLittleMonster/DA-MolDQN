#!/usr/bin/env python
"""
Precompute Morgan FP kNN graph using nvmolkit GPU acceleration.

This replaces the CPU-based build_morgan_knn() in training, reducing initialization
from ~2.5 hours (655K mols, CPU) to ~2 minutes (GPU).

Requirements:
    conda env: nvmolkit (has nvmolkit + rdkit + torch)

Usage:
    conda run -n nvmolkit python scripts/precompute_knn.py \
        --data_dir Data/uspto_full --k 200 --output Data/precomputed/knn_full_k200.pkl

    conda run -n nvmolkit python scripts/precompute_knn.py \
        --data_dir Data/uspto --k 200 --output Data/precomputed/knn_50k_k200.pkl
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_knn_nvmolkit(smiles_list, k=200, fp_bits=2048, fp_radius=2, batch_size=5000):
    """Build kNN graph using nvmolkit GPU-accelerated fingerprints and similarity.

    Returns the same format as build_morgan_knn():
        dict[int, list[(int, float)]] — mol_idx -> [(neighbor_idx, tanimoto_sim), ...]
    """
    import torch
    from rdkit import Chem
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarity

    n = len(smiles_list)
    print(f"Building kNN graph with nvmolkit GPU acceleration")
    print(f"  n={n}, k={k}, fp_bits={fp_bits}, radius={fp_radius}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    t_start = time.time()

    # Convert SMILES to RDKit mols
    t0 = time.time()
    mols = []
    null_count = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            mols.append(Chem.MolFromSmiles('C'))  # placeholder
            null_count += 1
        else:
            mols.append(m)
    print(f"  Parsed {n} SMILES ({null_count} failed, replaced with 'C'): {time.time()-t0:.1f}s")

    # Compute fingerprints on GPU
    t0 = time.time()
    gen = MorganFingerprintGenerator(radius=fp_radius, fpSize=fp_bits)
    fps_async = gen.GetFingerprints(mols, num_threads=0)  # 0 = all threads
    fps_tensor = fps_async.torch()  # packed FP tensor on GPU
    print(f"  FP generation: {time.time()-t0:.1f}s, shape={fps_tensor.shape}, device={fps_tensor.device}")

    # Batched cross-similarity + top-k on GPU
    t0 = time.time()
    all_topk_vals = []
    all_topk_idxs = []

    n_batches = (n + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_fps = fps_tensor[start:end]

        # Cross-similarity on GPU: (batch, N)
        sim_async = crossTanimotoSimilarity(batch_fps, fps_tensor)
        sim_gpu = sim_async.torch()

        # Mask self-similarity
        for i in range(start, end):
            sim_gpu[i - start, i] = -1.0

        # Top-k on GPU
        topk_vals, topk_idxs = torch.topk(sim_gpu, k, dim=1)
        all_topk_vals.append(topk_vals.cpu())
        all_topk_idxs.append(topk_idxs.cpu())

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / n_batches * 100
            print(f"    Batch {batch_idx+1}/{n_batches} ({pct:.0f}%), elapsed={elapsed:.1f}s")

    t_knn = time.time() - t0
    print(f"  kNN extraction: {t_knn:.1f}s")

    # Assemble kNN graph (same format as build_morgan_knn)
    t0 = time.time()
    topk_vals = torch.cat(all_topk_vals, dim=0).numpy()
    topk_idxs = torch.cat(all_topk_idxs, dim=0).numpy()

    knn_graph = {}
    for i in range(n):
        knn_graph[i] = [(int(topk_idxs[i, j]), float(topk_vals[i, j])) for j in range(k)]

    print(f"  Graph assembly: {time.time()-t0:.1f}s")

    total = time.time() - t_start
    print(f"  Total: {total:.1f}s ({total/60:.1f}min)")

    # Stats
    avg_sim = np.mean([knn_graph[i][0][1] for i in range(n)])
    min_sim = np.mean([knn_graph[i][-1][1] for i in range(n)])
    print(f"  Avg top-1 similarity: {avg_sim:.4f}")
    print(f"  Avg top-{k} similarity: {min_sim:.4f}")

    return knn_graph


def build_knn_cpu(smiles_list, k=200, fp_bits=2048, fp_radius=2):
    """CPU fallback using RDKit (same as original build_morgan_knn)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs

    n = len(smiles_list)
    print(f"Building kNN graph with CPU (RDKit)")
    print(f"  n={n}, k={k}, fp_bits={fp_bits}, radius={fp_radius}")
    t_start = time.time()

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=fp_bits, radius=fp_radius)
    features = np.zeros((n, fp_bits), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros(fp_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features[i] = arr

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features_norm = (features / norms).astype(np.float32)

    knn_graph = {}
    batch_size = 500
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = features_norm[start:end] @ features_norm.T
        for i in range(start, end):
            local_i = i - start
            sims[local_i, i] = -2.0
            top_k_idx = np.argpartition(sims[local_i], -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(-sims[local_i, top_k_idx])]
            knn_graph[i] = [(int(j), float(sims[local_i, j])) for j in top_k_idx]

    total = time.time() - t_start
    print(f"  Total: {total:.1f}s ({total/60:.1f}min)")
    return knn_graph


def main():
    parser = argparse.ArgumentParser(description='Precompute Morgan FP kNN graph')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory (e.g., Data/uspto or Data/uspto_full)')
    parser.add_argument('--k', type=int, default=200,
                        help='Number of nearest neighbors')
    parser.add_argument('--fp_bits', type=int, default=2048,
                        help='Morgan fingerprint size')
    parser.add_argument('--fp_radius', type=int, default=2,
                        help='Morgan fingerprint radius')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle path (default: <data_dir>/knn_k<k>.pkl)')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size for GPU similarity computation')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU computation (no nvmolkit)')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        args.output = os.path.join(args.data_dir, f'knn_k{args.k}.pkl')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load SMILES from reaction database
    from model_reactions.reactant_predictor import ReactionDatabase
    cache_path = os.path.join(args.data_dir, 'reaction_db_cache.pkl')
    db = ReactionDatabase(args.data_dir)
    if os.path.exists(cache_path):
        print(f"Loading cached reaction database from {cache_path}...")
        db.load(cache_path)
    else:
        print(f"Building reaction database from {args.data_dir}...")
        db.build()

    smiles_list = db.reactant_smiles
    print(f"Molecules: {len(smiles_list)}")

    # Build kNN
    use_gpu = not args.cpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                use_gpu = False
            else:
                import nvmolkit
        except ImportError:
            print("nvmolkit not available, falling back to CPU")
            use_gpu = False

    if use_gpu:
        knn_graph = build_knn_nvmolkit(
            smiles_list, k=args.k,
            fp_bits=args.fp_bits, fp_radius=args.fp_radius,
            batch_size=args.batch_size,
        )
    else:
        knn_graph = build_knn_cpu(
            smiles_list, k=args.k,
            fp_bits=args.fp_bits, fp_radius=args.fp_radius,
        )

    # Save
    print(f"\nSaving kNN graph to {args.output}...")
    save_data = {
        'knn_graph': knn_graph,
        'k': args.k,
        'fp_bits': args.fp_bits,
        'fp_radius': args.fp_radius,
        'n_molecules': len(smiles_list),
        'method': 'nvmolkit_tanimoto' if use_gpu else 'rdkit_cosine',
    }
    with open(args.output, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved: {file_size:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
