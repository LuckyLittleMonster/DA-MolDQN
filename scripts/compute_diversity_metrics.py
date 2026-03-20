#!/usr/bin/env python3
"""
Compute diversity metrics for reviewer response R1-07.

Metrics:
1. FBC (Fingerprint Bit Coverage) at various group sizes — reproduction of Table 5
2. Shannon entropy of Morgan FP bit distribution
3. Bemis-Murcko scaffold diversity (unique scaffolds / total molecules)
4. Internal diversity (mean pairwise Tanimoto distance)
5. Unique molecule count and ratios

Data sources:
- MolDQN (individual): trial 600 best paths (256 runs × 1 mol)
- MT-MolDQN (8 mols): trial 22000 best paths (8 ranks × 1 mol)
- DA-MolDQN (256 mols): trial 2912 best paths (64 ranks × 4 mols)
- AODB: Data/AODB_s.csv
- ChEMBL: Data/chembl_31_10k.txt
"""

import argparse
import glob
import gzip
import os
import pickle
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_moldqn_mols(base_dir):
    """Load MolDQN (individual) molecules from trial 600 best paths."""
    pattern = os.path.join(base_dir, "Experiments/trial_600*_best_actions_path.pickle")
    files = sorted(glob.glob(pattern))
    smiles_set = set()
    for f in files:
        with open(f, "rb") as fh:
            paths, rewards, bdes, ips = pickle.load(fh)
        for path in paths:
            for mol in path:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    smiles_set.add(smi)
    return list(smiles_set)


def load_mt_moldqn_mols(base_dir):
    """Load MT-MolDQN molecules from trial 22000 best paths."""
    pattern = os.path.join(base_dir, "Experiments/trial_22000_rank_*_best_actions_path.pickle")
    files = sorted(glob.glob(pattern))
    smiles_set = set()
    for f in files:
        with open(f, "rb") as fh:
            paths, rewards, bdes, ips = pickle.load(fh)
        for path in paths:
            for mol in path:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    smiles_set.add(smi)
    return list(smiles_set)


def load_da_moldqn_mols(experiments_dir):
    """Load DA-MolDQN molecules from trial 2912 path pickles (64 ranks)."""
    smiles_set = set()
    for rank in range(64):
        path = os.path.join(experiments_dir, f"ablation_rw_2912_{rank}_path.pickle")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Extract from 'last' (all saved episodes)
        for episode_data in data.get("last", []):
            mol_paths, rewards = episode_data
            for mol_path in mol_paths:
                for mol in mol_path:
                    if mol is not None:
                        smi = Chem.MolToSmiles(mol)
                        smiles_set.add(smi)
        # Extract from 'top'
        for score, tdata in data.get("top", []):
            for mol in tdata.get("path", []):
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    smiles_set.add(smi)
    return list(smiles_set)


def load_aodb_mols(data_dir):
    """Load AODB molecules from AODB_s.csv."""
    import csv
    path = os.path.join(data_dir, "AODB_s.csv")
    smiles_list = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            smi = row[0].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))
    return list(set(smiles_list))


def load_chembl_mols(data_dir):
    """Load ChEMBL molecules from chembl_31_10k.txt."""
    path = os.path.join(data_dir, "chembl_31_10k.txt")
    smiles_list = []
    with open(path) as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))
    return list(set(smiles_list))


def compute_morgan_fps(smiles_list, radius=3, n_bits=2048):
    """Compute Morgan fingerprints as ExplicitBitVects."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
    return fps


def compute_fbc(fps, group_size, n_trials=100, n_bits=2048):
    """
    Compute Fingerprint Bit Coverage (FBC) at a given group size.
    Randomly sample groups of `group_size` molecules, OR their FPs,
    count covered bits / total bits. Average over n_trials.
    """
    if len(fps) < group_size:
        # If not enough molecules, use all of them (single group)
        combined = DataStructs.ExplicitBitVect(n_bits)
        for fp in fps:
            combined |= fp
        return combined.GetNumOnBits() / n_bits

    rng = np.random.RandomState(42)
    coverages = []
    for _ in range(n_trials):
        indices = rng.choice(len(fps), size=group_size, replace=False)
        combined = DataStructs.ExplicitBitVect(n_bits)
        for idx in indices:
            combined |= fps[idx]
        coverages.append(combined.GetNumOnBits() / n_bits)
    return np.mean(coverages)


def compute_shannon_entropy(fps, n_bits=2048):
    """
    Compute Shannon entropy of Morgan FP bit distribution.
    For each bit position, compute p = fraction of molecules with that bit ON.
    H = -sum(p * log2(p) + (1-p) * log2(1-p)) / n_bits (normalized).
    """
    n_mols = len(fps)
    if n_mols == 0:
        return 0.0

    bit_counts = np.zeros(n_bits)
    for fp in fps:
        on_bits = list(fp.GetOnBits())
        bit_counts[on_bits] += 1

    # Fraction of molecules with each bit ON
    p = bit_counts / n_mols
    # Avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)

    # Per-bit entropy (binary entropy)
    bit_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    # Total Shannon entropy (sum, normalized by n_bits)
    total_entropy = np.sum(bit_entropy)
    max_entropy = n_bits  # max when all p=0.5
    return total_entropy / max_entropy  # normalized [0, 1]


def compute_bit_coverage_entropy(fps, n_bits=2048):
    """
    Alternative: Shannon entropy over the ON-bit frequency distribution.
    Treat the frequency of each ON bit across molecules as a probability distribution.
    """
    n_mols = len(fps)
    if n_mols == 0:
        return 0.0

    bit_counts = np.zeros(n_bits)
    for fp in fps:
        on_bits = list(fp.GetOnBits())
        bit_counts[on_bits] += 1

    # Only consider bits that are ON in at least one molecule
    on_mask = bit_counts > 0
    counts = bit_counts[on_mask]
    # Normalize to probability distribution
    probs = counts / counts.sum()
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    # Max entropy = log2(n_active_bits)
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy, max_entropy, entropy / max_entropy


def compute_scaffold_diversity(smiles_list):
    """
    Compute Bemis-Murcko scaffold diversity.
    Returns (n_unique_scaffolds, n_molecules, ratio).
    """
    scaffolds = set()
    n_valid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            n_valid += 1
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smi = Chem.MolToSmiles(scaffold)
                scaffolds.add(scaffold_smi)
            except:
                pass
    return len(scaffolds), n_valid, len(scaffolds) / max(n_valid, 1)


def compute_internal_diversity(fps, max_pairs=10000):
    """
    Compute internal diversity = 1 - mean pairwise Tanimoto similarity.
    For large sets, sample pairs randomly.
    """
    n = len(fps)
    if n < 2:
        return 0.0

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        # Compute all pairwise similarities
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        mean_sim = np.mean(sims)
    else:
        # Sample pairs
        rng = np.random.RandomState(42)
        sims = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        mean_sim = np.mean(sims)

    return 1.0 - mean_sim


def compute_fragment_entropy(smiles_list, max_mols=5000):
    """
    Compute Shannon entropy over BRICS fragment distribution.
    Higher entropy = more diverse fragment space.
    For large datasets, sample max_mols molecules.
    """
    from rdkit.Chem import BRICS
    fragment_counts = Counter()

    if len(smiles_list) > max_mols:
        rng = np.random.RandomState(42)
        smiles_list = list(rng.choice(smiles_list, size=max_mols, replace=False))

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                frags = BRICS.BRICSDecompose(mol)
                for frag in frags:
                    fragment_counts[frag] += 1
            except:
                pass

    if not fragment_counts:
        return 0.0, 0

    total = sum(fragment_counts.values())
    probs = np.array(list(fragment_counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs))
    return entropy, len(fragment_counts)


def main():
    parser = argparse.ArgumentParser(description="Compute diversity metrics for R1-07")
    parser.add_argument("--da_moldqn_dir", default="/shared/data1/Users/l1062811/git/DA-MolDQN")
    parser.add_argument("--rl4_working_dir", default="/shared/data1/Users/l1062811/git/RL4-working/RL4-working")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--fbc_only", action="store_true", help="Only compute FBC")
    args = parser.parse_args()

    # Load molecules
    print("Loading molecules...")
    datasets = {}

    print("  MolDQN (trial 600)...", end="", flush=True)
    datasets["MolDQN"] = load_moldqn_mols(args.rl4_working_dir)
    print(f" {len(datasets['MolDQN'])} unique")

    print("  MT-MolDQN (trial 22000)...", end="", flush=True)
    datasets["MT-MolDQN"] = load_mt_moldqn_mols(args.rl4_working_dir)
    print(f" {len(datasets['MT-MolDQN'])} unique")

    print("  DA-MolDQN (trial 2912)...", end="", flush=True)
    datasets["DA-MolDQN"] = load_da_moldqn_mols(os.path.join(args.da_moldqn_dir, "Experiments"))
    print(f" {len(datasets['DA-MolDQN'])} unique")

    print("  AODB...", end="", flush=True)
    datasets["AODB"] = load_aodb_mols(os.path.join(args.da_moldqn_dir, "Data"))
    print(f" {len(datasets['AODB'])} unique")

    print("  ChEMBL...", end="", flush=True)
    datasets["ChEMBL"] = load_chembl_mols(os.path.join(args.da_moldqn_dir, "Data"))
    print(f" {len(datasets['ChEMBL'])} unique")

    # Compute fingerprints
    print("\nComputing Morgan fingerprints (radius=3, 2048 bits)...")
    fps_dict = {}
    for name, smiles_list in datasets.items():
        fps_dict[name] = compute_morgan_fps(smiles_list)
        print(f"  {name}: {len(fps_dict[name])} valid FPs")

    # Compute metrics
    print("\n" + "=" * 80)
    print("DIVERSITY METRICS")
    print("=" * 80)

    algo_order = ["MolDQN", "MT-MolDQN", "DA-MolDQN", "AODB", "ChEMBL"]
    results = {}

    # 1. FBC at group size 512 (Table 5 comparison)
    print("\n--- FBC (group size 512) ---")
    for name in algo_order:
        fbc = compute_fbc(fps_dict[name], group_size=512)
        unexplored = (1 - fbc) * 100
        results.setdefault(name, {})["FBC_512"] = fbc
        results[name]["Unexplored_512"] = unexplored
        print(f"  {name:12s}: FBC={fbc:.4f}, Unexplored={unexplored:.1f}%")

    # FBC at all group sizes for comparison with original
    print("\n--- FBC at various group sizes ---")
    group_sizes = [128, 256, 512, 1024, 2048, 4096]
    for gs in group_sizes:
        row = []
        for name in algo_order:
            fbc = compute_fbc(fps_dict[name], group_size=gs)
            row.append(fbc)
        print(f"  Group {gs:4d}: {' '.join(f'{v:.4f}' for v in row)}")

    if args.fbc_only:
        return

    # 2. Shannon entropy of FP bits
    print("\n--- Shannon Entropy (normalized) ---")
    for name in algo_order:
        h = compute_shannon_entropy(fps_dict[name])
        results[name]["Shannon_H"] = h
        print(f"  {name:12s}: H={h:.4f}")

    # 3. Bit coverage entropy
    print("\n--- Bit Coverage Entropy ---")
    for name in algo_order:
        h, h_max, h_norm = compute_bit_coverage_entropy(fps_dict[name])
        results[name]["Bit_H"] = h
        results[name]["Bit_H_norm"] = h_norm
        n_active = sum(1 for _ in fps_dict[name][0].GetOnBits()) if fps_dict[name] else 0
        # Count total active bits
        combined = DataStructs.ExplicitBitVect(2048)
        for fp in fps_dict[name]:
            combined |= fp
        n_active_total = combined.GetNumOnBits()
        results[name]["Active_bits"] = n_active_total
        print(f"  {name:12s}: H={h:.2f} bits, H_max={h_max:.2f}, H_norm={h_norm:.4f}, active_bits={n_active_total}")

    # 4. Bemis-Murcko scaffold diversity
    print("\n--- Scaffold Diversity ---")
    for name in algo_order:
        n_scaff, n_mol, ratio = compute_scaffold_diversity(datasets[name])
        results[name]["N_scaffolds"] = n_scaff
        results[name]["Scaffold_ratio"] = ratio
        print(f"  {name:12s}: {n_scaff} scaffolds / {n_mol} mols = {ratio:.4f}")

    # 5. Internal diversity
    print("\n--- Internal Diversity (1 - mean Tanimoto sim) ---")
    for name in algo_order:
        div = compute_internal_diversity(fps_dict[name])
        results[name]["IntDiv"] = div
        print(f"  {name:12s}: {div:.4f}")

    # 6. BRICS fragment entropy
    print("\n--- BRICS Fragment Entropy ---")
    for name in algo_order:
        h, n_frags = compute_fragment_entropy(datasets[name])
        results[name]["Frag_H"] = h
        results[name]["N_fragments"] = n_frags
        print(f"  {name:12s}: H={h:.2f} bits, {n_frags} unique fragments")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (for paper)")
    print("=" * 80)
    header = f"{'Metric':25s} " + " ".join(f"{n:>12s}" for n in algo_order)
    print(header)
    print("-" * len(header))

    metrics = [
        ("Unique Molecules", lambda r, n: f"{len(datasets[n]):>12d}"),
        ("Unique Scaffolds", lambda r, n: f"{r['N_scaffolds']:>12d}"),
        ("Scaffold Diversity", lambda r, n: f"{r['Scaffold_ratio']:>12.3f}"),
        ("Internal Diversity", lambda r, n: f"{r['IntDiv']:>12.4f}"),
        ("Shannon Entropy (norm)", lambda r, n: f"{r['Shannon_H']:>12.4f}"),
        ("BRICS Fragments", lambda r, n: f"{r['N_fragments']:>12d}"),
        ("BRICS Entropy (bits)", lambda r, n: f"{r['Frag_H']:>12.2f}"),
        ("FBC@512", lambda r, n: f"{r['FBC_512']:>12.4f}"),
        ("Unexplored@512 (%)", lambda r, n: f"{r['Unexplored_512']:>12.1f}"),
    ]

    for metric_name, fmt_fn in metrics:
        row = f"{metric_name:25s} "
        for name in algo_order:
            row += fmt_fn(results[name], name)
        print(row)

    # Save CSV if requested
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric"] + algo_order)
            for metric_name, fmt_fn in metrics:
                row = [metric_name]
                for name in algo_order:
                    # Extract numeric value
                    val = fmt_fn(results[name], name).strip()
                    row.append(val)
                writer.writerow(row)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
