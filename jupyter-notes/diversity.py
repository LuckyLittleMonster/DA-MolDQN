"""Diversity metrics computation with caching for DA-MolDQN experiments."""

import csv
import glob as globmod
import os
import pickle
import numpy as np
from collections import Counter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


# Reusable Morgan FP generator (new API, replaces deprecated GetMorganFingerprintAsBitVect)
_morgan_gen = None

def get_morgan_generator(radius=3, n_bits=2048):
    global _morgan_gen
    if _morgan_gen is None:
        _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _morgan_gen


def compute_fps(smiles_list, radius=3, n_bits=2048):
    gen = get_morgan_generator(radius, n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
    return fps


def compute_shannon_entropy(fps, n_bits=2048):
    n = len(fps)
    if n == 0:
        return 0.0
    counts = np.zeros(n_bits)
    for fp in fps:
        for b in fp.GetOnBits():
            counts[b] += 1
    p = np.clip(counts / n, 1e-10, 1 - 1e-10)
    bit_h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.sum(bit_h) / n_bits


def compute_scaffold_diversity(smiles_list):
    scaffolds = set()
    n_valid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            n_valid += 1
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(s))
            except Exception:
                pass
    return len(scaffolds), n_valid


def compute_scaffold_entropy(smiles_list):
    scaffold_counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_counts[Chem.MolToSmiles(s)] += 1
            except Exception:
                pass
    if not scaffold_counts:
        return 0.0, 0.0, 0
    total = sum(scaffold_counts.values())
    probs = np.array(list(scaffold_counts.values())) / total
    h = -np.sum(probs * np.log2(probs))
    n_unique = len(scaffold_counts)
    h_max = np.log2(n_unique) if n_unique > 1 else 1.0
    return h, h / h_max, n_unique


def compute_internal_diversity(fps, max_pairs=50000):
    n = len(fps)
    if n < 2:
        return 0.0
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
                for i in range(n) for j in range(i + 1, n)]
    else:
        rng = np.random.RandomState(42)
        sims = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - np.mean(sims)


def compute_fragment_entropy(smiles_list, max_mols=5000):
    if len(smiles_list) > max_mols:
        rng = np.random.RandomState(42)
        smiles_list = list(rng.choice(smiles_list, size=max_mols, replace=False))
    counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                for frag in BRICS.BRICSDecompose(mol):
                    counts[frag] += 1
            except Exception:
                pass
    if not counts:
        return 0.0, 0
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs)), len(counts)


# --- Data loading ---

def load_all_path(exp_dir, experiment, trial):
    pattern = os.path.join(exp_dir, f"{experiment}_{trial}_*_all_path.txt")
    files = sorted(globmod.glob(pattern))
    all_smi, total = set(), 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                s = line.strip()
                if s:
                    total += 1
                    all_smi.add(s)
    return list(all_smi), total


def load_ref(data_dir, name):
    if name == "AODB":
        path = os.path.join(data_dir, "AODB_s.csv")
        smi_set, total = set(), 0
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                total += 1
                mol = Chem.MolFromSmiles(row[0].strip())
                if mol:
                    smi_set.add(Chem.MolToSmiles(mol))
        return list(smi_set), total
    elif name == "ChEMBL":
        path = os.path.join(data_dir, "chembl_31_10k.txt")
    elif name == "ZINC":
        path = os.path.join(data_dir, "zinc_10000.txt")
    else:
        return [], 0
    smi_set, total = set(), 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                total += 1
                mol = Chem.MolFromSmiles(s)
                if mol:
                    smi_set.add(Chem.MolToSmiles(mol))
    return list(smi_set), total


# --- Cached diversity pipeline ---

def compute_all_metrics(datasets, total_counts, cache_path=None):
    """Compute all diversity metrics for each dataset, with optional disk cache.

    Returns (fps_dict, results) where results[name] is a dict of metrics.
    Loads from cache_path if it exists; saves after computation if cache_path is set.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"Loaded cached diversity results from {cache_path}")
        return cached['fps_dict'], cached['results']

    print("Computing fingerprints...")
    fps_dict = {}
    for name, smiles in datasets.items():
        fps_dict[name] = compute_fps(smiles)
        print(f"  {name}: {len(fps_dict[name])} valid FPs")

    names = list(datasets.keys())
    results = {}
    for name in names:
        r = {}
        r['total'] = total_counts.get(name, len(datasets[name]))
        r['unique'] = len(datasets[name])
        r['Shannon_H'] = compute_shannon_entropy(fps_dict[name])
        n_scaff, n_mol = compute_scaffold_diversity(datasets[name])
        r['N_scaffolds'] = n_scaff
        r['Scaffold_ratio'] = n_scaff / max(n_mol, 1)
        s_h, s_h_norm, _ = compute_scaffold_entropy(datasets[name])
        r['Scaffold_H'] = s_h
        r['Scaffold_H_norm'] = s_h_norm
        r['IntDiv'] = compute_internal_diversity(fps_dict[name])
        frag_h, n_frag = compute_fragment_entropy(datasets[name])
        r['Frag_H'] = frag_h
        r['N_fragments'] = n_frag
        results[name] = r
        print(f"  {name}: done")

    # Print summary table
    header = (f"{'Dataset':12s} {'Total':>10s} {'Unique':>10s} {'Scaffolds':>10s} "
              f"{'ScaffDiv':>9s} {'IntDiv':>7s} {'Shannon':>8s} {'ScaffH_n':>9s} {'BRICS_H':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for name in names:
        r = results[name]
        print(f"{name:12s} {r['total']:>10,d} {r['unique']:>10,d} {r['N_scaffolds']:>10,d} "
              f"{r['Scaffold_ratio']:>9.3f} {r['IntDiv']:>7.3f} {r['Shannon_H']:>8.4f} "
              f"{r['Scaffold_H_norm']:>9.4f} {r['Frag_H']:>8.1f}")

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump({'fps_dict': fps_dict, 'results': results}, f)
        print(f"\nCached diversity results to {cache_path}")

    return fps_dict, results
