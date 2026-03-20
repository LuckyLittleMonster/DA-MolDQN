#!/usr/bin/env python3
"""
Analyze diversity metrics from the diversity_qed experiments.

Reads all_path.txt files from each trial, computes:
1. FBC at various group sizes
2. Shannon entropy of FP bit distribution
3. Bemis-Murcko scaffold diversity + scaffold entropy
4. Internal diversity (mean pairwise Tanimoto distance)
5. BRICS fragment entropy

Outputs:
- Console table (always)
- CSV (--output_csv)
- LaTeX table (--output_latex)
- Publication-quality figures (--output_fig, default docs/diversity_figures/)

Usage:
    python scripts/analyze_diversity_qed.py
    python scripts/analyze_diversity_qed.py --configs 1 8 256
    python scripts/analyze_diversity_qed.py --include_reference --output_latex docs/diversity_table.tex
    python scripts/analyze_diversity_qed.py --output_fig docs/diversity_figures/ --no_plot
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


EXPERIMENT = "diversity_qed"
TRIAL_BASE = 4000
ALL_CONFIGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load_all_path_smiles(experiment_dir, experiment, trial):
    """Load and deduplicate SMILES from all_path.txt files across all ranks.

    Returns (smiles_list, total_count) or (None, 0) if no data found.
    """
    pattern = os.path.join(experiment_dir, f"{experiment}_{trial}_*_all_path.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, 0

    all_smiles = set()
    total_lines = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                smi = line.strip()
                if smi:
                    total_lines += 1
                    all_smiles.add(smi)

    print(f"    Loaded {len(files)} files, {total_lines} total SMILES, {len(all_smiles)} unique")
    return list(all_smiles), total_lines


def load_reference_smiles(data_dir, name):
    """Load AODB, ChEMBL, or ZINC reference molecules.

    Returns (smiles_list, total_count).
    """
    if name == "AODB":
        import csv
        path = os.path.join(data_dir, "AODB_s.csv")
        smiles = set()
        total = 0
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                total += 1
                mol = Chem.MolFromSmiles(row[0].strip())
                if mol is not None:
                    smiles.add(Chem.MolToSmiles(mol))
        return list(smiles), total
    elif name == "ChEMBL":
        path = os.path.join(data_dir, "chembl_31_10k.txt")
        smiles = set()
        total = 0
        with open(path) as f:
            for line in f:
                smi = line.strip()
                if smi:
                    total += 1
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        smiles.add(Chem.MolToSmiles(mol))
        return list(smiles), total
    elif name == "ZINC":
        path = os.path.join(data_dir, "zinc_10000.txt")
        smiles = set()
        total = 0
        with open(path) as f:
            for line in f:
                smi = line.strip()
                if smi:
                    total += 1
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        smiles.add(Chem.MolToSmiles(mol))
        return list(smiles), total
    return [], 0


def compute_fps(smiles_list, radius=3, n_bits=2048):
    """Compute Morgan fingerprints."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
    return fps


def compute_fbc(fps, group_size, n_trials=100, n_bits=2048):
    """FBC at given group size."""
    if len(fps) < group_size:
        combined = DataStructs.ExplicitBitVect(n_bits)
        for fp in fps:
            combined |= fp
        return combined.GetNumOnBits() / n_bits

    rng = np.random.RandomState(42)
    coverages = []
    for _ in range(n_trials):
        idx = rng.choice(len(fps), size=group_size, replace=False)
        combined = DataStructs.ExplicitBitVect(n_bits)
        for i in idx:
            combined |= fps[i]
        coverages.append(combined.GetNumOnBits() / n_bits)
    return np.mean(coverages)


def compute_shannon_entropy(fps, n_bits=2048):
    """Normalized Shannon entropy of FP bit distribution."""
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
    """Bemis-Murcko scaffold diversity ratio."""
    scaffolds = set()
    n_valid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            n_valid += 1
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(s))
            except:
                pass
    return len(scaffolds), n_valid


def compute_internal_diversity(fps, max_pairs=50000):
    """Internal diversity = 1 - mean pairwise Tanimoto similarity."""
    n = len(fps)
    if n < 2:
        return 0.0
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    else:
        rng = np.random.RandomState(42)
        sims = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - np.mean(sims)


def compute_scaffold_entropy(smiles_list):
    """Shannon entropy over Murcko scaffold frequency distribution.

    Returns (entropy_bits, normalized_entropy, n_unique_scaffolds).
    Normalized entropy = H / log2(N) where N = number of unique scaffolds.
    """
    scaffold_counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_counts[Chem.MolToSmiles(s)] += 1
            except:
                pass
    if not scaffold_counts:
        return 0.0, 0.0, 0
    total = sum(scaffold_counts.values())
    probs = np.array(list(scaffold_counts.values())) / total
    h = -np.sum(probs * np.log2(probs))
    n_unique = len(scaffold_counts)
    h_max = np.log2(n_unique) if n_unique > 1 else 1.0
    return h, h / h_max, n_unique


def compute_fragment_entropy(smiles_list, max_mols=5000):
    """BRICS fragment entropy."""
    from rdkit.Chem import BRICS
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
            except:
                pass
    if not counts:
        return 0.0, 0
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs)), len(counts)


def generate_figures(configs, results, ref_results, output_dir):
    """Generate publication-quality diversity figures.

    Args:
        configs: list of int (#mols configs, e.g. [1, 2, 4, ...])
        results: dict {config_name: {metric: value}}
        ref_results: dict {ref_name: {metric: value}} for AODB/ChEMBL
        output_dir: directory to save figures
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    config_names = [f"{c}_mols" for c in configs]
    config_names = [n for n in config_names if n in results]
    x_vals = [int(n.split("_")[0]) for n in config_names]

    if not x_vals:
        print("WARNING: No config data for plotting, skipping figures.")
        return

    # --- Figure A: 2x2 diversity metrics vs #mols ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Diversity Metrics vs. Number of Parallel Molecules", fontsize=13)

    metric_specs = [
        ("Scaffold_ratio", "Scaffold Diversity (ratio)", axes[0, 0]),
        ("Scaffold_H_norm", "Scaffold Entropy (normalized)", axes[0, 1]),
        ("IntDiv", "Internal Diversity (1 - mean Tanimoto)", axes[1, 0]),
        ("Frag_H", "BRICS Fragment Entropy (bits)", axes[1, 1]),
    ]

    for metric_key, ylabel, ax in metric_specs:
        y_vals = [results[n].get(metric_key, 0) for n in config_names]
        ax.plot(x_vals, y_vals, "o-", color="#2563eb", linewidth=2, markersize=6, label="DA-MolDQN")
        ax.set_xlabel("# Parallel Molecules")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], fontsize=8)

        # Reference lines
        for ref_name, ref_data in ref_results.items():
            if metric_key in ref_data:
                ax.axhline(ref_data[metric_key], linestyle="--", linewidth=1.5,
                           label=ref_name, alpha=0.7)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"diversity_vs_nmols.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure A: diversity_vs_nmols.pdf/png")

    # --- Figure B: Absolute counts vs #mols ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Chemical Space Coverage vs. Number of Parallel Molecules", fontsize=13)

    count_specs = [
        ("N_scaffolds", "Unique Scaffolds", axes[0]),
        ("N_fragments", "Unique BRICS Fragments", axes[1]),
    ]

    for metric_key, ylabel, ax in count_specs:
        y_vals = [results[n].get(metric_key, 0) for n in config_names]
        ax.plot(x_vals, y_vals, "s-", color="#dc2626", linewidth=2, markersize=6, label="DA-MolDQN")
        ax.set_xlabel("# Parallel Molecules")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], fontsize=8)

        for ref_name, ref_data in ref_results.items():
            if metric_key in ref_data:
                ax.axhline(ref_data[metric_key], linestyle="--", linewidth=1.5,
                           label=ref_name, alpha=0.7)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"counts_vs_nmols.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure B: counts_vs_nmols.pdf/png")


def generate_latex_table(names, datasets, results, total_counts, output_path):
    """Generate LaTeX table for paper inclusion.

    Args:
        names: ordered list of dataset names
        datasets: dict {name: smiles_list}
        results: dict {name: {metric: value}}
        total_counts: dict {name: total_count_before_dedup}
        output_path: file path for .tex output
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Determine if we have reference datasets (non-numeric names)
    config_names = [n for n in names if n.endswith("_mols")]
    ref_names = [n for n in names if not n.endswith("_mols")]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Quantitative Diversity Metrics of Explored Chemical Space}")
    lines.append(r"  \label{tab:diversity_metrics}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{l r r r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & Total Mols & Unique Mols & Scaffolds & Scaffold Div & IntDiv"
                 r" & Shannon $H$ & Scaffold $H_{\mathrm{norm}}$ & BRICS $H$ \\")
    lines.append(r"    \midrule")

    for name in config_names:
        r = results[name]
        label = name.replace("_", r"\_")
        total = total_counts.get(name, len(datasets[name]))
        lines.append(
            f"    {label} & {total:,d} & {len(datasets[name]):,d} & {r['N_scaffolds']:,d}"
            f" & {r['Scaffold_ratio']:.3f} & {r['IntDiv']:.3f}"
            f" & {r['Shannon_H']:.3f} & {r['Scaffold_H_norm']:.3f}"
            f" & {r['Frag_H']:.1f} \\\\"
        )

    if ref_names:
        lines.append(r"    \midrule")
        for name in ref_names:
            r = results[name]
            total = total_counts.get(name, len(datasets[name]))
            lines.append(
                f"    {name} & {total:,d} & {len(datasets[name]):,d} & {r['N_scaffolds']:,d}"
                f" & {r['Scaffold_ratio']:.3f} & {r['IntDiv']:.3f}"
                f" & {r['Shannon_H']:.3f} & {r['Scaffold_H_norm']:.3f}"
                f" & {r['Frag_H']:.1f} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", default="./Experiments")
    parser.add_argument("--data_dir", default="./Data")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--trial_base", type=int, default=TRIAL_BASE)
    parser.add_argument("--configs", nargs="+", type=int, default=None)
    parser.add_argument("--include_reference", action="store_true")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_latex", default=None, help="Output LaTeX table file")
    parser.add_argument("--output_fig", default="docs/diversity_figures/",
                        help="Output directory for figures")
    parser.add_argument("--no_plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    configs = args.configs or ALL_CONFIGS

    # Load data
    datasets = {}
    total_counts = {}  # total SMILES before dedup
    for nmol in configs:
        trial = args.trial_base + nmol
        print(f"Loading {nmol} mols (trial {trial})...")
        smiles, total = load_all_path_smiles(args.experiment_dir, args.experiment, trial)
        if smiles is not None:
            name = f"{nmol}_mols"
            datasets[name] = smiles
            total_counts[name] = total
        else:
            print(f"    WARNING: No data found for trial {trial}")

    if args.include_reference:
        for ref_name in ["AODB", "ChEMBL", "ZINC"]:
            print(f"Loading {ref_name}...")
            smiles, total = load_reference_smiles(args.data_dir, ref_name)
            if smiles:
                datasets[ref_name] = smiles
                total_counts[ref_name] = total
                print(f"    {total} total, {len(smiles)} unique molecules")

    if not datasets:
        print("No data found. Exiting.")
        return

    # Compute fingerprints
    print("\nComputing fingerprints...")
    fps_dict = {}
    for name, smiles in datasets.items():
        fps_dict[name] = compute_fps(smiles)
        print(f"  {name}: {len(fps_dict[name])} valid FPs")

    # Compute metrics
    names = list(datasets.keys())
    results = {}

    print("\n" + "=" * 90)
    print("FBC @ group sizes")
    print("=" * 90)
    group_sizes = [128, 256, 512, 1024, 2048, 4096]
    for gs in group_sizes:
        row = []
        for name in names:
            fbc = compute_fbc(fps_dict[name], gs)
            row.append(fbc)
            results.setdefault(name, {})[f"FBC_{gs}"] = fbc
        print(f"  Group {gs:4d}: " + " ".join(f"{name}={v:.4f}" for name, v in zip(names, row)))

    print("\n" + "=" * 90)
    print("DIVERSITY METRICS")
    print("=" * 90)

    header = f"{'Metric':25s} " + " ".join(f"{n:>14s}" for n in names)
    print(header)
    print("-" * len(header))

    # Total molecules (before dedup)
    row = f"{'Total Molecules':25s} "
    for name in names:
        row += f"{total_counts.get(name, len(datasets[name])):>14d}"
    print(row)

    # Unique molecules
    row = f"{'Unique Molecules':25s} "
    for name in names:
        row += f"{len(datasets[name]):>14d}"
    print(row)

    # FBC@512
    row = f"{'FBC@512':25s} "
    for name in names:
        row += f"{results[name]['FBC_512']:>14.4f}"
    print(row)

    # Unexplored@512
    row = f"{'Unexplored@512 (%)':25s} "
    for name in names:
        val = (1 - results[name]['FBC_512']) * 100
        results[name]['Unexplored_512'] = val
        row += f"{val:>14.1f}"
    print(row)

    # Shannon entropy
    row = f"{'Shannon H (norm)':25s} "
    for name in names:
        h = compute_shannon_entropy(fps_dict[name])
        results[name]['Shannon_H'] = h
        row += f"{h:>14.4f}"
    print(row)

    # Scaffold diversity + scaffold entropy
    row_scaff = f"{'Unique Scaffolds':25s} "
    row_ratio = f"{'Scaffold Diversity':25s} "
    row_sh = f"{'Scaffold H (bits)':25s} "
    row_shn = f"{'Scaffold H (norm)':25s} "
    for name in names:
        n_scaff, n_mol = compute_scaffold_diversity(datasets[name])
        ratio = n_scaff / max(n_mol, 1)
        s_h, s_h_norm, _ = compute_scaffold_entropy(datasets[name])
        results[name]['N_scaffolds'] = n_scaff
        results[name]['Scaffold_ratio'] = ratio
        results[name]['Scaffold_H'] = s_h
        results[name]['Scaffold_H_norm'] = s_h_norm
        row_scaff += f"{n_scaff:>14d}"
        row_ratio += f"{ratio:>14.4f}"
        row_sh += f"{s_h:>14.2f}"
        row_shn += f"{s_h_norm:>14.4f}"
    print(row_scaff)
    print(row_ratio)
    print(row_sh)
    print(row_shn)

    # Internal diversity
    row = f"{'Internal Diversity':25s} "
    for name in names:
        d = compute_internal_diversity(fps_dict[name])
        results[name]['IntDiv'] = d
        row += f"{d:>14.4f}"
    print(row)

    # BRICS fragment entropy
    row_nf = f"{'BRICS Fragments':25s} "
    row_fh = f"{'BRICS Entropy (bits)':25s} "
    for name in names:
        h, nf = compute_fragment_entropy(datasets[name])
        results[name]['Frag_H'] = h
        results[name]['N_fragments'] = nf
        row_nf += f"{nf:>14d}"
        row_fh += f"{h:>14.2f}"
    print(row_nf)
    print(row_fh)

    # Save CSV
    if args.output_csv:
        import csv
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Metric"] + names)
            metrics = [
                ("Total Molecules", lambda r, n: total_counts.get(n, len(datasets[n]))),
                ("Unique Molecules", lambda r, n: len(datasets[n])),
                ("FBC@512", lambda r, n: f"{r['FBC_512']:.4f}"),
                ("Unexplored@512 (%)", lambda r, n: f"{r['Unexplored_512']:.1f}"),
                ("Shannon H (norm)", lambda r, n: f"{r['Shannon_H']:.4f}"),
                ("Unique Scaffolds", lambda r, n: r['N_scaffolds']),
                ("Scaffold Diversity", lambda r, n: f"{r['Scaffold_ratio']:.4f}"),
                ("Scaffold H (bits)", lambda r, n: f"{r['Scaffold_H']:.2f}"),
                ("Scaffold H (norm)", lambda r, n: f"{r['Scaffold_H_norm']:.4f}"),
                ("Internal Diversity", lambda r, n: f"{r['IntDiv']:.4f}"),
                ("BRICS Fragments", lambda r, n: r['N_fragments']),
                ("BRICS Entropy (bits)", lambda r, n: f"{r['Frag_H']:.2f}"),
            ]
            for metric_name, fn in metrics:
                w.writerow([metric_name] + [fn(results[n], n) for n in names])
        print(f"\nSaved CSV to {args.output_csv}")

    # Save LaTeX table
    if args.output_latex:
        generate_latex_table(names, datasets, results, total_counts, args.output_latex)

    # Generate figures
    if not args.no_plot:
        print("\nGenerating figures...")
        config_names = [n for n in names if n.endswith("_mols")]
        ref_names = [n for n in names if not n.endswith("_mols")]
        ref_results = {n: results[n] for n in ref_names}
        generate_figures(
            [int(n.split("_")[0]) for n in config_names],
            results, ref_results, args.output_fig
        )


if __name__ == "__main__":
    main()
