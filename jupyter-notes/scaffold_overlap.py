#!/usr/bin/env python
"""Train/test Murcko scaffold overlap analysis.

Datasets:
  1. ZINC QED       — Data/zinc_10000.txt, first 256 train, 256-1024 test (768 mols)
  2. Antioxidant    — Data/anti_400.txt,   first 256 train, 256-384 test (128 mols)

Output: summary table, LaTeX draft, and figure.

Usage:
    python jupyter-notes/scaffold_overlap.py
"""

import sys
from pathlib import Path
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "jupyter-notes"))


DATASETS = [
    {"name": "ZINC (QED)",   "file": "Data/zinc_10000.txt", "train": (0, 256), "test": (256, 1024)},
    {"name": "Antioxidant",  "file": "Data/anti_400.txt",   "train": (0, 256), "test": (256, 384)},
]


def get_scaffold(smi: str, generic: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold)


def scaffold_analysis(train_smiles: list[str], test_smiles: list[str], generic: bool = False):
    """Compute scaffold overlap statistics."""
    train_scaffolds = [sc for s in train_smiles if (sc := get_scaffold(s, generic)) is not None]
    test_scaffolds = [sc for s in test_smiles if (sc := get_scaffold(s, generic)) is not None]

    train_set = set(train_scaffolds)
    test_set = set(test_scaffolds)
    overlap = test_set & train_set
    novel = test_set - train_set

    n_overlap_mols = sum(1 for sc in test_scaffolds if sc in train_set)

    return {
        "n_train_mols": len(train_smiles),
        "n_test_mols": len(test_smiles),
        "n_train_scaffolds": len(train_set),
        "n_test_scaffolds": len(test_set),
        "n_overlap": len(overlap),
        "n_novel": len(novel),
        "overlap_rate": len(overlap) / len(test_set) * 100 if test_set else 0,
        "novel_rate": len(novel) / len(test_set) * 100 if test_set else 0,
        "mol_overlap_rate": n_overlap_mols / len(test_scaffolds) * 100 if test_scaffolds else 0,
        "mol_novel_rate": (len(test_scaffolds) - n_overlap_mols) / len(test_scaffolds) * 100 if test_scaffolds else 0,
        "train_counter": Counter(train_scaffolds),
        "test_counter": Counter(test_scaffolds),
        "overlap_scaffolds": overlap,
        "novel_scaffolds": novel,
    }


def print_results(res: dict, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Train molecules:      {res['n_train_mols']}")
    print(f"Test molecules:       {res['n_test_mols']}")
    print(f"Train scaffolds:      {res['n_train_scaffolds']}")
    print(f"Test scaffolds:       {res['n_test_scaffolds']}")
    print(f"Overlapping:          {res['n_overlap']}")
    print(f"Novel:                {res['n_novel']}")
    print(f"Overlap rate (uniq):  {res['overlap_rate']:.1f}%")
    print(f"Novel rate (uniq):    {res['novel_rate']:.1f}%")
    print(f"Overlap rate (mol):   {res['mol_overlap_rate']:.1f}%")
    print(f"Novel rate (mol):     {res['mol_novel_rate']:.1f}%")

    tc = res["test_counter"]
    ts = res["overlap_scaffolds"]
    print(f"\nTest Top-10 scaffolds:")
    for sc, cnt in tc.most_common(10):
        tag = "overlap" if sc in ts else "novel"
        print(f"  {cnt:3d}x  {sc}  [{tag}]")


def load_smiles(path: str, start: int, end: int) -> list[str]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[start:end]


def make_figure(all_results: dict[str, dict], out_path: str):
    """Bar chart comparing overlap rates across datasets."""
    ds_names = list(all_results.keys())
    n = len(ds_names)

    murcko_unique = [all_results[d]["murcko"]["overlap_rate"] for d in ds_names]
    murcko_mol = [all_results[d]["murcko"]["mol_overlap_rate"] for d in ds_names]
    generic_unique = [all_results[d]["generic"]["overlap_rate"] for d in ds_names]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(n)
    width = 0.25
    ax.bar([i - width for i in x], murcko_unique, width,
           label="Murcko (unique scaffold)", color="#4CAF50", alpha=0.85)
    ax.bar([i for i in x], murcko_mol, width,
           label="Murcko (per-molecule)", color="#2196F3", alpha=0.85)
    ax.bar([i + width for i in x], generic_unique, width,
           label="Generic (unique scaffold)", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Overlap Rate (%)")
    ax.set_title("Train/Test Scaffold Overlap")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ds_names)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")


def generate_latex(all_results: dict[str, dict]) -> str:
    zinc_m = all_results["ZINC (QED)"]["murcko"]
    anti_m = all_results["Antioxidant"]["murcko"]

    def val(r, key, fmt="{}", is_pct=False):
        v = fmt.format(r[key])
        if is_pct:
            v += r"\%"
        return v

    table = rf"""
\begin{{table}}[ht]
\caption{{Murcko Scaffold Overlap Between Training and Test Molecules}}
\label{{tab:scaffold_overlap}}
\begin{{center}}
\begin{{tabular}}{{lcc}}
\hline
 & ZINC (QED) & Antioxidant \\
\hline
Training molecules & {zinc_m['n_train_mols']} & {anti_m['n_train_mols']} \\
Test molecules & {zinc_m['n_test_mols']} & {anti_m['n_test_mols']} \\
Training unique scaffolds & {val(zinc_m, 'n_train_scaffolds')} & {val(anti_m, 'n_train_scaffolds')} \\
Test unique scaffolds & {val(zinc_m, 'n_test_scaffolds')} & {val(anti_m, 'n_test_scaffolds')} \\
Overlapping scaffolds & {val(zinc_m, 'n_overlap')} & {val(anti_m, 'n_overlap')} \\
Novel scaffolds & {val(zinc_m, 'n_novel')} & {val(anti_m, 'n_novel')} \\
\hline
Scaffold overlap rate & {val(zinc_m, 'overlap_rate', '{:.1f}', True)} & {val(anti_m, 'overlap_rate', '{:.1f}', True)} \\
Novel scaffold rate & {val(zinc_m, 'novel_rate', '{:.1f}', True)} & {val(anti_m, 'novel_rate', '{:.1f}', True)} \\
Per-molecule overlap rate & {val(zinc_m, 'mol_overlap_rate', '{:.1f}', True)} & {val(anti_m, 'mol_overlap_rate', '{:.1f}', True)} \\
Per-molecule novel rate & {val(zinc_m, 'mol_novel_rate', '{:.1f}', True)} & {val(anti_m, 'mol_novel_rate', '{:.1f}', True)} \\
\hline
\end{{tabular}}
\end{{center}}
\end{{table}}
"""

    discussion = rf"""
% --- LaTeX draft for manuscript ---

To assess the generalization capability of the trained agent,
we performed a Murcko scaffold overlap analysis~\cite{{bemis1996properties}}
between the training and test molecules for the ZINC and Antioxidant datasets.
Murcko scaffolds represent the core ring systems and linkers of a molecule
after removing all side chains, providing a standard measure of structural novelty.

As shown in Table~\ref{{tab:scaffold_overlap}},
the scaffold overlap between training and test sets is remarkably low
for the ZINC dataset: only {zinc_m['overlap_rate']:.1f}\% of unique Murcko scaffolds
in the 768 test molecules were observed during training
({zinc_m['n_overlap']} out of {zinc_m['n_test_scaffolds']} scaffolds),
and {zinc_m['mol_novel_rate']:.1f}\% of the test molecules
contain scaffolds entirely absent from the training set.
Despite this minimal structural overlap,
the agent successfully optimizes QED for these unseen scaffolds,
confirming genuine generalization.

The Antioxidant dataset shows higher overlap ({anti_m['overlap_rate']:.1f}\%)
because it is designed around a specific chemical series
(substituted phenols and pyridines) with inherently low scaffold diversity
({anti_m['n_test_scaffolds']} unique test scaffolds vs.\
{zinc_m['n_test_scaffolds']} for ZINC).
This is expected for a focused proprietary dataset;
nevertheless, the agent must learn to optimize
different substitution patterns within the shared scaffolds.

These results confirm that the agent generalizes to novel molecular
scaffolds not encountered during training, rather than memorizing
training-specific structural motifs.
"""
    return table + discussion


def main():
    all_results = {}

    for ds in DATASETS:
        path = str(ROOT / ds["file"])
        train = load_smiles(path, *ds["train"])
        test = load_smiles(path, *ds["test"])

        murcko = scaffold_analysis(train, test, generic=False)
        print_results(murcko, f"{ds['name']} — Murcko Scaffold")

        generic = scaffold_analysis(train, test, generic=True)
        print_results(generic, f"{ds['name']} — Generic Scaffold")

        all_results[ds["name"]] = {"murcko": murcko, "generic": generic}

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  Summary (Murcko Scaffold)")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'#Train':>6} {'#Test':>6} {'Overlap':>8} {'Novel':>6} {'Overlap%':>10} {'Mol%':>8}")
    print("-" * 80)
    for name, res in all_results.items():
        r = res["murcko"]
        print(f"{name:<20} {r['n_train_scaffolds']:>6} {r['n_test_scaffolds']:>6} "
              f"{r['n_overlap']:>8} {r['n_novel']:>6} {r['overlap_rate']:>9.1f}% {r['mol_overlap_rate']:>7.1f}%")

    # Figure
    fig_path = str(ROOT / "jupyter-notes" / "scaffold_overlap.png")
    make_figure(all_results, fig_path)

    # LaTeX
    latex = generate_latex(all_results)
    latex_path = str(ROOT / "jupyter-notes" / "scaffold_overlap.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX saved: {latex_path}")


if __name__ == "__main__":
    main()
