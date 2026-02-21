"""Deep analysis of PaRoutes target molecules as RL starting points.

Questions to answer:
1. QED distribution - room for optimization?
2. Drug-likeness (Lipinski, Veber, Ghose)
3. Docking suitability (size, flexibility, pharmacophores)
4. Comparison with our ZINC64 starting molecules
5. PAINS / structural alerts
6. Scaffold diversity
7. Substructure analysis (common pharmacophores)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, QED, rdMolDescriptors, FilterCatalog,
    Fragments, AllChem, rdFingerprintGenerator
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def compute_full_props(smi):
    """Compute comprehensive molecular properties."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return {
            'smiles': smi,
            'MW': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'QED': QED.qed(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'Rings': rdMolDescriptors.CalcNumRings(mol),
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(mol),
            'NumStereoCenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'mol': mol,
        }
    except Exception:
        return None


def check_pains(mol):
    """Check for PAINS alerts."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog.HasMatch(mol)


def check_lipinski(p):
    """Check Lipinski Rule of 5."""
    violations = 0
    if p['MW'] >= 500: violations += 1
    if p['LogP'] >= 5: violations += 1
    if p['HBD'] > 5: violations += 1
    if p['HBA'] > 10: violations += 1
    return violations


def check_veber(p):
    """Check Veber rules (oral bioavailability)."""
    return p['TPSA'] <= 140 and p['RotBonds'] <= 10


def check_ghose(p):
    """Check Ghose filter."""
    return (160 <= p['MW'] <= 480 and
            -0.4 <= p['LogP'] <= 5.6 and
            20 <= p['HeavyAtoms'] <= 70 and
            40 <= Descriptors.MolMR(p['mol']) <= 130)


def get_scaffold(mol):
    """Get Murcko scaffold SMILES."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def percentile(values, p):
    """Simple percentile calculation."""
    s = sorted(values)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def print_distribution(values, name, bins=None):
    """Print a text histogram."""
    if bins is None:
        bins = 10
    mn, mx = min(values), max(values)
    if isinstance(bins, int):
        step = (mx - mn) / bins if mx > mn else 1
        bins = [mn + i * step for i in range(bins + 1)]

    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1] or (i == len(bins) - 2 and v == bins[i + 1]):
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1
    print(f"\n  {name} distribution (n={len(values)}):")
    for i in range(len(bins) - 1):
        bar_len = int(counts[i] / max_count * 40)
        bar = '#' * bar_len
        pct = counts[i] / len(values) * 100
        print(f"    [{bins[i]:6.1f}, {bins[i+1]:6.1f}): {counts[i]:>5} ({pct:5.1f}%) {bar}")


def analyze_for_rl(routes_file, name, zinc64_file=None):
    """Analyze PaRoutes targets as RL starting molecules."""
    print(f"\n{'='*75}")
    print(f"  PaRoutes Quality Analysis: {name}")
    print(f"  Question: Are these good starting molecules for QED/Docking optimization?")
    print(f"{'='*75}")

    with open(routes_file) as f:
        routes = json.load(f)

    # Extract all target SMILES
    targets = [r['smiles'] for r in routes if r.get('smiles')]
    print(f"\n  Total targets: {len(targets)}")

    # Compute properties for all
    print("  Computing properties...")
    all_props = []
    for smi in targets:
        p = compute_full_props(smi)
        if p:
            all_props.append(p)
    print(f"  Valid molecules: {len(all_props)}")

    # =========================================================
    # 1. QED Analysis
    # =========================================================
    print(f"\n{'─'*75}")
    print("  1. QED Analysis — Is there room for optimization?")
    print(f"{'─'*75}")

    qeds = [p['QED'] for p in all_props]
    print(f"  Mean QED: {np.mean(qeds):.3f}")
    print(f"  Median QED: {np.median(qeds):.3f}")
    print(f"  Std QED: {np.std(qeds):.3f}")
    print(f"  Min QED: {min(qeds):.3f}, Max QED: {max(qeds):.3f}")

    qed_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print_distribution(qeds, "QED", qed_bins)

    low_qed = sum(1 for q in qeds if q < 0.3)
    mid_qed = sum(1 for q in qeds if 0.3 <= q < 0.6)
    high_qed = sum(1 for q in qeds if q >= 0.6)
    very_high = sum(1 for q in qeds if q >= 0.8)
    print(f"\n  QED < 0.3 (poor):     {low_qed:>5} ({low_qed/len(qeds)*100:.1f}%)")
    print(f"  QED 0.3-0.6 (medium): {mid_qed:>5} ({mid_qed/len(qeds)*100:.1f}%)")
    print(f"  QED >= 0.6 (good):    {high_qed:>5} ({high_qed/len(qeds)*100:.1f}%)")
    print(f"  QED >= 0.8 (excellent):{very_high:>4} ({very_high/len(qeds)*100:.1f}%)")

    print(f"\n  → QED优化空间: ", end="")
    if np.mean(qeds) < 0.5:
        print("大量空间 — 多数分子 QED 较低，RL 可以显著提升")
    elif np.mean(qeds) < 0.7:
        print("中等空间 — 分子已有一定 drug-likeness，RL 可进一步优化")
    else:
        print("有限空间 — 分子已经 drug-like，优化难度大")

    # =========================================================
    # 2. Drug-likeness Filters
    # =========================================================
    print(f"\n{'─'*75}")
    print("  2. Drug-likeness Filter Compliance")
    print(f"{'─'*75}")

    lipinski_pass = sum(1 for p in all_props if check_lipinski(p) <= 1)
    veber_pass = sum(1 for p in all_props if check_veber(p))
    ghose_pass = sum(1 for p in all_props if check_ghose(p))

    # PAINS (sample 2000 for speed)
    pains_sample = all_props[:2000]
    pains_hits = sum(1 for p in pains_sample if check_pains(p['mol']))

    n = len(all_props)
    print(f"  Lipinski Ro5 (≤1 violation): {lipinski_pass}/{n} ({lipinski_pass/n*100:.1f}%)")
    print(f"  Veber (TPSA≤140, RotB≤10):   {veber_pass}/{n} ({veber_pass/n*100:.1f}%)")
    print(f"  Ghose filter:                 {ghose_pass}/{n} ({ghose_pass/n*100:.1f}%)")
    print(f"  PAINS clean (sample 2K):      {len(pains_sample)-pains_hits}/{len(pains_sample)} "
          f"({(len(pains_sample)-pains_hits)/len(pains_sample)*100:.1f}%)")

    # Lipinski violation breakdown
    viol_dist = Counter(check_lipinski(p) for p in all_props)
    print(f"\n  Lipinski violation distribution:")
    for v in sorted(viol_dist.keys()):
        print(f"    {v} violations: {viol_dist[v]:>5} ({viol_dist[v]/n*100:.1f}%)")

    # =========================================================
    # 3. Docking Suitability
    # =========================================================
    print(f"\n{'─'*75}")
    print("  3. Docking Suitability Analysis")
    print(f"{'─'*75}")

    mws = [p['MW'] for p in all_props]
    logps = [p['LogP'] for p in all_props]
    tpsas = [p['TPSA'] for p in all_props]
    rotbs = [p['RotBonds'] for p in all_props]
    heavy = [p['HeavyAtoms'] for p in all_props]
    rings = [p['Rings'] for p in all_props]
    arom = [p['AromaticRings'] for p in all_props]
    fsp3 = [p['FractionCSP3'] for p in all_props]

    print(f"  {'Property':>20} {'Mean':>8} {'Median':>8} {'P10':>8} {'P90':>8} {'Ideal Range':>16}")
    print(f"  {'─'*72}")
    for prop_name, vals, ideal in [
        ('MW', mws, '250-500'),
        ('LogP', logps, '1-5'),
        ('TPSA', tpsas, '40-140'),
        ('RotBonds', rotbs, '2-8'),
        ('HeavyAtoms', heavy, '15-35'),
        ('Rings', rings, '2-5'),
        ('AromaticRings', arom, '1-3'),
        ('FractionCSP3', fsp3, '0.2-0.5'),
    ]:
        print(f"  {prop_name:>20} {np.mean(vals):8.1f} {np.median(vals):8.1f} "
              f"{percentile(vals, 10):8.1f} {percentile(vals, 90):8.1f} {ideal:>16}")

    # Docking-friendly size (250-550 Da, typical for kinase inhibitors)
    dock_size_ok = sum(1 for mw in mws if 250 <= mw <= 550)
    print(f"\n  Docking-friendly size (250-550 Da): {dock_size_ok}/{n} ({dock_size_ok/n*100:.1f}%)")

    # Flexibility check (too flexible = entropy penalty in docking)
    flex_ok = sum(1 for rb in rotbs if rb <= 10)
    print(f"  Flexibility OK (RotB≤10): {flex_ok}/{n} ({flex_ok/n*100:.1f}%)")

    # Has aromatic system (most kinase inhibitors)
    has_arom = sum(1 for a in arom if a >= 1)
    print(f"  Has aromatic ring(s): {has_arom}/{n} ({has_arom/n*100:.1f}%)")

    # "Sweet spot" for docking: MW 300-500, LogP 1-5, rings 2-4, RotB ≤ 8
    dock_sweet = sum(1 for p in all_props if
                     300 <= p['MW'] <= 500 and
                     1 <= p['LogP'] <= 5 and
                     2 <= p['Rings'] <= 5 and
                     p['RotBonds'] <= 8)
    print(f"\n  Docking sweet spot (MW 300-500, LogP 1-5, Rings 2-5, RotB≤8):")
    print(f"    {dock_sweet}/{n} ({dock_sweet/n*100:.1f}%)")

    # =========================================================
    # 4. Scaffold Diversity
    # =========================================================
    print(f"\n{'─'*75}")
    print("  4. Scaffold Diversity")
    print(f"{'─'*75}")

    scaffolds = []
    for p in all_props:
        sc = get_scaffold(p['mol'])
        if sc:
            scaffolds.append(sc)

    unique_scaffolds = set(scaffolds)
    scaffold_counts = Counter(scaffolds)
    top_scaffolds = scaffold_counts.most_common(10)

    print(f"  Total molecules: {len(scaffolds)}")
    print(f"  Unique scaffolds: {len(unique_scaffolds)} ({len(unique_scaffolds)/len(scaffolds)*100:.1f}%)")
    print(f"  Singleton scaffolds: {sum(1 for c in scaffold_counts.values() if c == 1)}")

    print(f"\n  Top 10 most common scaffolds:")
    for i, (sc, count) in enumerate(top_scaffolds):
        pct = count / len(scaffolds) * 100
        print(f"    {i+1}. {sc[:60]:60s} count={count} ({pct:.1f}%)")

    # =========================================================
    # 5. Compare with ZINC64 (our current starting molecules)
    # =========================================================
    print(f"\n{'─'*75}")
    print("  5. Comparison with Current ZINC64 Starting Molecules")
    print(f"{'─'*75}")

    zinc64_path = Path('/shared/data1/Users/l1062811/git/DA-MolDQN/rl/template/data/zinc_first64.smi')
    if zinc64_path.exists():
        zinc_props = []
        with open(zinc64_path) as f:
            for line in f:
                smi = line.strip().split()[0]
                if smi:
                    p = compute_full_props(smi)
                    if p:
                        zinc_props.append(p)

        if zinc_props:
            print(f"\n  {'Property':>15} {'ZINC64':>12} {'PaRoutes':>12} {'Verdict':>20}")
            print(f"  {'─'*60}")

            comparisons = [
                ('MW', [p['MW'] for p in zinc_props], mws),
                ('LogP', [p['LogP'] for p in zinc_props], logps),
                ('QED', [p['QED'] for p in zinc_props], qeds),
                ('TPSA', [p['TPSA'] for p in zinc_props], tpsas),
                ('RotBonds', [p['RotBonds'] for p in zinc_props], rotbs),
                ('HeavyAtoms', [p['HeavyAtoms'] for p in zinc_props], heavy),
                ('Rings', [p['Rings'] for p in zinc_props], rings),
            ]

            for prop_name, zinc_vals, pa_vals in comparisons:
                z_mean = np.mean(zinc_vals)
                p_mean = np.mean(pa_vals)
                if prop_name == 'QED':
                    verdict = "PaRoutes更好" if p_mean > z_mean else "ZINC64更好"
                elif prop_name in ('MW', 'RotBonds'):
                    verdict = "PaRoutes更大/灵活" if p_mean > z_mean else "ZINC64更大/灵活"
                else:
                    verdict = "相似" if abs(p_mean - z_mean) / max(abs(z_mean), 0.1) < 0.15 else "差异显著"
                print(f"  {prop_name:>15} {z_mean:>12.1f} {p_mean:>12.1f} {verdict:>20}")
    else:
        print("  ZINC64 文件未找到，跳过对比")

    # =========================================================
    # 6. Substructure Analysis (common pharmacophores)
    # =========================================================
    print(f"\n{'─'*75}")
    print("  6. Common Pharmacophore Substructures")
    print(f"{'─'*75}")

    pharmacophores = {
        'Amide (-CONHR)': '[C](=O)[NH]',
        'Sulfonamide (-SO2NR)': '[S](=O)(=O)[N]',
        'Amine (-NR2)': '[N;!$(N=*);!$(N#*)]',
        'Carboxylic acid': '[C](=O)[OH]',
        'Ester (-COOR)': '[C](=O)[O;!H]',
        'Ether (-O-)': '[OD2]([#6])[#6]',
        'Halogen (F/Cl/Br)': '[F,Cl,Br]',
        'Fluorine': '[F]',
        'Nitrogen heterocycle': '[nR]',
        'Oxygen heterocycle': '[oR]',
        'Piperidine/piperazine': '[NR1]1[CR1][CR1][CR1,NR1][CR1][CR1]1',
        'Benzimidazole': 'c1ccc2[nH]cnc2c1',
        'Pyridine': 'c1ccncc1',
        'Pyrimidine': 'c1ccnc(n1)',
        'Indole': 'c1ccc2[nH]ccc2c1',
    }

    sample = all_props[:5000]
    print(f"  (Analyzing {len(sample)} molecules)")
    for name_ph, smarts in pharmacophores.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        hits = sum(1 for p in sample if p['mol'].HasSubstructMatch(pattern))
        pct = hits / len(sample) * 100
        bar = '#' * int(pct / 3)
        print(f"    {name_ph:>28}: {hits:>5} ({pct:5.1f}%) {bar}")

    # =========================================================
    # 7. Overall Assessment
    # =========================================================
    print(f"\n{'─'*75}")
    print("  7. Overall Assessment: Suitability as RL Starting Molecules")
    print(f"{'─'*75}")

    # Score different aspects
    qed_score = "Good" if 0.4 <= np.mean(qeds) <= 0.7 else ("Low" if np.mean(qeds) < 0.4 else "Already high")
    size_score = "Good" if 250 <= np.mean(mws) <= 500 else "Suboptimal"
    diversity_score = "High" if len(unique_scaffolds) / len(scaffolds) > 0.3 else "Low"
    lipinski_score = "Good" if lipinski_pass / n > 0.7 else "Concerning"
    pains_score = "Good" if pains_hits / len(pains_sample) < 0.1 else "Concerning"

    print(f"""
  QED for optimization:     {qed_score}
    → Mean QED = {np.mean(qeds):.3f}, RL 目标通常是 >0.9
    → {sum(1 for q in qeds if q < 0.7)}/{n} ({sum(1 for q in qeds if q < 0.7)/n*100:.0f}%) 有优化空间 (QED<0.7)

  Molecular size:           {size_score}
    → Mean MW = {np.mean(mws):.0f} Da
    → Docking 最佳范围 300-500 Da: {dock_size_ok}/{n} ({dock_size_ok/n*100:.0f}%)
    → MW > 500 (Ro5 violation): {sum(1 for mw in mws if mw > 500)}/{n} ({sum(1 for mw in mws if mw > 500)/n*100:.0f}%)

  Drug-likeness:            {lipinski_score}
    → Lipinski 合规: {lipinski_pass/n*100:.0f}%
    → Veber 合规: {veber_pass/n*100:.0f}%
    → PAINS 干净: {(len(pains_sample)-pains_hits)/len(pains_sample)*100:.0f}%

  Scaffold diversity:       {diversity_score}
    → {len(unique_scaffolds)} unique scaffolds / {len(scaffolds)} molecules
    → 多样性有利于 RL 探索不同化学空间

  Docking suitability:
    → Sweet spot 命中: {dock_sweet/n*100:.0f}%
    → 芳香环系统: {has_arom/n*100:.0f}% (对蛋白-配体相互作用重要)
    → 柔性合适 (RotB≤10): {flex_ok/n*100:.0f}%
""")

    # Final verdict
    print(f"  {'='*60}")
    if np.mean(qeds) < 0.5:
        print("  QED优化: ★★★★☆ 大量优化空间，非常适合 QED RL 训练")
    elif np.mean(qeds) < 0.7:
        print("  QED优化: ★★★☆☆ 中等优化空间，适合 QED RL 训练")
    else:
        print("  QED优化: ★★☆☆☆ 有限优化空间，QED 已经较高")

    if dock_sweet / n > 0.3:
        print("  Docking: ★★★★☆ 大量分子在 docking sweet spot")
    elif dock_sweet / n > 0.15:
        print("  Docking: ★★★☆☆ 适中比例在 docking sweet spot")
    else:
        print("  Docking: ★★☆☆☆ 偏少分子适合 docking")

    if len(unique_scaffolds) / len(scaffolds) > 0.5:
        print("  多样性: ★★★★★ 极高骨架多样性")
    elif len(unique_scaffolds) / len(scaffolds) > 0.3:
        print("  多样性: ★★★★☆ 高骨架多样性")
    else:
        print("  多样性: ★★★☆☆ 中等骨架多样性")

    print(f"  {'='*60}")

    return all_props


if __name__ == '__main__':
    data_dir = Path('/shared/data1/Users/l1062811/git/DA-MolDQN/Data/paroutes')

    props_n1 = analyze_for_rl(
        data_dir / 'n1_routes.json',
        name='N1 Benchmark'
    )

    # Brief N5 comparison
    print(f"\n\n{'='*75}")
    print("  Quick N5 comparison (key differences only)")
    print(f"{'='*75}")
    with open(data_dir / 'n5_routes.json') as f:
        routes_n5 = json.load(f)
    targets_n5 = [r['smiles'] for r in routes_n5 if r.get('smiles')]
    n5_qeds = []
    n5_mws = []
    for smi in targets_n5[:3000]:
        p = compute_full_props(smi)
        if p:
            n5_qeds.append(p['QED'])
            n5_mws.append(p['MW'])
    if n5_qeds:
        n1_qeds = [p['QED'] for p in props_n1]
        print(f"  N1 mean QED: {np.mean(n1_qeds):.3f}, N5 mean QED: {np.mean(n5_qeds):.3f}")
        print(f"  N1 mean MW:  {np.mean([p['MW'] for p in props_n1]):.0f}, N5 mean MW:  {np.mean(n5_mws):.0f}")
        print(f"  N5 molecules are {'larger' if np.mean(n5_mws) > np.mean([p['MW'] for p in props_n1]) else 'smaller'} "
              f"and {'more' if np.mean(n5_qeds) > np.mean(n1_qeds) else 'less'} drug-like than N1")
