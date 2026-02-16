#!/usr/bin/env python
"""
RL Path Quality Analysis - Analyze chemical reasonability of RL synthesis paths.

Computes per-step metrics:
  - Tanimoto similarity (input -> output per step)
  - MW change, heavy atom count change
  - T5v2 "jump" detection (Tanimoto < threshold)
  - Path length vs QED improvement correlation

Usage:
    python scripts/analyze_rl_paths.py \
        --paths Experiments/qed_hypergraph_0_paths.pickle \
               Experiments/qed_mlp_0_paths.pickle \
               Experiments/qed_planbc_0_paths.pickle \
        --labels hypergraph mlp planbc \
        -o Experiments/path_quality_analysis.txt
"""

import argparse
import pickle
import sys
import os
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def parse_args():
    p = argparse.ArgumentParser(description='Analyze RL path quality')
    p.add_argument('--paths', type=str, nargs='+', required=True,
                   help='Path pickle files to analyze')
    p.add_argument('--labels', type=str, nargs='+', default=None,
                   help='Labels for each path file (default: filenames)')
    p.add_argument('--evals', type=str, nargs='+', default=None,
                   help='Eval pickle files (optional, preferred over training paths)')
    p.add_argument('-o', '--output', type=str, default=None,
                   help='Output file (default: stdout)')
    p.add_argument('--jump_threshold', type=float, default=0.3,
                   help='Tanimoto threshold below which a step is a "jump" (default: 0.3)')
    p.add_argument('--episode', type=int, default=-1,
                   help='Training episode to use (-1 = last)')
    return p.parse_args()


def mol_properties(smiles):
    """Compute molecular properties from SMILES. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'mw': Descriptors.ExactMolWt(mol),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_rings': Descriptors.RingCount(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),
        'logp': Descriptors.MolLogP(mol),
    }


def tanimoto_sim(smi1, smi2, radius=2, nbits=2048):
    """Compute Tanimoto similarity between two SMILES."""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def load_training_paths(path, episode_idx=-1):
    """Load training paths, return list of molecule trajectories."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    lp = data['last_paths'][episode_idx]
    n_mols = len(lp['paths'][0])
    n_steps = len(lp['paths'])

    trajectories = []
    for i in range(n_mols):
        smiles = [lp['paths'][s][i] for s in range(n_steps)]
        qeds = [lp['qeds'][s][i] for s in range(len(lp['qeds']))]
        sas = []
        if len(lp['sas']) == n_steps:
            sas = [lp['sas'][s][i] for s in range(n_steps)]
        elif len(lp['sas']) == n_steps - 1:
            sas = [None] + [lp['sas'][s][i] for s in range(len(lp['sas']))]

        trajectories.append({
            'smiles': smiles,
            'qeds': qeds,
            'sas': sas,
        })

    return trajectories, lp['episode']


def load_eval_paths(path):
    """Load eval paths. Handles both old and new formats."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # New format: mol_results list
    if 'mol_results' in data:
        trajectories = []
        for r in data['mol_results']:
            trajectories.append({
                'smiles': r['path_smiles'],
                'qeds': r['path_qeds'],
                'sas': r.get('path_sas', []),
                'actions': r.get('path_actions', []),
            })
        return trajectories

    # Old format: path[step][mol_idx] structure
    if 'path' in data:
        n_steps = len(data['path'])
        n_mols = len(data['path'][0])
        trajectories = []
        for i in range(n_mols):
            smiles = [data['path'][s][i] for s in range(n_steps)]
            qeds = [data['qeds'][s][i] for s in range(n_steps)]
            sas = []
            if 'sas' in data:
                sas = [data['sas'][s][i] for s in range(len(data['sas']))]
            trajectories.append({
                'smiles': smiles,
                'qeds': qeds,
                'sas': sas,
            })
        return trajectories

    return []


def analyze_trajectory(traj, jump_threshold=0.3):
    """Analyze a single molecule trajectory.

    Returns dict with per-step and overall metrics.
    """
    smiles = traj['smiles']
    qeds = traj['qeds']
    n_steps = len(smiles)

    steps = []
    for s in range(1, n_steps):
        prev_smi = smiles[s - 1]
        curr_smi = smiles[s]
        changed = prev_smi != curr_smi

        step_info = {
            'step': s,
            'changed': changed,
            'prev_smiles': prev_smi,
            'curr_smiles': curr_smi,
            'qed_prev': qeds[s - 1] if s - 1 < len(qeds) else None,
            'qed_curr': qeds[s] if s < len(qeds) else None,
        }

        if changed:
            sim = tanimoto_sim(prev_smi, curr_smi)
            step_info['tanimoto'] = sim
            step_info['is_jump'] = sim is not None and sim < jump_threshold

            prev_props = mol_properties(prev_smi)
            curr_props = mol_properties(curr_smi)

            if prev_props and curr_props:
                step_info['mw_change'] = curr_props['mw'] - prev_props['mw']
                step_info['ha_change'] = curr_props['heavy_atoms'] - prev_props['heavy_atoms']
                step_info['ring_change'] = curr_props['num_rings'] - prev_props['num_rings']
                step_info['prev_mw'] = prev_props['mw']
                step_info['curr_mw'] = curr_props['mw']
            else:
                step_info['mw_change'] = None
                step_info['ha_change'] = None
                step_info['ring_change'] = None
        else:
            step_info['tanimoto'] = 1.0
            step_info['is_jump'] = False
            step_info['mw_change'] = 0.0
            step_info['ha_change'] = 0

        steps.append(step_info)

    # Overall metrics
    init_qed = qeds[0] if qeds else None
    final_qed = qeds[-1] if qeds else None
    qed_improvement = final_qed - init_qed if init_qed is not None and final_qed is not None else None

    effective_steps = sum(1 for s in steps if s['changed'])
    n_jumps = sum(1 for s in steps if s.get('is_jump', False))

    tanimotos = [s['tanimoto'] for s in steps if s['changed'] and s['tanimoto'] is not None]
    min_tanimoto = min(tanimotos) if tanimotos else None
    mean_tanimoto = np.mean(tanimotos) if tanimotos else None

    mw_changes = [s['mw_change'] for s in steps if s['changed'] and s['mw_change'] is not None]
    total_mw_change = sum(mw_changes) if mw_changes else None
    max_abs_mw_change = max(abs(x) for x in mw_changes) if mw_changes else None

    ha_changes = [s['ha_change'] for s in steps if s['changed'] and s['ha_change'] is not None]
    total_ha_change = sum(ha_changes) if ha_changes else None

    # Init/final molecular properties
    init_props = mol_properties(smiles[0])
    final_props = mol_properties(smiles[-1])

    return {
        'steps': steps,
        'init_qed': init_qed,
        'final_qed': final_qed,
        'qed_improvement': qed_improvement,
        'effective_steps': effective_steps,
        'total_steps': n_steps - 1,
        'n_jumps': n_jumps,
        'min_tanimoto': min_tanimoto,
        'mean_tanimoto': mean_tanimoto,
        'total_mw_change': total_mw_change,
        'max_abs_mw_change': max_abs_mw_change,
        'total_ha_change': total_ha_change,
        'init_props': init_props,
        'final_props': final_props,
        'init_smiles': smiles[0],
        'final_smiles': smiles[-1],
    }


def format_summary(label, analyses, jump_threshold):
    """Format summary statistics for a set of trajectory analyses."""
    lines = []
    n = len(analyses)

    lines.append(f"{'=' * 90}")
    lines.append(f"  {label}  ({n} molecules)")
    lines.append(f"{'=' * 90}")

    # QED statistics
    init_qeds = [a['init_qed'] for a in analyses if a['init_qed'] is not None]
    final_qeds = [a['final_qed'] for a in analyses if a['final_qed'] is not None]
    improvements = [a['qed_improvement'] for a in analyses if a['qed_improvement'] is not None]

    if init_qeds:
        lines.append(f"\n  QED Statistics:")
        lines.append(f"    Init QED:     mean={np.mean(init_qeds):.4f}  median={np.median(init_qeds):.4f}  "
                      f"std={np.std(init_qeds):.4f}")
        lines.append(f"    Final QED:    mean={np.mean(final_qeds):.4f}  median={np.median(final_qeds):.4f}  "
                      f"std={np.std(final_qeds):.4f}")
        lines.append(f"    Improvement:  mean={np.mean(improvements):+.4f}  median={np.median(improvements):+.4f}")
        lines.append(f"    Improved:     {sum(1 for x in improvements if x > 0.01)}/{n}  "
                      f"Degraded: {sum(1 for x in improvements if x < -0.01)}/{n}  "
                      f"Unchanged: {sum(1 for x in improvements if abs(x) <= 0.01)}/{n}")

    # Step statistics
    effective_steps = [a['effective_steps'] for a in analyses]
    total_steps = [a['total_steps'] for a in analyses]
    lines.append(f"\n  Step Statistics:")
    lines.append(f"    Total steps/mol:     {np.mean(total_steps):.1f}")
    lines.append(f"    Effective steps/mol:  mean={np.mean(effective_steps):.2f}  "
                  f"median={np.median(effective_steps):.1f}")
    lines.append(f"    Mols with 0 changes:  {sum(1 for e in effective_steps if e == 0)}/{n}")

    # Tanimoto statistics
    all_tanimotos = []
    for a in analyses:
        for s in a['steps']:
            if s['changed'] and s['tanimoto'] is not None:
                all_tanimotos.append(s['tanimoto'])

    min_tanimotos = [a['min_tanimoto'] for a in analyses if a['min_tanimoto'] is not None]
    mean_tanimotos = [a['mean_tanimoto'] for a in analyses if a['mean_tanimoto'] is not None]

    lines.append(f"\n  Structural Similarity (Tanimoto, per changed step):")
    if all_tanimotos:
        lines.append(f"    All steps:  mean={np.mean(all_tanimotos):.4f}  "
                      f"median={np.median(all_tanimotos):.4f}  "
                      f"min={np.min(all_tanimotos):.4f}  max={np.max(all_tanimotos):.4f}")
        lines.append(f"    Per-mol min:  mean={np.mean(min_tanimotos):.4f}  "
                      f"median={np.median(min_tanimotos):.4f}")
        lines.append(f"    Per-mol mean: mean={np.mean(mean_tanimotos):.4f}  "
                      f"median={np.median(mean_tanimotos):.4f}")

    # Jump detection
    n_jumps = [a['n_jumps'] for a in analyses]
    total_jumps = sum(n_jumps)
    mols_with_jumps = sum(1 for j in n_jumps if j > 0)
    lines.append(f"\n  T5v2 Jump Detection (Tanimoto < {jump_threshold}):")
    lines.append(f"    Total jumps:       {total_jumps}")
    lines.append(f"    Mols with jumps:   {mols_with_jumps}/{n} ({100*mols_with_jumps/n:.1f}%)")
    if all_tanimotos:
        lines.append(f"    Jump rate:         {total_jumps}/{len(all_tanimotos)} steps "
                      f"({100*total_jumps/len(all_tanimotos):.1f}%)")

    # Tanimoto distribution
    if all_tanimotos:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(all_tanimotos, bins=bins)
        lines.append(f"\n    Tanimoto distribution:")
        for i in range(len(hist)):
            bar = '#' * int(hist[i] * 40 / max(max(hist), 1))
            lines.append(f"      [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d}  {bar}")

    # MW change statistics
    mw_changes = []
    for a in analyses:
        for s in a['steps']:
            if s['changed'] and s.get('mw_change') is not None:
                mw_changes.append(s['mw_change'])

    if mw_changes:
        lines.append(f"\n  MW Change (per changed step):")
        lines.append(f"    mean={np.mean(mw_changes):+.1f}  median={np.median(mw_changes):+.1f}  "
                      f"std={np.std(mw_changes):.1f}")
        lines.append(f"    min={np.min(mw_changes):+.1f}  max={np.max(mw_changes):+.1f}")
        lines.append(f"    |MW change| > 200: {sum(1 for x in mw_changes if abs(x) > 200)}/{len(mw_changes)}")
        lines.append(f"    |MW change| > 300: {sum(1 for x in mw_changes if abs(x) > 300)}/{len(mw_changes)}")

    total_mw = [a['total_mw_change'] for a in analyses if a['total_mw_change'] is not None]
    if total_mw:
        lines.append(f"    Total MW change/mol: mean={np.mean(total_mw):+.1f}  "
                      f"median={np.median(total_mw):+.1f}")

    # Heavy atom change
    ha_changes = []
    for a in analyses:
        for s in a['steps']:
            if s['changed'] and s.get('ha_change') is not None:
                ha_changes.append(s['ha_change'])
    if ha_changes:
        lines.append(f"\n  Heavy Atom Change (per changed step):")
        lines.append(f"    mean={np.mean(ha_changes):+.1f}  median={np.median(ha_changes):+.1f}")
        lines.append(f"    |HA change| > 10: {sum(1 for x in ha_changes if abs(x) > 10)}/{len(ha_changes)}")
        lines.append(f"    |HA change| > 20: {sum(1 for x in ha_changes if abs(x) > 20)}/{len(ha_changes)}")

    # Correlation: effective steps vs QED improvement
    if improvements and effective_steps:
        valid = [(e, imp) for e, imp in zip(effective_steps, improvements) if e > 0]
        if len(valid) >= 3:
            es, imps = zip(*valid)
            corr = np.corrcoef(es, imps)[0, 1] if np.std(es) > 0 and np.std(imps) > 0 else 0
            lines.append(f"\n  Correlation (effective_steps vs QED_improvement): r={corr:.4f}")

    # Worst jumps (for debugging)
    jump_examples = []
    for a in analyses:
        for s in a['steps']:
            if s.get('is_jump', False) and s['tanimoto'] is not None:
                jump_examples.append({
                    'tanimoto': s['tanimoto'],
                    'prev': s['prev_smiles'][:60],
                    'curr': s['curr_smiles'][:60],
                    'mw_change': s.get('mw_change'),
                    'step': s['step'],
                })
    jump_examples.sort(key=lambda x: x['tanimoto'])

    if jump_examples:
        lines.append(f"\n  Worst Jumps (lowest Tanimoto, top 10):")
        for j in jump_examples[:10]:
            mw_str = f"dMW={j['mw_change']:+.0f}" if j['mw_change'] is not None else ""
            lines.append(f"    Tan={j['tanimoto']:.3f}  S{j['step']}  {mw_str}")
            lines.append(f"      prev: {j['prev']}")
            lines.append(f"      curr: {j['curr']}")

    lines.append("")
    return lines


def format_detailed(label, analyses, top_n=5):
    """Show detailed per-step analysis for worst and best molecules."""
    lines = []
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  DETAILED: {label}")
    lines.append(f"{'=' * 90}")

    # Sort by improvement
    sorted_by_imp = sorted(enumerate(analyses), key=lambda x: -(x[1]['qed_improvement'] or -999))

    # Best molecules
    lines.append(f"\n  Top {top_n} by QED improvement:")
    for rank, (idx, a) in enumerate(sorted_by_imp[:top_n]):
        lines.append(f"\n    Mol #{idx}  QED: {a['init_qed']:.3f} -> {a['final_qed']:.3f} "
                      f"(+{a['qed_improvement']:.3f})  eff_steps={a['effective_steps']}")
        lines.append(f"      init: {a['init_smiles'][:70]}")
        lines.append(f"      final: {a['final_smiles'][:70]}")
        for s in a['steps']:
            if s['changed']:
                t = s['tanimoto']
                t_str = f"{t:.3f}" if t is not None else "N/A"
                mw_str = f"dMW={s['mw_change']:+.0f}" if s.get('mw_change') is not None else ""
                jump_str = " JUMP!" if s.get('is_jump') else ""
                lines.append(f"      S{s['step']}: Tan={t_str}  {mw_str}{jump_str}")

    # Worst molecules (most jumps or most degraded)
    sorted_by_jumps = sorted(enumerate(analyses), key=lambda x: -x[1]['n_jumps'])
    lines.append(f"\n  Top {top_n} by number of jumps:")
    for rank, (idx, a) in enumerate(sorted_by_jumps[:top_n]):
        if a['n_jumps'] == 0:
            break
        lines.append(f"\n    Mol #{idx}  QED: {a['init_qed']:.3f} -> {a['final_qed']:.3f}  "
                      f"jumps={a['n_jumps']}  eff_steps={a['effective_steps']}")
        lines.append(f"      init: {a['init_smiles'][:70]}")
        lines.append(f"      final: {a['final_smiles'][:70]}")
        for s in a['steps']:
            if s['changed']:
                t = s['tanimoto']
                t_str = f"{t:.3f}" if t is not None else "N/A"
                mw_str = f"dMW={s['mw_change']:+.0f}" if s.get('mw_change') is not None else ""
                jump_str = " JUMP!" if s.get('is_jump') else ""
                lines.append(f"      S{s['step']}: Tan={t_str}  {mw_str}{jump_str}")

    lines.append("")
    return lines


def main():
    args = parse_args()

    if args.labels:
        labels = args.labels
    else:
        labels = [os.path.basename(p).replace('_paths.pickle', '').replace('_eval.pickle', '')
                  for p in args.paths]

    out = []
    out.append("=" * 90)
    out.append("  RL PATH QUALITY ANALYSIS")
    out.append(f"  Jump threshold: Tanimoto < {args.jump_threshold}")
    out.append("=" * 90)

    all_analyses = {}

    for path_file, label in zip(args.paths, labels):
        print(f"Analyzing {label}...", file=sys.stderr)

        # Try to load eval if provided
        eval_file = None
        if args.evals:
            for ef in args.evals:
                if label in ef:
                    eval_file = ef
                    break

        if eval_file:
            trajectories = load_eval_paths(eval_file)
            out.append(f"\n  Source: {eval_file} (eval)")
        else:
            trajectories, episode = load_training_paths(path_file, args.episode)
            out.append(f"\n  Source: {path_file} (training, episode {episode})")

        # Analyze each trajectory
        analyses = []
        for traj in trajectories:
            a = analyze_trajectory(traj, jump_threshold=args.jump_threshold)
            analyses.append(a)

        all_analyses[label] = analyses

        # Summary
        out.extend(format_summary(label, analyses, args.jump_threshold))

        # Detailed
        out.extend(format_detailed(label, analyses, top_n=5))

    # Cross-method comparison
    if len(all_analyses) > 1:
        out.append(f"\n{'=' * 90}")
        out.append(f"  CROSS-METHOD COMPARISON")
        out.append(f"{'=' * 90}")

        header = f"  {'Metric':<30}"
        for label in all_analyses:
            header += f"  {label:>15}"
        out.append(header)
        out.append("  " + "-" * (30 + 17 * len(all_analyses)))

        metrics = [
            ('Mean QED improve', lambda a: np.mean([x['qed_improvement'] for x in a if x['qed_improvement'] is not None])),
            ('Median QED improve', lambda a: np.median([x['qed_improvement'] for x in a if x['qed_improvement'] is not None])),
            ('Mean final QED', lambda a: np.mean([x['final_qed'] for x in a if x['final_qed'] is not None])),
            ('Mean eff. steps', lambda a: np.mean([x['effective_steps'] for x in a])),
            ('Jump rate (%)', lambda a: 100 * sum(x['n_jumps'] for x in a) / max(sum(1 for x in a for s in x['steps'] if s['changed']), 1)),
            ('Mols with jumps (%)', lambda a: 100 * sum(1 for x in a if x['n_jumps'] > 0) / max(len(a), 1)),
            ('Mean step Tanimoto', lambda a: np.mean([s['tanimoto'] for x in a for s in x['steps'] if s['changed'] and s['tanimoto'] is not None]) if any(s['changed'] and s['tanimoto'] is not None for x in a for s in x['steps']) else float('nan')),
            ('Mean |MW change|/step', lambda a: np.mean([abs(s['mw_change']) for x in a for s in x['steps'] if s['changed'] and s.get('mw_change') is not None]) if any(s['changed'] and s.get('mw_change') is not None for x in a for s in x['steps']) else float('nan')),
        ]

        for name, fn in metrics:
            row = f"  {name:<30}"
            for label in all_analyses:
                val = fn(all_analyses[label])
                if isinstance(val, float):
                    row += f"  {val:>15.4f}"
                else:
                    row += f"  {val!s:>15}"
            out.append(row)

    out.append("")
    text = "\n".join(out)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(text + "\n")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == '__main__':
    main()
