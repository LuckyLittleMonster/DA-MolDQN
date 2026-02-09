#!/usr/bin/env python
"""
Analyze results from the RL experiment matrix.

Reads Experiments/*_history.pickle and *_paths.pickle files to compare
across experiments (E1-E6).

Features:
  - Training performance: reward curves, QED progression
  - Path quality: Tanimoto step similarity, MW changes, jump rate
  - QED distribution: final molecule QED percentiles
  - Pairwise comparisons: key experiment contrasts with delta reporting
  - Markdown output mode for reports

Usage:
    python scripts/analyze_experiment_matrix.py
    python scripts/analyze_experiment_matrix.py --markdown
    python scripts/analyze_experiment_matrix.py --experiments E1_baseline_fixed E2_aio_basic
"""

import argparse
import glob
import os
import pickle
import numpy as np
from collections import defaultdict

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator

fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)

# Experiment descriptions for reporting
EXP_DESCRIPTIONS = {
    'E1_baseline_fixed': 'Hypergraph 2-step + filter',
    'E2_aio_basic': 'AIO + filter',
    'E3_aio_reward_shaped': 'AIO + filter + reward shaping',
    'E4_baseline_reward_shaped': 'Hypergraph 2-step + filter + reward',
    'E5_aio_no_filter': 'AIO (no filter)',
    'E6_aio_topk5': 'AIO + filter (top_k=5)',
}

# Key pairwise comparisons
COMPARISONS = [
    ('E1_baseline_fixed', 'E2_aio_basic', 'Hypergraph vs AIO (same filter, same k)'),
    ('E2_aio_basic', 'E3_aio_reward_shaped', 'Effect of reward shaping (AIO)'),
    ('E1_baseline_fixed', 'E4_baseline_reward_shaped', 'Effect of reward shaping (2-step)'),
    ('E2_aio_basic', 'E5_aio_no_filter', 'Value of product filter (AIO)'),
    ('E2_aio_basic', 'E6_aio_topk5', 'top_k=20 vs top_k=5 (AIO)'),
    ('E3_aio_reward_shaped', 'E4_baseline_reward_shaped', 'Best AIO vs Best 2-step'),
]


def tanimoto(smi_a, smi_b):
    """Compute Tanimoto similarity between two SMILES."""
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(
        fp_gen.GetFingerprint(mol_a), fp_gen.GetFingerprint(mol_b))


def analyze_history(history_file):
    """Analyze training history pickle."""
    with open(history_file, 'rb') as f:
        data = pickle.load(f)

    rewards = data.get('rewards', data.get('all_rewards', []))
    qeds = data.get('qeds', data.get('all_qeds', []))

    if not rewards:
        return None

    n = len(rewards)
    window = min(100, n // 4) if n > 40 else min(10, n)

    stats = {
        'episodes': n,
        'final_reward': float(np.mean(rewards[-window:])),
        'max_reward': float(np.max(rewards)),
        'final_qed': float(np.mean(qeds[-window:])) if qeds and len(qeds) >= window else None,
        'max_qed': float(np.max(qeds)) if qeds else None,
    }

    # Learning curve: rewards at 25%, 50%, 75%
    for pct in [25, 50, 75]:
        idx = int(n * pct / 100)
        w = min(10, idx)
        stats[f'reward_at_{pct}pct'] = float(np.mean(rewards[max(0, idx-w):idx+w]))

    # QED distribution (final 20% of episodes)
    if qeds:
        tail = qeds[int(n * 0.8):]
        if tail:
            stats['qed_p25'] = float(np.percentile(tail, 25))
            stats['qed_p50'] = float(np.percentile(tail, 50))
            stats['qed_p75'] = float(np.percentile(tail, 75))
            stats['qed_mean'] = float(np.mean(tail))

    return stats


def analyze_paths(paths_file):
    """Analyze synthesis paths pickle."""
    with open(paths_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        paths = data.get('top_paths', data.get('paths', []))
    elif isinstance(data, list):
        paths = data
    else:
        return None

    if not paths:
        return None

    all_tani = []
    all_mw_delta = []
    all_qed_delta = []
    all_path_lengths = []
    all_final_qed = []
    n_jumps = 0
    n_steps_total = 0

    for path_data in paths:
        if isinstance(path_data, dict):
            path = path_data.get('path', [])
        elif isinstance(path_data, (list, tuple)):
            path = path_data
        else:
            continue

        if len(path) < 2:
            continue

        all_path_lengths.append(len(path) - 1)

        # Final molecule QED
        final_smi = path[-1] if isinstance(path[-1], str) else Chem.MolToSmiles(path[-1])
        final_mol = Chem.MolFromSmiles(final_smi) if isinstance(final_smi, str) else final_smi
        if final_mol is not None:
            all_final_qed.append(QED.qed(final_mol))

        for i in range(len(path) - 1):
            smi_a = path[i] if isinstance(path[i], str) else Chem.MolToSmiles(path[i])
            smi_b = path[i+1] if isinstance(path[i+1], str) else Chem.MolToSmiles(path[i+1])

            t = tanimoto(smi_a, smi_b)
            all_tani.append(t)
            n_steps_total += 1
            if t < 0.3:
                n_jumps += 1

            mol_a = Chem.MolFromSmiles(smi_a)
            mol_b = Chem.MolFromSmiles(smi_b)
            if mol_a and mol_b:
                mw_a = Descriptors.ExactMolWt(mol_a)
                mw_b = Descriptors.ExactMolWt(mol_b)
                all_mw_delta.append(abs(mw_b - mw_a))
                all_qed_delta.append(QED.qed(mol_b) - QED.qed(mol_a))

    if not all_tani:
        return None

    stats = {
        'n_paths': len(all_path_lengths),
        'avg_path_length': float(np.mean(all_path_lengths)),
        'avg_tanimoto': float(np.mean(all_tani)),
        'median_tanimoto': float(np.median(all_tani)),
        'jump_rate': n_jumps / n_steps_total if n_steps_total > 0 else 0,
        'avg_mw_delta': float(np.mean(all_mw_delta)) if all_mw_delta else 0,
        'avg_qed_delta': float(np.mean(all_qed_delta)) if all_qed_delta else 0,
    }

    if all_final_qed:
        stats['final_qed_mean'] = float(np.mean(all_final_qed))
        stats['final_qed_max'] = float(np.max(all_final_qed))
        stats['final_qed_p75'] = float(np.percentile(all_final_qed, 75))

    return stats


def find_experiment_files(exp_dir, exp_prefix):
    """Find history and paths files for an experiment prefix."""
    history_files = sorted(glob.glob(os.path.join(exp_dir, f'{exp_prefix}_*_history.pickle')))
    paths_files = sorted(glob.glob(os.path.join(exp_dir, f'{exp_prefix}_*_paths.pickle')))
    return history_files, paths_files


def fmt(val, fmt_str='.3f'):
    """Format a value, handling None."""
    if val is None:
        return '-'
    return f'{val:{fmt_str}}'


def print_plain_tables(results, exp_names):
    """Print results as plain text tables."""
    print("\n" + "=" * 90)
    print("TRAINING PERFORMANCE")
    print("=" * 90)
    header = f"{'Experiment':<28} {'Episodes':>8} {'Final Rew':>9} {'Max Rew':>7} {'Final QED':>9} {'Max QED':>7} {'QED p50':>7}"
    print(header)
    print("-" * 90)

    for exp_name in exp_names:
        if exp_name not in results:
            continue
        h = results[exp_name].get('history', {})
        print(f"{exp_name:<28} {h.get('episodes', '-'):>8} "
              f"{fmt(h.get('final_reward'), '.4f'):>9} "
              f"{fmt(h.get('max_reward'), '.4f'):>7} "
              f"{fmt(h.get('final_qed'), '.3f'):>9} "
              f"{fmt(h.get('max_qed'), '.3f'):>7} "
              f"{fmt(h.get('qed_p50'), '.3f'):>7}")

    print("\n" + "=" * 90)
    print("PATH QUALITY")
    print("=" * 90)
    header = f"{'Experiment':<28} {'Paths':>5} {'AvgLen':>6} {'AvgTani':>7} {'MedTani':>7} {'Jump%':>5} {'MWdelta':>7} {'dQED':>7}"
    print(header)
    print("-" * 90)

    for exp_name in exp_names:
        if exp_name not in results:
            continue
        p = results[exp_name].get('paths', {})
        if not p:
            continue
        print(f"{exp_name:<28} {p.get('n_paths', 0):>5} "
              f"{p.get('avg_path_length', 0):>6.1f} "
              f"{p.get('avg_tanimoto', 0):>7.3f} "
              f"{p.get('median_tanimoto', 0):>7.3f} "
              f"{100*p.get('jump_rate', 0):>4.0f}% "
              f"{p.get('avg_mw_delta', 0):>7.1f} "
              f"{p.get('avg_qed_delta', 0):>+7.3f}")


def print_markdown_tables(results, exp_names):
    """Print results as markdown tables."""
    print("\n## Training Performance\n")
    print("| Experiment | Description | Episodes | Final Reward | Max Reward | Final QED | Max QED | QED p50 |")
    print("|:-----------|:------------|-------:|------:|------:|------:|------:|------:|")

    for exp_name in exp_names:
        if exp_name not in results:
            continue
        h = results[exp_name].get('history', {})
        desc = EXP_DESCRIPTIONS.get(exp_name, '')
        print(f"| {exp_name} | {desc} | "
              f"{h.get('episodes', '-')} | "
              f"{fmt(h.get('final_reward'), '.4f')} | "
              f"{fmt(h.get('max_reward'), '.4f')} | "
              f"{fmt(h.get('final_qed'), '.3f')} | "
              f"{fmt(h.get('max_qed'), '.3f')} | "
              f"{fmt(h.get('qed_p50'), '.3f')} |")

    print("\n## Path Quality\n")
    print("| Experiment | Paths | Avg Len | Avg Tani | Med Tani | Jump% | MW Delta | dQED |")
    print("|:-----------|------:|------:|------:|------:|------:|------:|------:|")

    for exp_name in exp_names:
        if exp_name not in results:
            continue
        p = results[exp_name].get('paths', {})
        if not p:
            continue
        print(f"| {exp_name} | "
              f"{p.get('n_paths', 0)} | "
              f"{p.get('avg_path_length', 0):.1f} | "
              f"{p.get('avg_tanimoto', 0):.3f} | "
              f"{p.get('median_tanimoto', 0):.3f} | "
              f"{100*p.get('jump_rate', 0):.0f}% | "
              f"{p.get('avg_mw_delta', 0):.1f} | "
              f"{p.get('avg_qed_delta', 0):+.3f} |")


def print_comparisons(results, markdown=False):
    """Print pairwise comparison analysis."""
    if markdown:
        print("\n## Pairwise Comparisons\n")
    else:
        print("\n" + "=" * 90)
        print("PAIRWISE COMPARISONS")
        print("=" * 90)

    for exp_a, exp_b, description in COMPARISONS:
        if exp_a not in results or exp_b not in results:
            continue

        ha = results[exp_a].get('history', {})
        hb = results[exp_b].get('history', {})
        pa = results[exp_a].get('paths', {})
        pb = results[exp_b].get('paths', {})

        if markdown:
            print(f"\n### {description}")
            print(f"**{exp_a}** vs **{exp_b}**\n")
        else:
            print(f"\n  {description}: {exp_a} vs {exp_b}")

        metrics = []

        # Reward comparison
        ra = ha.get('final_reward')
        rb = hb.get('final_reward')
        if ra is not None and rb is not None:
            delta = rb - ra
            pct = (delta / abs(ra) * 100) if ra != 0 else 0
            winner = exp_b if delta > 0 else exp_a
            metrics.append(('Final Reward', f'{ra:.4f}', f'{rb:.4f}', f'{delta:+.4f} ({pct:+.1f}%)', winner))

        # QED comparison
        qa = ha.get('final_qed')
        qb = hb.get('final_qed')
        if qa is not None and qb is not None:
            delta = qb - qa
            winner = exp_b if delta > 0 else exp_a
            metrics.append(('Final QED', f'{qa:.3f}', f'{qb:.3f}', f'{delta:+.3f}', winner))

        # Path Tanimoto
        ta = pa.get('avg_tanimoto')
        tb = pb.get('avg_tanimoto')
        if ta is not None and tb is not None:
            delta = tb - ta
            winner = exp_b if delta > 0 else exp_a
            metrics.append(('Avg Tanimoto', f'{ta:.3f}', f'{tb:.3f}', f'{delta:+.3f}', winner))

        # Jump rate (lower is better)
        ja = pa.get('jump_rate')
        jb = pb.get('jump_rate')
        if ja is not None and jb is not None:
            delta = jb - ja
            winner = exp_b if delta < 0 else exp_a  # lower is better
            metrics.append(('Jump Rate', f'{100*ja:.1f}%', f'{100*jb:.1f}%', f'{100*delta:+.1f}pp', winner))

        if not metrics:
            if markdown:
                print("No comparable data available.\n")
            else:
                print("    No comparable data available.")
            continue

        if markdown:
            print(f"| Metric | {exp_a} | {exp_b} | Delta | Winner |")
            print("|:-------|------:|------:|------:|:-------|")
            for name, va, vb, delta, winner in metrics:
                short_winner = winner.split('_')[0] + '_' + '_'.join(winner.split('_')[1:3])
                print(f"| {name} | {va} | {vb} | {delta} | {short_winner} |")
        else:
            for name, va, vb, delta, winner in metrics:
                print(f"    {name:<15} {va:>10} -> {vb:>10}  ({delta})  winner: {winner}")


def main():
    parser = argparse.ArgumentParser(description='Analyze RL experiment matrix results')
    parser.add_argument('--exp_dir', type=str, default='Experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--experiments', nargs='+',
                        default=['E1_baseline_fixed', 'E2_aio_basic',
                                 'E3_aio_reward_shaped',
                                 'E4_baseline_reward_shaped',
                                 'E5_aio_no_filter', 'E6_aio_topk5'],
                        help='Experiment names to analyze')
    parser.add_argument('--markdown', action='store_true',
                        help='Output in markdown format')
    args = parser.parse_args()

    if args.markdown:
        print("# RL Experiment Matrix Results")
    else:
        print("=" * 90)
        print("RL Experiment Matrix - Results Analysis")
        print("=" * 90)

    results = {}

    for exp_name in args.experiments:
        history_files, paths_files = find_experiment_files(args.exp_dir, exp_name)

        if not history_files and not paths_files:
            if not args.markdown:
                print(f"\n{exp_name}: No files found")
            continue

        exp_results = {'name': exp_name}

        if history_files:
            history_stats = analyze_history(history_files[0])
            if history_stats:
                exp_results['history'] = history_stats

        if paths_files:
            path_stats = analyze_paths(paths_files[0])
            if path_stats:
                exp_results['paths'] = path_stats

        results[exp_name] = exp_results

    if not results:
        print("\nNo experiment results found. Run experiments first:")
        print("  sbatch scripts/rl_experiment_matrix.sh")
        return

    if args.markdown:
        print_markdown_tables(results, args.experiments)
    else:
        print_plain_tables(results, args.experiments)

    print_comparisons(results, markdown=args.markdown)

    # Summary
    if results:
        if args.markdown:
            print("\n## Summary\n")
        else:
            print("\n" + "=" * 90)
            print("SUMMARY")
            print("=" * 90)

        # Find best experiment by final reward
        best_reward_exp = max(
            (k for k in results if 'history' in results[k]),
            key=lambda k: results[k]['history'].get('final_reward', -999),
            default=None
        )
        best_qed_exp = max(
            (k for k in results if 'history' in results[k]),
            key=lambda k: results[k]['history'].get('max_qed', -999),
            default=None
        )
        best_tani_exp = max(
            (k for k in results if 'paths' in results[k]),
            key=lambda k: results[k]['paths'].get('avg_tanimoto', -999),
            default=None
        )

        lines = []
        if best_reward_exp:
            r = results[best_reward_exp]['history']['final_reward']
            lines.append(f"Best final reward: **{best_reward_exp}** ({r:.4f})" if args.markdown
                        else f"  Best final reward: {best_reward_exp} ({r:.4f})")
        if best_qed_exp:
            q = results[best_qed_exp]['history']['max_qed']
            lines.append(f"Best max QED: **{best_qed_exp}** ({q:.3f})" if args.markdown
                        else f"  Best max QED: {best_qed_exp} ({q:.3f})")
        if best_tani_exp:
            t = results[best_tani_exp]['paths']['avg_tanimoto']
            lines.append(f"Best path quality (Tanimoto): **{best_tani_exp}** ({t:.3f})" if args.markdown
                        else f"  Best path quality (Tanimoto): {best_tani_exp} ({t:.3f})")

        for line in lines:
            print(f"- {line}" if args.markdown else line)

    if not args.markdown:
        print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
