#!/usr/bin/env python
"""
Visualize synthesis paths from training and/or eval results as readable txt.

Usage:
    # Show both training (last episode) and eval paths
    python show_paths.py \
        --train Experiments/qed_planbc_0_paths.pickle \
        --eval Experiments/qed_planbc_0_eval.pickle \
        -o Experiments/qed_planbc_0_paths.txt

    # Show only eval paths
    python show_paths.py \
        --eval Experiments/qed_planbc_0_eval.pickle

    # Show specific molecules
    python show_paths.py \
        --eval Experiments/qed_planbc_0_eval.pickle \
        --mols 0 3 29 55

    # Sort by final eval QED (default), or by improvement
    python show_paths.py \
        --eval Experiments/qed_planbc_0_eval.pickle \
        --sort improvement
"""

import argparse
import pickle
import sys
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Visualize synthesis paths')
    p.add_argument('--train', type=str, default=None, help='Training paths pickle')
    p.add_argument('--eval', type=str, default=None, help='Eval results pickle')
    p.add_argument('-o', '--output', type=str, default=None,
                   help='Output txt path (default: stdout)')
    p.add_argument('--mols', type=int, nargs='+', default=None,
                   help='Specific molecule indices to show (default: all)')
    p.add_argument('--top', type=int, default=None,
                   help='Show only top N molecules (by sort key)')
    p.add_argument('--sort', type=str, default='final_qed',
                   choices=['final_qed', 'improvement', 'init_qed', 'index'],
                   help='Sort order for molecules')
    p.add_argument('--train_episode', type=int, default=-1,
                   help='Which episode from last_paths to use (-1 = last)')
    return p.parse_args()


def load_training_paths(path, episode_idx=-1):
    """Load training paths and return per-molecule data."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    lp = data['last_paths'][episode_idx]
    episode = lp['episode']
    n_mols = len(lp['paths'][0])
    n_steps = len(lp['paths'])  # includes init

    mols = []
    for i in range(n_mols):
        smiles = [lp['paths'][s][i] for s in range(n_steps)]
        qeds = [lp['qeds'][s][i] for s in range(len(lp['qeds']))]
        # SA may be missing for init step
        if len(lp['sas']) == n_steps:
            sas = [lp['sas'][s][i] for s in range(n_steps)]
        elif len(lp['sas']) == n_steps - 1:
            sas = [None] + [lp['sas'][s][i] for s in range(len(lp['sas']))]
        else:
            sas = [None] * n_steps

        mols.append({
            'smiles': smiles,
            'qeds': qeds,
            'sas': sas,
        })

    # Also extract top_paths
    top_paths = data.get('top_paths', [])

    return mols, episode, top_paths


def load_eval_paths(path):
    """Load eval results and return per-molecule data."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    mols = []
    for r in data['mol_results']:
        mols.append({
            'smiles': r['path_smiles'],
            'qeds': r['path_qeds'],
            'sas': r['path_sas'],
            'actions': r['path_actions'],
            'effective_steps': r['effective_steps'],
        })

    summary = data.get('summary', {})
    return mols, summary


def fmt_qed(q):
    return f"{q:.3f}" if q is not None else " N/A "


def fmt_sa(s):
    return f"{s:.2f}" if s is not None else " N/A"


def fmt_delta(prev, curr):
    if prev is None or curr is None:
        return ""
    d = curr - prev
    if abs(d) < 0.0005:
        return "  =="
    sign = "+" if d > 0 else ""
    return f"{sign}{d:+.3f}"[0:6]


def render_path(smiles, qeds, sas, actions=None, indent="  "):
    """Render a single molecule's synthesis path as lines of text."""
    lines = []
    n = len(smiles)
    for s in range(n):
        # Step label
        if s == 0:
            label = "init"
        else:
            label = f"S{s}  "

        # Change marker
        changed = s > 0 and smiles[s] != smiles[s-1]
        marker = ">>>" if changed else "   "

        # QED delta
        qed_delta = fmt_delta(qeds[s-1], qeds[s]) if s > 0 else "     "

        # Action info
        action_str = ""
        if actions and s > 0 and (s-1) < len(actions):
            act = actions[s-1]
            src = act.get('source', '?')
            if src == 'no_modification':
                action_str = "[keep]"
            else:
                action_str = "[rxn]"
                co = act.get('co_reactant')
                if co:
                    action_str += f"  + {co}"
                score = act.get('reaction_score')
                if score is not None:
                    action_str += f"  (score={score:.3f})"
            qv = act.get('q_value')
            if qv is not None:
                action_str += f"  Q={qv:.3f}"

        sa_str = fmt_sa(sas[s]) if s < len(sas) else " N/A"
        lines.append(
            f"{indent}{marker} {label} | QED={fmt_qed(qeds[s])} {qed_delta} | "
            f"SA={sa_str} | {smiles[s]}"
        )
        if action_str:
            lines.append(f"{indent}         {action_str}")

    return lines


def main():
    args = parse_args()

    if args.train is None and args.eval is None:
        print("Error: provide at least one of --train or --eval", file=sys.stderr)
        sys.exit(1)

    # Load data
    train_mols, train_ep, top_paths = None, None, []
    eval_mols, eval_summary = None, {}

    if args.train:
        train_mols, train_ep, top_paths = load_training_paths(args.train, args.train_episode)
    if args.eval:
        eval_mols, eval_summary = load_eval_paths(args.eval)

    # Determine number of molecules
    if train_mols and eval_mols:
        n_mols = min(len(train_mols), len(eval_mols))
    elif train_mols:
        n_mols = len(train_mols)
    else:
        n_mols = len(eval_mols)

    # Build molecule indices to show
    if args.mols is not None:
        indices = [i for i in args.mols if i < n_mols]
    else:
        indices = list(range(n_mols))

    # Sort
    def sort_key(i):
        if args.sort == 'index':
            return i
        if eval_mols:
            if args.sort == 'final_qed':
                return -eval_mols[i]['qeds'][-1]
            elif args.sort == 'improvement':
                return -(eval_mols[i]['qeds'][-1] - eval_mols[i]['qeds'][0])
            elif args.sort == 'init_qed':
                return -eval_mols[i]['qeds'][0]
        elif train_mols:
            if args.sort == 'final_qed':
                return -train_mols[i]['qeds'][-1]
            elif args.sort == 'improvement':
                return -(train_mols[i]['qeds'][-1] - train_mols[i]['qeds'][0])
            elif args.sort == 'init_qed':
                return -train_mols[i]['qeds'][0]
        return i

    indices.sort(key=sort_key)

    if args.top is not None:
        indices = indices[:args.top]

    # Render output
    out = []

    # Header
    out.append("=" * 100)
    out.append("SYNTHESIS PATH REPORT")
    out.append("=" * 100)

    if args.train:
        out.append(f"Training paths: {args.train} (episode {train_ep})")
    if args.eval:
        out.append(f"Eval paths:     {args.eval}")
    out.append(f"Molecules: {len(indices)} / {n_mols}")
    out.append(f"Sort: {args.sort}")
    out.append("")

    # Summary table
    if eval_mols:
        init_qeds = np.array([eval_mols[i]['qeds'][0] for i in range(n_mols)])
        final_qeds = np.array([eval_mols[i]['qeds'][-1] for i in range(n_mols)])
        improvements = final_qeds - init_qeds

        out.append("-" * 80)
        out.append("EVAL SUMMARY")
        out.append("-" * 80)
        out.append(f"  Init QED:    mean={init_qeds.mean():.3f}  median={np.median(init_qeds):.3f}")
        out.append(f"  Final QED:   mean={final_qeds.mean():.3f}  median={np.median(final_qeds):.3f}")
        out.append(f"  Improved (delta>0.01):  {(improvements > 0.01).sum()}/{n_mols}")
        out.append(f"  Degraded (delta<-0.01): {(improvements < -0.01).sum()}/{n_mols}")
        out.append(f"  Unchanged:              {((improvements >= -0.01) & (improvements <= 0.01)).sum()}/{n_mols}")
        out.append(f"  QED > 0.9: {(final_qeds > 0.9).sum()}/{n_mols}")
        out.append(f"  QED > 0.8: {(final_qeds > 0.8).sum()}/{n_mols}")
        out.append(f"  QED > 0.7: {(final_qeds > 0.7).sum()}/{n_mols}")
        out.append("")

    if train_mols:
        train_init = np.array([train_mols[i]['qeds'][0] for i in range(n_mols)])
        train_final = np.array([train_mols[i]['qeds'][-1] for i in range(n_mols)])
        train_imp = train_final - train_init
        out.append("-" * 80)
        out.append(f"TRAINING SUMMARY (episode {train_ep}, with eps-greedy exploration)")
        out.append("-" * 80)
        out.append(f"  Init QED:    mean={train_init.mean():.3f}  median={np.median(train_init):.3f}")
        out.append(f"  Final QED:   mean={train_final.mean():.3f}  median={np.median(train_final):.3f}")
        out.append(f"  Improved: {(train_imp > 0.01).sum()}/{n_mols}  "
                    f"Degraded: {(train_imp < -0.01).sum()}/{n_mols}")
        out.append("")

    # Top paths from training (global best across all episodes)
    if top_paths:
        out.append("-" * 80)
        out.append("GLOBAL TOP PATHS (best across all training episodes)")
        out.append("-" * 80)
        for i, tp in enumerate(top_paths):
            out.append(f"\n  Top {i+1}: QED={tp['final_qed']:.3f}  "
                        f"Reward={tp['final_reward']:.3f}  Episode={tp['episode']}")
            path_smi = tp['path']
            path_qed = tp['qeds']
            path_sa = tp.get('sas', [None] * len(path_smi))
            for s in range(len(path_smi)):
                changed = s > 0 and path_smi[s] != path_smi[s-1]
                marker = ">>>" if changed else "   "
                label = "init" if s == 0 else f"S{s}  "
                qed_d = fmt_delta(path_qed[s-1], path_qed[s]) if s > 0 else "     "
                sa_s = fmt_sa(path_sa[s]) if s < len(path_sa) else " N/A"
                out.append(f"    {marker} {label} | QED={fmt_qed(path_qed[s])} {qed_d} | "
                           f"SA={sa_s} | {path_smi[s]}")
        out.append("")

    # Per-molecule details
    out.append("")
    out.append("=" * 100)
    out.append("PER-MOLECULE SYNTHESIS PATHS")
    out.append("=" * 100)

    for rank, i in enumerate(indices):
        out.append("")
        out.append("-" * 100)

        # Header for this molecule
        init_smi = (eval_mols or train_mols)[i]['smiles'][0]
        init_qed = (eval_mols or train_mols)[i]['qeds'][0]

        header = f"Mol #{i}  (rank {rank+1}/{len(indices)})"
        if eval_mols:
            eq = eval_mols[i]
            delta = eq['qeds'][-1] - eq['qeds'][0]
            sign = "+" if delta >= 0 else ""
            header += (f"  |  Eval: {eq['qeds'][0]:.3f} -> {eq['qeds'][-1]:.3f} "
                       f"({sign}{delta:.3f})  steps={eq['effective_steps']}")
        out.append(header)
        out.append(f"  Init: {init_smi}")
        out.append("")

        # Eval path
        if eval_mols:
            eq = eval_mols[i]
            out.append("  [Eval - Greedy (eps=0)]")
            lines = render_path(
                eq['smiles'], eq['qeds'], eq['sas'],
                actions=eq.get('actions'), indent="    ")
            out.extend(lines)
            out.append("")

        # Training path
        if train_mols:
            tm = train_mols[i]
            out.append(f"  [Train - Episode {train_ep} (eps-greedy)]")
            lines = render_path(
                tm['smiles'], tm['qeds'], tm['sas'],
                actions=None, indent="    ")
            out.extend(lines)
            out.append("")

    # Footer
    out.append("")
    out.append("=" * 100)
    out.append(f"Total molecules shown: {len(indices)}")
    out.append("=" * 100)

    # Output
    text = "\n".join(out)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(text + "\n")
        print(f"Saved to {args.output} ({len(indices)} molecules)")
    else:
        print(text)


if __name__ == '__main__':
    main()
