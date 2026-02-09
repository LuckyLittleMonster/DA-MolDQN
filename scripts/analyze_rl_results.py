#!/usr/bin/env python
"""
RL Results Analysis Tool.

Parses Experiments/*.pickle files and generates a comparative markdown report
across all available RL experiments.

Usage:
    python scripts/analyze_rl_results.py
    python scripts/analyze_rl_results.py --experiments qed_mlp_0 qed_planbc_0
    python scripts/analyze_rl_results.py -o report.md
    python scripts/analyze_rl_results.py --convergence-plot convergence.png
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP_DIR = Path("Experiments")


# =============================================================================
# Data Loading
# =============================================================================

def discover_experiments(exp_dir: Path = EXP_DIR) -> Dict[str, Dict[str, Path]]:
    """Discover all experiment pickle files and group by experiment name."""
    experiments = {}
    for f in sorted(exp_dir.iterdir()):
        if not f.name.endswith(".pickle"):
            continue
        # Skip non-experiment files
        if f.name.startswith(("test_", ".")):
            continue
        stem = f.stem  # e.g. qed_mlp_0_history
        # Detect suffix: history, paths, eval
        for suffix in ("_history", "_paths", "_eval"):
            if stem.endswith(suffix):
                name = stem[: -len(suffix)]
                if name not in experiments:
                    experiments[name] = {}
                experiments[name][suffix.strip("_")] = f
                break
    return experiments


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_history(path: Path) -> Optional[Dict]:
    """Load training history (rewards, qeds, sas per episode)."""
    try:
        data = load_pickle(path)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}", file=sys.stderr)
        return None


def load_paths(path: Path) -> Optional[Dict]:
    """Load training paths (top_paths, last_paths)."""
    try:
        return load_pickle(path)
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}", file=sys.stderr)
        return None


def load_eval(path: Path) -> Optional[Dict]:
    """Load eval results. Handles both new format (mol_results) and old format."""
    try:
        data = load_pickle(path)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        print(f"  Warning: failed to load {path}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_history(history: Dict) -> Dict:
    """Compute training curve statistics from history."""
    rewards = np.array(history.get("rewards", []))
    qeds = np.array(history.get("qeds", []))
    sas = np.array(history.get("sas", []))

    n_episodes = len(rewards)
    if n_episodes == 0:
        return {"n_episodes": 0}

    # Smooth with rolling window
    window = max(1, n_episodes // 20)

    def rolling_mean(arr, w):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    stats = {
        "n_episodes": n_episodes,
        "reward_final": float(rewards[-1]),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
    }

    if len(qeds) > 0:
        stats["qed_final"] = float(qeds[-1])
        stats["qed_mean"] = float(qeds.mean())
        stats["qed_max"] = float(qeds.max())
        # Smoothed peak
        smoothed = rolling_mean(qeds, window)
        stats["qed_smoothed_max"] = float(smoothed.max()) if len(smoothed) > 0 else 0.0
        # First episode to reach 0.7, 0.8, 0.9
        for threshold in [0.7, 0.8, 0.9]:
            above = np.where(smoothed >= threshold)[0]
            stats[f"qed_first_above_{threshold}"] = int(above[0]) if len(above) > 0 else None

    if len(sas) > 0:
        stats["sa_final"] = float(sas[-1])
        stats["sa_mean"] = float(sas.mean())

    # Convergence: is the reward still improving in last 10%?
    if n_episodes >= 20:
        last_10pct = rewards[-n_episodes // 10:]
        first_10pct = rewards[:n_episodes // 10]
        stats["converged"] = float(np.mean(last_10pct)) - float(np.mean(first_10pct))

    return stats


def analyze_paths(paths_data: Dict) -> Dict:
    """Analyze synthesis paths from training."""
    stats = {}

    top_paths = paths_data.get("top_paths", [])
    if top_paths:
        stats["n_top_paths"] = len(top_paths)
        stats["best_qed"] = max(tp["final_qed"] for tp in top_paths)
        stats["best_reward"] = max(tp["final_reward"] for tp in top_paths)
        # Path lengths (effective steps with actual changes)
        path_lengths = []
        for tp in top_paths:
            path_smi = tp["path"]
            changes = sum(1 for i in range(1, len(path_smi)) if path_smi[i] != path_smi[i - 1])
            path_lengths.append(changes)
        stats["avg_effective_steps"] = float(np.mean(path_lengths))

    last_paths = paths_data.get("last_paths", [])
    if last_paths:
        # Use the last snapshot
        lp = last_paths[-1]
        n_mols = len(lp["paths"][0])
        n_steps = len(lp["paths"])
        init_qeds = np.array(lp["qeds"][0])
        final_qeds = np.array(lp["qeds"][-1])
        improvements = final_qeds - init_qeds

        stats["last_episode"] = lp["episode"]
        stats["n_mols"] = n_mols
        stats["n_steps"] = n_steps - 1  # exclude init
        stats["last_init_qed_mean"] = float(init_qeds.mean())
        stats["last_final_qed_mean"] = float(final_qeds.mean())
        stats["last_improved_count"] = int((improvements > 0.01).sum())
        stats["last_degraded_count"] = int((improvements < -0.01).sum())

    return stats


def analyze_eval(eval_data: Dict) -> Dict:
    """Analyze evaluation results."""
    stats = {}

    # New format with mol_results
    if "mol_results" in eval_data:
        results = eval_data["mol_results"]
        n = len(results)
        init_qeds = np.array([r["init_qed"] for r in results])
        final_qeds = np.array([r["final_qed"] for r in results])
        improvements = final_qeds - init_qeds

        stats["n_mols"] = n
        stats["init_qed_mean"] = float(init_qeds.mean())
        stats["init_qed_median"] = float(np.median(init_qeds))
        stats["final_qed_mean"] = float(final_qeds.mean())
        stats["final_qed_median"] = float(np.median(final_qeds))
        stats["improvement_mean"] = float(improvements.mean())
        stats["improvement_median"] = float(np.median(improvements))
        stats["n_improved"] = int((improvements > 0.01).sum())
        stats["n_degraded"] = int((improvements < -0.01).sum())
        stats["n_unchanged"] = n - stats["n_improved"] - stats["n_degraded"]
        stats["pct_improved"] = stats["n_improved"] / n * 100
        stats["qed_above_0.9"] = int((final_qeds > 0.9).sum())
        stats["qed_above_0.8"] = int((final_qeds > 0.8).sum())
        stats["qed_above_0.7"] = int((final_qeds > 0.7).sum())

        # Effective steps distribution
        eff_steps = [r["effective_steps"] for r in results]
        stats["avg_effective_steps"] = float(np.mean(eff_steps))
        stats["max_steps"] = eval_data.get("args", {}).get("max_steps", "?")

        # SA scores
        init_sas = np.array([r.get("init_sa", 0) for r in results])
        final_sas = np.array([r.get("final_sa", 0) for r in results])
        if init_sas.sum() > 0:
            stats["init_sa_mean"] = float(init_sas.mean())
            stats["final_sa_mean"] = float(final_sas.mean())

        # Molecular diversity: Tanimoto among final molecules
        stats["diversity"] = _compute_diversity([r["final_smiles"] for r in results])

    # Old format (single molecule eval)
    elif "path" in eval_data and "qeds" in eval_data:
        qeds = eval_data["qeds"]
        n_steps = len(qeds)
        if n_steps > 0 and isinstance(qeds[0], list):
            # Multi-mol format
            n_mols = len(qeds[0])
            init_qeds = np.array(qeds[0])
            final_qeds = np.array(qeds[-1])
        elif n_steps > 0:
            n_mols = 1
            init_qeds = np.array([qeds[0]])
            final_qeds = np.array([qeds[-1]])
        else:
            return stats
        improvements = final_qeds - init_qeds
        stats["n_mols"] = n_mols
        stats["format"] = "legacy"
        stats["init_qed_mean"] = float(init_qeds.mean())
        stats["final_qed_mean"] = float(final_qeds.mean())
        stats["improvement_mean"] = float(improvements.mean())

    return stats


def _compute_diversity(smiles_list: List[str]) -> Optional[float]:
    """Compute average pairwise Tanimoto distance among molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return None

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    # Diversity = 1 - average similarity
    return 1.0 - float(np.mean(sims))


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(experiments: Dict[str, Dict[str, Path]],
                    selected: Optional[List[str]] = None) -> str:
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# RL Experiment Results Analysis")
    lines.append("")
    lines.append(f"Experiments directory: `{EXP_DIR}`")
    lines.append(f"Total experiments found: {len(experiments)}")
    lines.append("")

    if selected:
        experiments = {k: v for k, v in experiments.items() if k in selected}

    # Collect all analysis results
    all_stats = {}
    for name in sorted(experiments):
        files = experiments[name]
        stats = {"name": name, "files": list(files.keys())}

        if "history" in files:
            h = load_history(files["history"])
            if h:
                stats["history"] = analyze_history(h)

        if "paths" in files:
            p = load_paths(files["paths"])
            if p:
                stats["paths"] = analyze_paths(p)

        if "eval" in files:
            e = load_eval(files["eval"])
            if e:
                stats["eval"] = analyze_eval(e)

        all_stats[name] = stats

    # ===== Comparison Table =====
    lines.append("## Summary Comparison")
    lines.append("")

    # Filter to experiments with meaningful data
    meaningful = {k: v for k, v in all_stats.items()
                  if "history" in v and v["history"].get("n_episodes", 0) > 10}

    if meaningful:
        lines.append("### Training Curves")
        lines.append("")
        lines.append("| Experiment | Episodes | Reward (final) | QED (max) | QED (smoothed max) | SA (final) |")
        lines.append("|:-----------|-------:|-------:|------:|------:|------:|")
        for name, s in sorted(meaningful.items()):
            h = s["history"]
            ep = h["n_episodes"]
            r_final = f'{h.get("reward_final", 0):.3f}'
            q_max = f'{h.get("qed_max", 0):.3f}'
            q_sm = f'{h.get("qed_smoothed_max", 0):.3f}'
            sa = f'{h.get("sa_final", 0):.2f}'
            lines.append(f"| {name} | {ep} | {r_final} | {q_max} | {q_sm} | {sa} |")
        lines.append("")

    # Eval comparison (include both new and legacy format)
    eval_exps = {k: v for k, v in all_stats.items()
                 if "eval" in v and "final_qed_mean" in v.get("eval", {})}
    if eval_exps:
        lines.append("### Evaluation Results")
        lines.append("")
        lines.append("| Experiment | N | Init QED | Final QED | Improvement | Improved | Degraded | QED>0.8 | QED>0.9 | Diversity |")
        lines.append("|:-----------|---:|------:|------:|------:|------:|------:|------:|------:|------:|")
        for name, s in sorted(eval_exps.items()):
            e = s["eval"]
            n = e["n_mols"]
            iq = f'{e["init_qed_mean"]:.3f}'
            fq = f'{e["final_qed_mean"]:.3f}'
            imp = f'{e.get("improvement_mean", 0):+.3f}'
            ni = f'{e.get("n_improved", "?")} ({e.get("pct_improved", 0):.0f}%)'
            nd = str(e.get("n_degraded", "?"))
            q8 = str(e.get("qed_above_0.8", "?"))
            q9 = str(e.get("qed_above_0.9", "?"))
            div = f'{e["diversity"]:.3f}' if e.get("diversity") is not None else "N/A"
            lines.append(f"| {name} | {n} | {iq} | {fq} | {imp} | {ni} | {nd} | {q8} | {q9} | {div} |")
        lines.append("")

    # Top paths comparison
    paths_exps = {k: v for k, v in all_stats.items() if "paths" in v and "best_qed" in v.get("paths", {})}
    if paths_exps:
        lines.append("### Best Synthesis Paths (Training)")
        lines.append("")
        lines.append("| Experiment | Best QED | Best Reward | Avg Effective Steps |")
        lines.append("|:-----------|------:|------:|------:|")
        for name, s in sorted(paths_exps.items()):
            p = s["paths"]
            bq = f'{p["best_qed"]:.4f}'
            br = f'{p["best_reward"]:.4f}'
            es = f'{p.get("avg_effective_steps", 0):.1f}'
            lines.append(f"| {name} | {bq} | {br} | {es} |")
        lines.append("")

    # ===== Per-Experiment Details =====
    lines.append("---")
    lines.append("")
    lines.append("## Per-Experiment Details")
    lines.append("")

    for name, s in sorted(all_stats.items()):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Available files: {', '.join(s['files'])}")
        lines.append("")

        if "history" in s:
            h = s["history"]
            lines.append(f"**Training** ({h['n_episodes']} episodes)")
            lines.append("")
            lines.append(f"- Reward: final={h.get('reward_final', 0):.3f}, mean={h.get('reward_mean', 0):.3f}, max={h.get('reward_max', 0):.3f}")
            if "qed_max" in h:
                lines.append(f"- QED: final={h.get('qed_final', 0):.3f}, mean={h.get('qed_mean', 0):.3f}, max={h.get('qed_max', 0):.3f}")
                for t in [0.7, 0.8, 0.9]:
                    ep = h.get(f"qed_first_above_{t}")
                    lines.append(f"  - First episode QED >= {t}: {ep if ep is not None else 'never'}")
            if "sa_mean" in h:
                lines.append(f"- SA: final={h.get('sa_final', 0):.2f}, mean={h.get('sa_mean', 0):.2f}")
            if "converged" in h:
                delta = h["converged"]
                status = "improving" if delta > 0.01 else ("converged" if abs(delta) <= 0.01 else "degrading")
                lines.append(f"- Convergence: last 10% vs first 10% delta = {delta:+.3f} ({status})")
            lines.append("")

        if "paths" in s:
            p = s["paths"]
            if "best_qed" in p:
                lines.append(f"**Top Paths**: best QED = {p['best_qed']:.4f}, best reward = {p['best_reward']:.4f}")
                lines.append(f"- Avg effective steps: {p.get('avg_effective_steps', 0):.1f}")
            if "last_final_qed_mean" in p:
                lines.append(f"**Last Episode** ({p.get('last_episode', '?')}): "
                             f"init QED = {p['last_init_qed_mean']:.3f}, "
                             f"final QED = {p['last_final_qed_mean']:.3f}, "
                             f"improved = {p['last_improved_count']}/{p['n_mols']}")
            lines.append("")

        if "eval" in s:
            e = s["eval"]
            if e.get("format") == "legacy":
                imp = e.get("improvement_mean", 0)
                lines.append(f"**Eval** (legacy format, {e['n_mols']} mol): "
                             f"init QED = {e['init_qed_mean']:.3f}, "
                             f"final QED = {e['final_qed_mean']:.3f}, "
                             f"delta = {imp:+.3f}")
            elif "final_qed_mean" in e:
                lines.append(f"**Eval** ({e['n_mols']} molecules, max_steps={e.get('max_steps', '?')})")
                lines.append("")
                lines.append(f"- Init QED: mean={e['init_qed_mean']:.3f}, median={e.get('init_qed_median', 0):.3f}")
                lines.append(f"- Final QED: mean={e['final_qed_mean']:.3f}, median={e.get('final_qed_median', 0):.3f}")
                lines.append(f"- Improvement: mean={e.get('improvement_mean', 0):+.3f}, median={e.get('improvement_median', 0):+.3f}")
                lines.append(f"- Improved: {e.get('n_improved', '?')}/{e['n_mols']} ({e.get('pct_improved', 0):.0f}%), "
                             f"Degraded: {e.get('n_degraded', '?')}/{e['n_mols']}")
                lines.append(f"- QED > 0.9: {e.get('qed_above_0.9', '?')}, "
                             f"QED > 0.8: {e.get('qed_above_0.8', '?')}, "
                             f"QED > 0.7: {e.get('qed_above_0.7', '?')}")
                lines.append(f"- Avg effective steps: {e.get('avg_effective_steps', 0):.1f}")
                if e.get("init_sa_mean"):
                    lines.append(f"- SA: init={e['init_sa_mean']:.2f}, final={e['final_sa_mean']:.2f}")
                if e.get("diversity") is not None:
                    lines.append(f"- Molecular diversity: {e['diversity']:.3f}")
            lines.append("")

    return "\n".join(lines)


def generate_convergence_data(experiments: Dict[str, Dict[str, Path]],
                              selected: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Extract QED training curves for convergence plotting."""
    if selected:
        experiments = {k: v for k, v in experiments.items() if k in selected}

    curves = {}
    for name in sorted(experiments):
        files = experiments[name]
        if "history" not in files:
            continue
        h = load_history(files["history"])
        if h is None:
            continue
        qeds = np.array(h.get("qeds", []))
        if len(qeds) > 10:
            curves[name] = qeds
    return curves


def save_convergence_plot(curves: Dict[str, np.ndarray], output_path: str):
    """Save convergence plot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, qeds in sorted(curves.items()):
        # Smooth
        window = max(1, len(qeds) // 50)
        kernel = np.ones(window) / window
        smoothed = np.convolve(qeds, kernel, mode="valid")
        ax.plot(smoothed, label=name, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean QED")
    ax.set_title("RL Training Convergence (Smoothed QED)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot to {output_path}")


# =============================================================================
# QED Distribution Analysis
# =============================================================================

def generate_qed_distribution_report(experiments: Dict[str, Dict[str, Path]],
                                     selected: Optional[List[str]] = None) -> str:
    """Generate QED distribution comparison across eval experiments."""
    if selected:
        experiments = {k: v for k, v in experiments.items() if k in selected}

    lines = []
    lines.append("## QED Distribution Comparison")
    lines.append("")

    eval_exps = {}
    for name, files in experiments.items():
        if "eval" not in files:
            continue
        e = load_eval(files["eval"])
        if e is None or "mol_results" not in e:
            continue
        final_qeds = [r["final_qed"] for r in e["mol_results"]]
        init_qeds = [r["init_qed"] for r in e["mol_results"]]
        eval_exps[name] = {"init": np.array(init_qeds), "final": np.array(final_qeds)}

    if not eval_exps:
        lines.append("No eval data with mol_results found.")
        return "\n".join(lines)

    # Percentile comparison
    percentiles = [10, 25, 50, 75, 90]
    lines.append("### Final QED Percentiles")
    lines.append("")
    header = "| Experiment | " + " | ".join(f"P{p}" for p in percentiles) + " |"
    sep = "|:-----------|" + "|".join("------:" for _ in percentiles) + "|"
    lines.append(header)
    lines.append(sep)
    for name, data in sorted(eval_exps.items()):
        vals = [f"{np.percentile(data['final'], p):.3f}" for p in percentiles]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")

    # Improvement histogram (text-based)
    lines.append("### Improvement Distribution")
    lines.append("")
    bins = [(-1, -0.2), (-0.2, -0.05), (-0.05, 0.05), (0.05, 0.2), (0.2, 1)]
    bin_labels = ["<-0.2", "-0.2~-0.05", "-0.05~0.05", "0.05~0.2", ">0.2"]
    header = "| Experiment | " + " | ".join(bin_labels) + " |"
    sep = "|:-----------|" + "|".join("------:" for _ in bins) + "|"
    lines.append(header)
    lines.append(sep)
    for name, data in sorted(eval_exps.items()):
        imps = data["final"] - data["init"]
        counts = []
        for lo, hi in bins:
            c = int(((imps >= lo) & (imps < hi)).sum())
            counts.append(str(c))
        lines.append(f"| {name} | " + " | ".join(counts) + " |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Analyze RL experiment results")
    p.add_argument("--experiments", "-e", nargs="+", default=None,
                   help="Specific experiments to analyze (default: all)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output markdown file (default: stdout)")
    p.add_argument("--convergence-plot", type=str, default=None,
                   help="Save convergence plot to this path (PNG)")
    p.add_argument("--exp-dir", type=str, default="Experiments",
                   help="Experiments directory")
    return p.parse_args()


def main():
    args = parse_args()
    global EXP_DIR
    EXP_DIR = Path(args.exp_dir)

    experiments = discover_experiments(EXP_DIR)
    if not experiments:
        print(f"No experiments found in {EXP_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} experiments: {', '.join(sorted(experiments))}", file=sys.stderr)

    report = generate_report(experiments, args.experiments)
    report += "\n" + generate_qed_distribution_report(experiments, args.experiments)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)

    if args.convergence_plot:
        curves = generate_convergence_data(experiments, args.experiments)
        if curves:
            save_convergence_plot(curves, args.convergence_plot)
        else:
            print("No training curves available for plotting", file=sys.stderr)


if __name__ == "__main__":
    main()
