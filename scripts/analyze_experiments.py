"""Comprehensive analysis of ReaSyn RL experiment results."""
import pickle
import numpy as np
from collections import defaultdict

BASE = "Experiments"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def print_header(title, char="="):
    width = 80
    print(f"\n{char*width}")
    print(f"  {title}")
    print(f"{char*width}")

def print_subheader(title, char="-"):
    print(f"\n  {char*40}")
    print(f"  {title}")
    print(f"  {char*40}")

def analyze_history(name, history):
    """Analyze the episode-level history dict."""
    print_header(f"EXPERIMENT: {name}")
    
    episodes = history["episode"]
    mean_rewards = history["mean_reward"]
    mean_scores = history["mean_score"]
    best_scores = history["best_score"]
    best_ever = history["best_score_ever"]
    eps_vals = history["eps"]
    time_s = history["time_s"]
    losses = history["loss"]
    cache_mols = history["cache_mols"]
    cache_edges = history["cache_edges"]
    reward_mode = history.get("reward_mode", "unknown")
    config = history.get("config", {})
    
    n = len(episodes)
    
    # Basic info
    print(f"\n  Reward mode: {reward_mode}")
    print(f"  Total episodes: {n}")
    print(f"  Config highlights:")
    for k in ["method", "reward", "exp_name", "max_steps", "batch_size", "replay_size", "lr", 
              "gamma", "eps_start", "eps_end", "eps_decay", "n_mols"]:
        if k in config:
            print(f"    {k}: {config[k]}")
    
    # Timing
    total_time = sum(time_s)
    avg_time = np.mean(time_s)
    print(f"\n  Total wall time: {total_time/3600:.2f} hours ({total_time:.0f}s)")
    print(f"  Avg time/episode: {avg_time:.1f}s")
    print(f"  Eps range: {eps_vals[0]:.4f} -> {eps_vals[-1]:.4f}")
    
    # Reward analysis
    print_subheader("REWARD ANALYSIS")
    mr = np.array(mean_rewards)
    print(f"  Mean reward range: {mr.min():.4f} to {mr.max():.4f}")
    print(f"  Mean reward (all): {mr.mean():.4f} +/- {mr.std():.4f}")
    
    # Best scores (docking = more negative is better)
    bs = np.array(best_scores)
    be = np.array(best_ever)
    print(f"\n  Best score (per-episode): min={bs.min():.4f}, max={bs.max():.4f}")
    print(f"  Best score ever (running): final={be[-1]:.4f}")
    
    # Find best episode
    best_ep_idx = np.argmin(bs)  # most negative dock score
    print(f"  Best single-episode score: {bs[best_ep_idx]:.4f} at episode {episodes[best_ep_idx]}")
    
    # Top-K analysis on mean_reward
    sorted_mr = np.sort(mr)[::-1]  # descending
    print(f"\n  Top-10 avg mean_reward: {sorted_mr[:10].mean():.4f}")
    if n >= 100:
        print(f"  Top-100 avg mean_reward: {sorted_mr[:100].mean():.4f}")
    print(f"  Bottom-10 avg mean_reward: {sorted_mr[-10:].mean():.4f}")
    
    # Top-K on best_score (most negative = best for docking)
    sorted_bs = np.sort(bs)  # ascending (most negative first)
    print(f"\n  Top-10 avg best_score: {sorted_bs[:10].mean():.4f}")
    if n >= 100:
        print(f"  Top-100 avg best_score: {sorted_bs[:100].mean():.4f}")
    
    # Learning curve: split into 5 phases
    print_subheader("LEARNING CURVE (5 phases)")
    phase_size = n // 5
    for i in range(5):
        start = i * phase_size
        end = (i+1) * phase_size if i < 4 else n
        phase_mr = mr[start:end]
        phase_bs = bs[start:end]
        phase_be = be[start:end]
        ep_range = f"ep {episodes[start]}-{episodes[end-1]}"
        print(f"  Phase {i+1} ({ep_range}):")
        print(f"    mean_reward: {phase_mr.mean():.4f} +/- {phase_mr.std():.4f}")
        print(f"    best_score:  avg={phase_bs.mean():.4f}, min={phase_bs.min():.4f}")
        print(f"    best_ever:   {phase_be[-1]:.4f}")
    
    # Loss curve
    print_subheader("LOSS CURVE")
    l = np.array(losses)
    for i in range(5):
        start = i * phase_size
        end = (i+1) * phase_size if i < 4 else n
        phase_l = l[start:end]
        ep_range = f"ep {episodes[start]}-{episodes[end-1]}"
        print(f"  Phase {i+1} ({ep_range}): loss={phase_l.mean():.6f} +/- {phase_l.std():.6f}")
    
    # Cache growth
    print_subheader("CACHE GROWTH")
    print(f"  Start: {cache_mols[0]} mols, {cache_edges[0]} edges")
    print(f"  End:   {cache_mols[-1]} mols, {cache_edges[-1]} edges")
    
    return mr, bs, be

def analyze_paths(name, paths):
    """Analyze synthesis paths."""
    print_subheader(f"SYNTHESIS PATHS ({name})")
    print(f"  Number of saved paths: {len(paths)}")
    
    all_final_smiles = []
    for p in paths:
        ep = p["episode"]
        init = p["init_smiles"]
        final = p["final_smiles"]
        best = p["best_score"]
        n_steps = len(p["path"])
        scores = p["scores"]
        rewards = p["rewards"]
        
        all_final_smiles.append(final)
        
        print(f"\n  Episode {ep}:")
        print(f"    Init:  {init}")
        print(f"    Final: {final}")
        print(f"    Steps: {n_steps}")
        print(f"    Best dock score: {best:.4f}")
        print(f"    Score trajectory: {' -> '.join(f'{s:.2f}' for s in scores)}")
        print(f"    Reward trajectory: {' -> '.join(f'{r:.3f}' for r in rewards)}")
        
        # Show synthesis steps
        synthesis = p.get("synthesis", [])
        if synthesis:
            print(f"    Synthesis steps:")
            for si, s in enumerate(synthesis):
                print(f"      Step {si+1}: {s[:120]}{'...' if len(s) > 120 else ''}")
    
    # Diversity of final molecules
    if len(all_final_smiles) > 1:
        unique = len(set(all_final_smiles))
        print(f"\n  Unique final SMILES: {unique}/{len(all_final_smiles)}")

def analyze_recent_episodes(name, recent):
    """Analyze recent episodes with per-molecule data."""
    print_subheader(f"RECENT EPISODES ({name})")
    print(f"  Number of recent episode snapshots: {len(recent)}")
    
    all_scores = []
    all_rewards = []
    all_final_smiles = []
    
    for ep_data in recent:
        ep = ep_data["episode"]
        mols = ep_data["molecules"]
        print(f"\n  Episode {ep}: {len(mols)} molecules")
        
        ep_scores = []
        ep_rewards = []
        for m in mols:
            final = m.get("final_smiles", "")
            path = m.get("path", [])
            all_final_smiles.append(final)
            
            # Try to extract scores/rewards
            scores = m.get("scores", [])
            rewards = m.get("rewards", [])
            dock = m.get("dock_score", m.get("dock_proxy", None))
            qed = m.get("qed", None)
            sa = m.get("sa", None)
            reward = m.get("reward", m.get("total_reward", None))
            
            if scores:
                ep_scores.extend(scores)
            if dock is not None:
                ep_scores.append(dock)
            if rewards:
                ep_rewards.extend(rewards)
            if reward is not None:
                ep_rewards.append(reward)
        
        # Summarize
        if ep_scores:
            print(f"    Scores: mean={np.mean(ep_scores):.4f}, min={np.min(ep_scores):.4f}, max={np.max(ep_scores):.4f}")
        if ep_rewards:
            print(f"    Rewards: mean={np.mean(ep_rewards):.4f}")
        
        # Show a few sample molecules
        for i, m in enumerate(mols[:3]):
            print(f"    Mol {i}: {m.get('init_smiles', '?')[:40]} -> {m.get('final_smiles', '?')[:40]}")
            if "path" in m:
                print(f"      Path length: {len(m['path'])} steps")
            # Print all available keys for first mol in first episode (for debugging)
            if i == 0 and ep == recent[0]["episode"]:
                print(f"      Available keys: {list(m.keys())}")
                # Print any numeric values
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        print(f"      {k} = {v}")
    
    # Overall diversity
    unique_finals = set(all_final_smiles)
    print(f"\n  Total molecules across recent episodes: {len(all_final_smiles)}")
    print(f"  Unique final SMILES: {len(unique_finals)}")
    if len(unique_finals) > 0:
        print(f"  Diversity ratio: {len(unique_finals)/len(all_final_smiles):.2%}")

def compare_moo_trials(h1, h2):
    """Compare two MOO trials for consistency."""
    print_header("MOO TRIAL COMPARISON: proxy_1 vs proxy_2")
    
    mr1 = np.array(h1["mean_reward"])
    mr2 = np.array(h2["mean_reward"])
    bs1 = np.array(h1["best_score"])
    bs2 = np.array(h2["best_score"])
    be1 = np.array(h1["best_score_ever"])
    be2 = np.array(h2["best_score_ever"])
    
    print(f"\n  {'Metric':<30} {'Trial 1':>12} {'Trial 2':>12} {'Diff':>12}")
    print(f"  {'-'*66}")
    
    metrics = [
        ("Final best_score_ever", be1[-1], be2[-1]),
        ("Mean reward (all)", mr1.mean(), mr2.mean()),
        ("Mean reward (last 100)", mr1[-100:].mean(), mr2[-100:].mean()),
        ("Mean reward (top 10)", np.sort(mr1)[-10:].mean(), np.sort(mr2)[-10:].mean()),
        ("Best score (best)", bs1.min(), bs2.min()),
        ("Best score (avg)", bs1.mean(), bs2.mean()),
        ("Best score (last 100 avg)", bs1[-100:].mean(), bs2[-100:].mean()),
    ]
    
    for name, v1, v2 in metrics:
        diff = v2 - v1
        print(f"  {name:<30} {v1:>12.4f} {v2:>12.4f} {diff:>+12.4f}")
    
    # Correlation
    corr_mr = np.corrcoef(mr1, mr2)[0, 1]
    corr_bs = np.corrcoef(bs1, bs2)[0, 1]
    print(f"\n  Correlation (mean_reward): {corr_mr:.4f}")
    print(f"  Correlation (best_score):  {corr_bs:.4f}")
    
    # Phase-by-phase comparison
    print_subheader("PHASE-BY-PHASE COMPARISON")
    n = len(mr1)
    phase_size = n // 5
    print(f"\n  {'Phase':<10} {'T1 mean_rew':>12} {'T2 mean_rew':>12} {'T1 best_sc':>12} {'T2 best_sc':>12}")
    print(f"  {'-'*58}")
    for i in range(5):
        start = i * phase_size
        end = (i+1) * phase_size if i < 4 else n
        ep_range = f"P{i+1}"
        print(f"  {ep_range:<10} {mr1[start:end].mean():>12.4f} {mr2[start:end].mean():>12.4f} "
              f"{bs1[start:end].mean():>12.4f} {bs2[start:end].mean():>12.4f}")


def compare_dock_vs_moo(dock_h, moo_h, dock_name="Dock", moo_name="MOO"):
    """Compare dock-only vs MOO experiments."""
    print_header(f"COMPARISON: {dock_name} vs {moo_name}")
    
    mr_d = np.array(dock_h["mean_reward"])
    mr_m = np.array(moo_h["mean_reward"])
    bs_d = np.array(dock_h["best_score"])
    bs_m = np.array(moo_h["best_score"])
    be_d = np.array(dock_h["best_score_ever"])
    be_m = np.array(moo_h["best_score_ever"])
    
    print(f"\n  {'Metric':<35} {dock_name:>12} {moo_name:>12}")
    print(f"  {'-'*59}")
    
    metrics = [
        ("Reward mode", dock_h["reward_mode"], moo_h["reward_mode"]),
        ("Final best_score_ever", f"{be_d[-1]:.4f}", f"{be_m[-1]:.4f}"),
        ("Mean reward (all)", f"{mr_d.mean():.4f}", f"{mr_m.mean():.4f}"),
        ("Mean reward (last 100)", f"{mr_d[-100:].mean():.4f}", f"{mr_m[-100:].mean():.4f}"),
        ("Mean reward (top 10)", f"{np.sort(mr_d)[-10:].mean():.4f}", f"{np.sort(mr_m)[-10:].mean():.4f}"),
        ("Best score (best)", f"{bs_d.min():.4f}", f"{bs_m.min():.4f}"),
        ("Best score (avg last 100)", f"{bs_d[-100:].mean():.4f}", f"{bs_m[-100:].mean():.4f}"),
        ("Total time (hours)", f"{sum(dock_h['time_s'])/3600:.2f}", f"{sum(moo_h['time_s'])/3600:.2f}"),
        ("Final cache mols", f"{dock_h['cache_mols'][-1]}", f"{moo_h['cache_mols'][-1]}"),
        ("Final cache edges", f"{dock_h['cache_edges'][-1]}", f"{moo_h['cache_edges'][-1]}"),
    ]
    
    for row in metrics:
        if len(row) == 3:
            name, v1, v2 = row
            print(f"  {name:<35} {v1:>12} {v2:>12}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("\n" + "#"*80)
print("#  REASYN RL EXPERIMENT ANALYSIS")
print("#  Date: 2026-02-20")
print("#"*80)

# Load all data
dock_h = load_pickle(f"{BASE}/reasyn_dock_proxy_1_history.pickle")
dock_p = load_pickle(f"{BASE}/reasyn_dock_proxy_1_paths.pickle")
dock_r = load_pickle(f"{BASE}/reasyn_dock_proxy_1_recent_episodes.pickle")

moo2_h = load_pickle(f"{BASE}/reasyn_multi_proxy_2_history.pickle")
moo2_p = load_pickle(f"{BASE}/reasyn_multi_proxy_2_paths.pickle")
moo2_r = load_pickle(f"{BASE}/reasyn_multi_proxy_2_recent_episodes.pickle")

moo1_h = load_pickle(f"{BASE}/reasyn_multi_proxy_1_history.pickle")

# Detailed analysis for each experiment
for name, h in [("ReaSyn + sEH Dock Proxy (Trial 1)", dock_h),
                ("ReaSyn + sEH MOO Proxy (Trial 1)", moo1_h),
                ("ReaSyn + sEH MOO Proxy (Trial 2)", moo2_h)]:
    analyze_history(name, h)

# Paths analysis
analyze_paths("Dock Proxy 1", dock_p)
analyze_paths("MOO Proxy 2", moo2_p)

# Recent episodes analysis
analyze_recent_episodes("Dock Proxy 1", dock_r)
analyze_recent_episodes("MOO Proxy 2", moo2_r)

# Comparisons
compare_moo_trials(moo1_h, moo2_h)
compare_dock_vs_moo(dock_h, moo2_h, "Dock Proxy 1", "MOO Proxy 2")

# Final summary
print_header("EXECUTIVE SUMMARY")
dock_be = np.array(dock_h["best_score_ever"])
moo1_be = np.array(moo1_h["best_score_ever"])
moo2_be = np.array(moo2_h["best_score_ever"])
dock_mr = np.array(dock_h["mean_reward"])
moo1_mr = np.array(moo1_h["mean_reward"])
moo2_mr = np.array(moo2_h["mean_reward"])

print(f"""
  Three experiments completed (500 episodes each):
  
  1. Dock Proxy 1 (reward=dock):
     - Best dock score: {dock_be[-1]:.4f}
     - Mean reward (last 100): {dock_mr[-100:].mean():.4f}
     - Top-10 mean reward: {np.sort(dock_mr)[-10:].mean():.4f}
  
  2. MOO Proxy 1 (reward=multi):
     - Best dock score: {moo1_be[-1]:.4f}
     - Mean reward (last 100): {moo1_mr[-100:].mean():.4f}
     - Top-10 mean reward: {np.sort(moo1_mr)[-10:].mean():.4f}
  
  3. MOO Proxy 2 (reward=multi):
     - Best dock score: {moo2_be[-1]:.4f}
     - Mean reward (last 100): {moo2_mr[-100:].mean():.4f}
     - Top-10 mean reward: {np.sort(moo2_mr)[-10:].mean():.4f}
  
  Key findings:
  - Dock-only vs MOO: dock-only optimizes pure docking score harder
  - MOO trials consistency: seed variation between trial 1 and 2
  - All experiments ran for {sum(dock_h['time_s'])/3600:.1f}h / {sum(moo1_h['time_s'])/3600:.1f}h / {sum(moo2_h['time_s'])/3600:.1f}h
""")

