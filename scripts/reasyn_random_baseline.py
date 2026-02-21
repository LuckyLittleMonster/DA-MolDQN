"""Compare two ReaSyn random search baselines (no DQN).

Strategy 1 (One-shot Wide): Generate ~N candidates in one step, pick best reward.
Strategy 2 (Greedy Deep):   K steps × M candidates/step, greedy best-reward.

Both strategies use the same total evaluation budget (N ≈ K × M).

Usage:
    # QED reward (default), 64 molecules
    python scripts/reasyn_random_baseline.py

    # Multi-objective with docking
    python scripts/reasyn_random_baseline.py --reward multi --target seh

    # Custom budget: 300 total candidates, 15 steps × 20 per step
    python scripts/reasyn_random_baseline.py --total_budget 300 --greedy_steps 15

    # More workers for speed
    python scripts/reasyn_random_baseline.py --num_workers 16
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED as QEDModule, AllChem, Descriptors

RDLogger.DisableLog('rdApp.*')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REASYN_ROOT = os.path.join(PROJECT, 'refs', 'ReaSyn')
sys.path.insert(0, PROJECT)
sys.path.insert(0, REASYN_ROOT)

# sklearn compat for fpindex pickle
try:
    import sklearn.metrics._dist_metrics as _dm
    if not hasattr(_dm, 'ManhattanDistance'):
        _dm.ManhattanDistance = _dm.ManhattanDistance64
except Exception:
    pass


# ---------------------------------------------------------------------------
# Model loading (reused from benchmark_reasyn_depth.py)
# ---------------------------------------------------------------------------

def load_models(model_dir, device="cuda", fp16=True):
    from rl.reasyn.models.reasyn import ReaSyn
    from omegaconf import OmegaConf

    model_files = [
        "nv-reasyn-ar-166m-v2.ckpt",
        "nv-reasyn-eb-174m-v2.ckpt",
    ]
    models = []
    reasyn_config = None
    for f in model_files:
        p = os.path.join(model_dir, f)
        ckpt = torch.load(p, map_location="cpu")
        reasyn_config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        m = ReaSyn(reasyn_config.model)
        m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        if fp16:
            m = m.half()
        m = m.to(device)
        m.eval()
        models.append(m)

    fpindex = pickle.load(
        open(os.path.join(REASYN_ROOT, reasyn_config.chem.fpindex), "rb"))
    rxn_matrix = pickle.load(
        open(os.path.join(REASYN_ROOT, reasyn_config.chem.rxn_matrix), "rb"))
    return models, fpindex, rxn_matrix


# ---------------------------------------------------------------------------
# ReaSyn sampler call
# ---------------------------------------------------------------------------

def run_sampler(smiles, models, fpindex, rxn_matrix, top_k=20,
                expansion_width=16, search_width=8, max_evolve_steps=4,
                num_cycles=1, timeout=30):
    """Run FastSampler on one molecule. Returns list of (smiles, score, synthesis)."""
    from rl.reasyn.chem.mol import Molecule
    from rl.reasyn.sampler.sampler_fast import FastSampler
    from rl.reasyn.utils.sample_utils import TimeLimit

    mol = Molecule(smiles)
    sampler = FastSampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix,
        mol=mol, model=models,
        factor=search_width,
        max_active_states=expansion_width,
        use_fp16=True,
        max_branch_states=8,
        skip_editflow=True,
        rxn_product_limit=1,
    )
    tl = TimeLimit(timeout)
    sampler.evolve(
        gpu_lock=None, time_limit=tl,
        max_evolve_steps=max_evolve_steps,
        num_cycles=num_cycles,
        num_editflow_samples=10,
        num_editflow_steps=50,
    )
    torch.cuda.synchronize()

    df = sampler.get_dataframe()
    if len(df) == 0:
        return []

    df = df.drop_duplicates(subset='smiles')
    df = df.sort_values('score', ascending=False).head(top_k)
    return list(zip(df['smiles'].tolist(), df['score'].tolist(),
                    df['synthesis'].tolist()))


# ---------------------------------------------------------------------------
# Reward computation (simplified from main.py)
# ---------------------------------------------------------------------------

_sascorer = None


def _load_sascorer():
    global _sascorer
    if _sascorer is None:
        from rdkit import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        _sascorer = sascorer
    return _sascorer


def compute_qed_reward(smiles, qed_weight=0.8, sa_weight=0.2):
    """Compute QED+SA reward (no discount)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'valid': False}
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)
    raw = qed_weight * qed + sa_weight * sa_norm
    return {'reward': raw, 'qed': qed, 'sa': sa, 'valid': True}


def compute_multi_reward(smiles, dock_scorer=None):
    """Product-strategy multi: dock_norm × QED × SA_norm."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'dock_score': 0.0,
                'valid': False}
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)
    dock_score = 0.0
    dock_norm = 0.0
    if dock_scorer is not None:
        scores = dock_scorer.batch_dock([smiles])
        dock_score = scores[0]
        dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
    raw = dock_norm * qed * sa_norm
    return {'reward': raw, 'qed': qed, 'sa': sa, 'dock_score': dock_score,
            'valid': True}


def batch_compute_reward(smiles_list, reward_mode='qed', dock_scorer=None):
    """Compute rewards for a batch of SMILES."""
    # Batch dock first if needed
    if reward_mode in ('multi', 'dock') and dock_scorer is not None:
        dock_scorer.batch_dock(smiles_list)

    results = []
    for smi in smiles_list:
        if reward_mode == 'qed':
            r = compute_qed_reward(smi)
        elif reward_mode in ('multi', 'dock'):
            r = compute_multi_reward(smi, dock_scorer=dock_scorer)
        else:
            r = compute_qed_reward(smi)
        r['smiles'] = smi
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Strategy 1: One-shot Wide
# ---------------------------------------------------------------------------

def strategy_oneshot(smiles, models, fpindex, rxn_matrix,
                     total_budget=200, reward_mode='qed', dock_scorer=None):
    """Generate ~total_budget candidates in one step, pick best reward.

    Uses wider ReaSyn search (expansion_width=64, 8 evolve steps × 2 cycles)
    to generate many candidates from a single starting molecule.
    """
    t0 = time.perf_counter()

    # Wider search to get more candidates
    scored = run_sampler(
        smiles, models, fpindex, rxn_matrix,
        top_k=total_budget,
        expansion_width=64,
        search_width=16,
        max_evolve_steps=8,
        num_cycles=2,
        timeout=60,
    )

    gen_time = time.perf_counter() - t0
    n_generated = len(scored)

    if not scored:
        return {
            'strategy': 'oneshot',
            'init_smiles': smiles,
            'best_smiles': smiles,
            'best_reward': 0.0,
            'best_qed': 0.0,
            'best_synthesis': '',
            'n_generated': 0,
            'n_evaluated': 0,
            'gen_time': gen_time,
            'total_time': gen_time,
            'path': [smiles],
        }

    # Evaluate rewards for all candidates
    t1 = time.perf_counter()
    candidate_smiles = [s for s, _, _ in scored]
    synthesis_map = {s: syn for s, _, syn in scored}
    reward_results = batch_compute_reward(candidate_smiles, reward_mode,
                                          dock_scorer)
    eval_time = time.perf_counter() - t1

    # Pick best
    best = max(reward_results, key=lambda r: r['reward'])
    total_time = time.perf_counter() - t0

    return {
        'strategy': 'oneshot',
        'init_smiles': smiles,
        'best_smiles': best['smiles'],
        'best_reward': best['reward'],
        'best_qed': best.get('qed', 0.0),
        'best_dock': best.get('dock_score', 0.0),
        'best_synthesis': synthesis_map.get(best['smiles'], ''),
        'n_generated': n_generated,
        'n_evaluated': len(reward_results),
        'gen_time': gen_time,
        'eval_time': eval_time,
        'total_time': total_time,
        'path': [smiles, best['smiles']],
        'all_rewards': [r['reward'] for r in reward_results],
    }


# ---------------------------------------------------------------------------
# Strategy 2: Greedy Deep
# ---------------------------------------------------------------------------

def strategy_greedy(smiles, models, fpindex, rxn_matrix,
                    num_steps=10, top_k=20, reward_mode='qed',
                    dock_scorer=None):
    """K steps × M candidates, greedy best-reward selection at each step."""
    t0 = time.perf_counter()
    current = smiles
    path = [smiles]
    step_details = []
    total_generated = 0
    total_evaluated = 0
    best_ever_reward = 0.0
    best_ever_smiles = smiles
    best_ever_qed = 0.0
    best_ever_dock = 0.0
    best_ever_synthesis = ''

    # Track cumulative synthesis
    synthesis_chain = []

    for step in range(num_steps):
        t_step = time.perf_counter()

        scored = run_sampler(
            current, models, fpindex, rxn_matrix,
            top_k=top_k,
            expansion_width=16,
            search_width=8,
            max_evolve_steps=4,
            num_cycles=1,
            timeout=30,
        )

        n_gen = len(scored)
        total_generated += n_gen

        if not scored:
            step_details.append({
                'step': step, 'n_cands': 0,
                'best_reward': 0.0, 'stayed': True,
                'time': time.perf_counter() - t_step,
            })
            continue

        # Evaluate
        candidate_smiles = [current]  # include staying option
        synthesis_map = {}
        for s, _, syn in scored:
            if s != current:
                candidate_smiles.append(s)
            if syn:
                synthesis_map[s] = syn

        reward_results = batch_compute_reward(candidate_smiles, reward_mode,
                                              dock_scorer)
        total_evaluated += len(reward_results)

        # Pick best
        best = max(reward_results, key=lambda r: r['reward'])
        stayed = (best['smiles'] == current)

        # Update best-ever tracking
        if best['reward'] > best_ever_reward:
            best_ever_reward = best['reward']
            best_ever_smiles = best['smiles']
            best_ever_qed = best.get('qed', 0.0)
            best_ever_dock = best.get('dock_score', 0.0)
            best_ever_synthesis = synthesis_map.get(best['smiles'], '')

        step_time = time.perf_counter() - t_step
        step_details.append({
            'step': step,
            'n_cands': len(candidate_smiles),
            'best_reward': best['reward'],
            'best_qed': best.get('qed', 0.0),
            'selected': best['smiles'][:50],
            'stayed': stayed,
            'time': step_time,
        })

        current = best['smiles']
        path.append(current)

    total_time = time.perf_counter() - t0

    return {
        'strategy': 'greedy',
        'init_smiles': smiles,
        'best_smiles': best_ever_smiles,
        'best_reward': best_ever_reward,
        'best_qed': best_ever_qed,
        'best_dock': best_ever_dock,
        'best_synthesis': best_ever_synthesis,
        'n_generated': total_generated,
        'n_evaluated': total_evaluated,
        'num_steps': num_steps,
        'total_time': total_time,
        'path': path,
        'step_details': step_details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare ReaSyn random search baselines')
    parser.add_argument('--reward', default='qed',
                        choices=['qed', 'multi', 'dock'])
    parser.add_argument('--target', default='seh',
                        help='Docking target (seh, drd2, gsk3b)')
    parser.add_argument('--scoring_method', default='proxy',
                        choices=['dock', 'proxy'],
                        help='Docking backend')
    parser.add_argument('--total_budget', type=int, default=200,
                        help='Total candidate evaluation budget')
    parser.add_argument('--greedy_steps', type=int, default=10,
                        help='Number of greedy steps (strategy 2)')
    parser.add_argument('--num_molecules', type=int, default=64,
                        help='Number of starting molecules')
    parser.add_argument('--mol_json', default=None,
                        help='JSON file with molecules')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    top_k_per_step = args.total_budget // args.greedy_steps

    print(f"\n{'='*70}")
    print("ReaSyn Random Baseline Comparison")
    print(f"  Reward: {args.reward}" +
          (f" (target={args.target})" if args.reward != 'qed' else ''))
    print(f"  Strategy 1 (One-shot): ~{args.total_budget} candidates, 1 step")
    print(f"  Strategy 2 (Greedy):   {args.greedy_steps} steps × "
          f"{top_k_per_step} candidates/step")
    print(f"{'='*70}\n")

    # Load models
    print("Loading ReaSyn models...", flush=True)
    model_dir = os.path.join(PROJECT, "refs", "ReaSyn", "data", "trained_model")
    models, fpindex, rxn_matrix = load_models(model_dir)
    print("  Models loaded.\n", flush=True)

    # Load dock scorer if needed
    dock_scorer = None
    if args.reward in ('multi', 'dock'):
        sys.path.insert(0, PROJECT)
        from reward.docking_score.factory import make_dock_scorer
        from omegaconf import OmegaConf
        reward_cfg = OmegaConf.create({
            'name': args.reward,
            'target': args.target,
            'scoring_method': args.scoring_method,
        })
        dock_scorer = make_dock_scorer(reward_cfg)
        print(f"  Dock scorer ready: {args.target} ({args.scoring_method})\n")

    # Load starting molecules
    if args.mol_json:
        with open(args.mol_json) as f:
            mol_data = json.load(f)
        all_smiles = [d['smiles'] for d in mol_data[:args.num_molecules]]
    else:
        zinc_path = os.path.join(
            PROJECT, "refs", "ReaSyn", "data", "zinc_first64.txt")
        with open(zinc_path) as f:
            lines = f.read().strip().split('\n')
        if lines[0].upper() == 'SMILES':
            lines = lines[1:]
        all_smiles = [s.strip() for s in lines[:args.num_molecules]]

    print(f"Starting molecules: {len(all_smiles)}\n")

    # Run both strategies
    oneshot_results = []
    greedy_results = []

    for i, smi in enumerate(all_smiles):
        short = smi[:40] + ('...' if len(smi) > 40 else '')
        print(f"[{i+1}/{len(all_smiles)}] {short}")

        # Strategy 1: One-shot wide
        r1 = strategy_oneshot(
            smi, models, fpindex, rxn_matrix,
            total_budget=args.total_budget,
            reward_mode=args.reward,
            dock_scorer=dock_scorer,
        )
        oneshot_results.append(r1)
        print(f"  One-shot: {r1['n_generated']} cands, "
              f"reward={r1['best_reward']:.4f}, "
              f"QED={r1['best_qed']:.3f}, "
              f"time={r1['total_time']:.1f}s")

        # Strategy 2: Greedy deep
        r2 = strategy_greedy(
            smi, models, fpindex, rxn_matrix,
            num_steps=args.greedy_steps,
            top_k=top_k_per_step,
            reward_mode=args.reward,
            dock_scorer=dock_scorer,
        )
        greedy_results.append(r2)
        print(f"  Greedy:   {r2['n_generated']} cands, "
              f"{r2['num_steps']} steps, "
              f"reward={r2['best_reward']:.4f}, "
              f"QED={r2['best_qed']:.3f}, "
              f"time={r2['total_time']:.1f}s")
        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def summarize(results, name):
        rewards = [r['best_reward'] for r in results]
        qeds = [r['best_qed'] for r in results]
        times = [r['total_time'] for r in results]
        n_gens = [r['n_generated'] for r in results]
        print(f"\n{name}:")
        print(f"  Reward: mean={np.mean(rewards):.4f}, "
              f"median={np.median(rewards):.4f}, "
              f"max={np.max(rewards):.4f}, "
              f"std={np.std(rewards):.4f}")
        print(f"  QED:    mean={np.mean(qeds):.3f}, "
              f"median={np.median(qeds):.3f}, "
              f"max={np.max(qeds):.3f}")
        print(f"  Cands:  mean={np.mean(n_gens):.0f}, "
              f"total={np.sum(n_gens):.0f}")
        print(f"  Time:   mean={np.mean(times):.1f}s, "
              f"total={np.sum(times):.0f}s")
        return rewards

    r1_rewards = summarize(oneshot_results,
                           f"Strategy 1 (One-shot ~{args.total_budget})")
    r2_rewards = summarize(greedy_results,
                           f"Strategy 2 (Greedy {args.greedy_steps}×"
                           f"{top_k_per_step})")

    # Head-to-head comparison
    print(f"\n{'─'*70}")
    print("Head-to-head (per molecule):")
    oneshot_wins = sum(1 for a, b in zip(r1_rewards, r2_rewards) if a > b)
    greedy_wins = sum(1 for a, b in zip(r1_rewards, r2_rewards) if b > a)
    ties = len(r1_rewards) - oneshot_wins - greedy_wins
    print(f"  One-shot wins: {oneshot_wins}, "
          f"Greedy wins: {greedy_wins}, "
          f"Ties: {ties}")

    delta = [b - a for a, b in zip(r1_rewards, r2_rewards)]
    print(f"  Greedy advantage: mean={np.mean(delta):+.4f}, "
          f"median={np.median(delta):+.4f}")

    # Top-10 molecules across both strategies
    print(f"\n{'─'*70}")
    print("Top-10 molecules (across both strategies):")
    all_results = (
        [(r, 'oneshot') for r in oneshot_results] +
        [(r, 'greedy') for r in greedy_results]
    )
    all_results.sort(key=lambda x: x[0]['best_reward'], reverse=True)
    seen = set()
    rank = 0
    for r, strat in all_results:
        smi = r['best_smiles']
        if smi in seen:
            continue
        seen.add(smi)
        rank += 1
        dock_str = (f", dock={r.get('best_dock', 0.0):.2f}"
                    if args.reward != 'qed' else '')
        print(f"  {rank:2d}. [{strat:7s}] reward={r['best_reward']:.4f}, "
              f"QED={r['best_qed']:.3f}{dock_str}")
        print(f"      {r['init_smiles'][:35]} -> {smi[:35]}")
        if rank >= 10:
            break

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            PROJECT, 'Experiments',
            f'reasyn_baseline_{args.reward}_{args.seed}.json')

    save_data = {
        'args': vars(args),
        'oneshot': oneshot_results,
        'greedy': greedy_results,
        'summary': {
            'oneshot_mean_reward': float(np.mean(r1_rewards)),
            'greedy_mean_reward': float(np.mean(r2_rewards)),
            'oneshot_wins': oneshot_wins,
            'greedy_wins': greedy_wins,
        },
    }
    # Remove non-serializable fields
    for r_list in [save_data['oneshot'], save_data['greedy']]:
        for r in r_list:
            r.pop('all_rewards', None)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
