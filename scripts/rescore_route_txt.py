"""Re-score route_multi_1 products with SA + Dock and regenerate txt.

Loads the saved pickle, computes SA scores from SMILES, runs UniDock
batch docking for all unique molecules, and writes a comprehensive
synthesis route txt with all reward components.

Usage (needs GPU):
    python scripts/rescore_route_txt.py
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rdkit import Chem, RDConfig
from rdkit.Chem import QED as QEDModule

# Load sascorer
sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if sa_path not in sys.path:
    sys.path.append(sa_path)
import sascorer


def compute_sa(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 10.0
    return sascorer.calculateScore(mol)


def compute_qed(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    return QEDModule.qed(mol)


def main():
    pickle_path = PROJECT_ROOT / 'Experiments' / 'route_multi_1_history.pickle'
    output_path = PROJECT_ROOT / 'Experiments' / 'route_multi_1_synthesis_routes.txt'

    print(f"Loading {pickle_path}")
    with open(pickle_path, 'rb') as f:
        history = pickle.load(f)

    cfg = history['config']
    reward_cfg = cfg['reward']

    # Collect all unique SMILES
    all_smiles = set()
    for p in history['best_products']:
        all_smiles.add(p['smiles'])
    for ri, top in history['route_top5'].items():
        for p in top:
            all_smiles.add(p['smiles'])
    for ep in history['last_episodes']:
        for r in ep['routes']:
            all_smiles.add(r['smiles'])

    all_smiles = list(all_smiles)
    print(f"Unique SMILES: {len(all_smiles)}")

    # Compute SA scores
    print("Computing SA scores...")
    t0 = time.perf_counter()
    sa_map = {}
    for smi in all_smiles:
        sa_map[smi] = compute_sa(smi)
    print(f"  SA computed for {len(sa_map)} molecules in "
          f"{time.perf_counter() - t0:.1f}s")

    # Compute dock scores via UniDock
    dock_map = {}
    target = reward_cfg.get('target')
    if target:
        print(f"Setting up UniDockScorer for target={target}...")
        target_dir = PROJECT_ROOT / 'Data' / 'docking_targets' / target
        config_path = target_dir / 'config.json'
        receptor_path = target_dir / 'receptor.pdbqt'

        if config_path.exists() and receptor_path.exists():
            with open(config_path) as f:
                tgt_cfg = json.load(f)

            from reward.docking_score.unidock import UniDockScorer
            scorer = UniDockScorer(
                receptor_pdbqt=str(receptor_path),
                center_x=tgt_cfg['center_x'],
                center_y=tgt_cfg['center_y'],
                center_z=tgt_cfg['center_z'],
                size_x=reward_cfg.get('size_x', 22.5),
                size_y=reward_cfg.get('size_y', 22.5),
                size_z=reward_cfg.get('size_z', 22.5),
                num_workers=16,
            )

            print(f"Batch docking {len(all_smiles)} molecules...")
            t0 = time.perf_counter()
            scores = scorer.batch_dock(all_smiles)
            for smi, score in zip(all_smiles, scores):
                dock_map[smi] = score
            print(f"  Docking done in {time.perf_counter() - t0:.1f}s")
            print(f"  {scorer.timing_summary}")
        else:
            print(f"  WARNING: target files not found, skipping docking")
    else:
        print("No dock target configured, skipping docking")

    # Generate txt
    print(f"\nWriting {output_path}")
    lines = []

    def L(s=''):
        lines.append(s)

    L('=' * 80)
    L(f'Route-DQN Multi-Objective ({target or "?"}) - Trial {cfg["trial"]}'
      f' - {cfg["episodes"]} Episodes')
    L(f'Method: Route-DQN with {cfg["method"]["decompose_method"]} decomposition')
    L(f'Reward: {reward_cfg["name"]} '
      f'(QED {reward_cfg.get("primary_weight", 0.6)} + '
      f'Dock {reward_cfg.get("dock_weight", 0.4)} + '
      f'SA {reward_cfg.get("sa_weight", 0.2)})')
    L('=' * 80)
    L()
    L(f'Episodes: {cfg["episodes"]}')
    L(f'Routes: {len(history["route_top5"])} '
      f'({cfg["method"]["decompose_method"]}), '
      f'{cfg["max_steps"]} steps/episode')
    L(f'Target: {target or "N/A"}')
    L()

    # Training summary
    L('-' * 80)
    L('TRAINING SUMMARY')
    L('-' * 80)
    rewards = history['rewards']
    qeds = history['qeds']
    sas = history['sas']
    docks = history.get('docks', [])
    L(f'  Final reward: {rewards[-1]:.4f}')
    L(f'  Max reward:   {max(rewards):.4f} '
      f'(ep {rewards.index(max(rewards)) + 1})')
    L(f'  Final QED:    {qeds[-1]:.4f}')
    L(f'  Max QED:      {max(qeds):.4f}')
    L(f'  Final SA:     {sas[-1]:.4f}')
    if docks:
        L(f'  Final Dock:   {docks[-1]:.2f}')
        L(f'  Best Dock:    {min(docks):.2f} '
          f'(ep {docks.index(min(docks)) + 1})')
    L()

    # Top-20 best products
    L('-' * 80)
    L('TOP-20 BEST PRODUCTS (by QED)')
    L('-' * 80)
    for i, p in enumerate(history['best_products'][:20]):
        smi = p['smiles']
        sa = sa_map.get(smi, p.get('sa'))
        dock = dock_map.get(smi, p.get('dock'))
        qed = p['qed']

        sa_str = f'  SA={sa:.2f}' if sa is not None else ''
        dock_str = f'  Dock={dock:.1f}' if dock is not None and dock != 0.0 else ''
        L(f'  #{i+1:2d}  QED={qed:.3f}{sa_str}{dock_str}'
          f'  ep={p["episode"]:3d}  route={p["route_idx"]}  steps={p["n_steps"]}')
        L(f'       Init:    {p["init_mol"]}')
        L(f'       Product: {smi}')
    L()

    # Per-route top-5
    L('=' * 80)
    L(f'PER-ROUTE TOP-5 PRODUCTS ({len(history["route_top5"])} routes)')
    L('=' * 80)
    L()

    for ri in sorted(history['route_top5'].keys()):
        top = history['route_top5'][ri]
        if not top:
            continue
        init_mol = top[0].get('init_mol', '?')
        L(f'--- Route {ri} (init: {init_mol}) ---')
        for j, p in enumerate(top):
            smi = p['smiles']
            sa = sa_map.get(smi, p.get('sa'))
            dock = dock_map.get(smi, p.get('dock'))
            qed = p['qed']

            sa_str = f'  SA={sa:.2f}' if sa is not None else ''
            dock_str = (f'  Dock={dock:.1f}'
                        if dock is not None and dock != 0.0 else '')
            L(f'  #{j+1} QED={qed:.3f}{sa_str}{dock_str}'
              f' ep={p["episode"]:3d}'
              f' | {smi}')

            if 'steps' in p:
                steps = p['steps']
                L(f'     Synthesis ({len(steps)} steps):')
                for si, s in enumerate(steps):
                    rxn_type = 'uni' if s['is_uni'] else 'bi'
                    t_idx = s['template_idx']
                    bb_smi = s['block_smi'] or ''
                    prod = s['intermediate_smi'] or '?'
                    if s['is_uni']:
                        L(f'       Step {si}: [{rxn_type}] T{t_idx}: '
                          f'{bb_smi} -> {prod}')
                    else:
                        L(f'       Step {si}: [{rxn_type}] T{t_idx} '
                          f'+ {bb_smi} -> {prod}')
        L()

    # Last episodes
    L('=' * 80)
    L(f'LAST {len(history["last_episodes"])} EPISODES (full route snapshots)')
    L('=' * 80)
    L()

    for ep_data in history['last_episodes']:
        ep_num = ep_data['episode']
        ep_routes = ep_data['routes']
        n_success = sum(1 for r in ep_routes
                        if r.get('smiles') and r['qed'] > 0)
        L(f'Episode {ep_num + 1} ({len(ep_routes)} routes, '
          f'{n_success} successful)')

        # Show top 5 by QED from this episode
        sorted_routes = sorted(ep_routes, key=lambda x: x['qed'],
                                reverse=True)[:5]
        for j, r in enumerate(sorted_routes):
            smi = r['smiles']
            sa = sa_map.get(smi, r.get('sa'))
            dock = dock_map.get(smi, r.get('dock'))
            qed = r['qed']

            sa_str = f'  SA={sa:.2f}' if sa is not None else ''
            dock_str = (f'  Dock={dock:.1f}'
                        if dock is not None and dock != 0.0 else '')
            L(f'  Top {j+1}: QED={qed:.3f}{sa_str}{dock_str}'
              f' route={r["route_idx"]} | {smi}')
        L()

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nDone! {len(lines)} lines written to {output_path}")


if __name__ == '__main__':
    main()
