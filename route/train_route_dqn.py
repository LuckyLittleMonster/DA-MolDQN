#!/usr/bin/env python
"""Training script for Route-DQN synthesis route optimization.

Usage:
    # Smoke test (fast)
    python route/train_route_dqn.py --episodes 20 --num_molecules 8 --max_steps 3

    # Full training
    python route/train_route_dqn.py --episodes 200 --num_molecules 64 --max_steps 5 --device cuda
"""

import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train Route-DQN for synthesis route optimization')

    # Experiment
    parser.add_argument('--experiment', type=str, default='route_dqn')
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    # Training
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=5,
                        help="BB-swap steps per episode")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--polyak', type=float, default=0.995)

    # Epsilon
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_decay', type=float, default=0.995)
    parser.add_argument('--eps_min', type=float, default=0.05)

    # Route generation
    parser.add_argument('--init_mol', type=str, nargs='+',
                        default=['c1ccc(N)cc1', 'c1ccc(O)cc1', 'CCO',
                                 'CC(=O)O', 'c1ccccc1', 'CC(C)N',
                                 'c1ccc(C=O)cc1', 'CCCC'])
    parser.add_argument('--init_mol_path', type=str, default=None)
    parser.add_argument('--num_molecules', type=int, default=None)
    parser.add_argument('--route_steps', type=int, default=3,
                        help="Number of steps in generated routes")
    parser.add_argument('--routes_per_mol', type=int, default=1,
                        help="Routes to generate per init mol")
    parser.add_argument('--decompose_method', type=str, default='random',
                        choices=['random', 'dfs', 'retro', 'aizynth', 'paroutes'],
                        help="Route generation method")
    parser.add_argument('--aizynth_path', type=str, default=None,
                        help="Path to AiZynthFinder results JSON")
    parser.add_argument('--paroutes_path', type=str, default=None,
                        help="Path to PaRoutes annotated JSON "
                             "(default: Data/paroutes/n1_routes_matched.json)")
    parser.add_argument('--bb_library', type=str, default=None,
                        help="Override building block library path")

    # Reward
    parser.add_argument('--qed_weight', type=float, default=0.8)
    parser.add_argument('--sa_weight', type=float, default=0.2)
    parser.add_argument('--discount', type=float, default=0.9)

    # Model
    parser.add_argument('--fp_dim', type=int, default=4096)
    parser.add_argument('--template_emb_dim', type=int, default=128)
    parser.add_argument('--block_emb_dim', type=int, default=64)
    parser.add_argument('--route_emb_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--max_route_len', type=int, default=10)

    # Infrastructure
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cascade_workers', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=5)
    parser.add_argument('--load_checkpoint', type=str, default=None)

    return parser.parse_args(argv)


def _route_to_dict(route, episode: int, route_idx: int) -> dict | None:
    """Serialize a SynthesisRoute to a picklable dict with full path info."""
    mol = Chem.MolFromSmiles(route.final_product_smi)
    if mol is None:
        return None
    qed_val = QED.qed(mol)
    return {
        'smiles': route.final_product_smi,
        'qed': qed_val,
        'episode': episode,
        'route_idx': route_idx,
        'init_mol': route.init_mol_smi,
        'n_steps': len(route),
        'steps': [
            {
                'template_idx': s.template_idx,
                'bi_rxn_idx': s.bi_rxn_idx,
                'block_idx': s.block_idx,
                'block_smi': s.block_smi,
                'intermediate_smi': s.intermediate_smi,
                'is_uni': s.is_uni,
            }
            for s in route.steps
        ],
    }


def train(args):
    """Main training loop."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. Load template predictor (BEFORE CUDA / mp pool) ---
    print("\n--- Loading template predictor ---")
    from template.template_predictor import TemplateReactionPredictor
    from pathlib import Path
    _data_dir = Path(__file__).resolve().parent.parent / 'template' / 'data'
    if args.decompose_method == 'aizynth':
        bb_path = (args.bb_library
                   or str(_data_dir / 'building_blocks_merged.smi'))
        tp = TemplateReactionPredictor(
            template_path=str(_data_dir / 'templates_aizynth.txt'),
            building_block_path=bb_path,
            num_workers=0,
        )
    elif args.decompose_method == 'paroutes':
        bb_path = (args.bb_library
                   or str(_data_dir / 'building_blocks_paroutes.smi'))
        tp = TemplateReactionPredictor(
            template_path=str(_data_dir / 'templates_paroutes.txt'),
            building_block_path=bb_path,
            num_workers=0,
        )
    else:
        if args.bb_library:
            tp = TemplateReactionPredictor(
                building_block_path=args.bb_library, num_workers=0)
        else:
            tp = TemplateReactionPredictor(num_workers=0)
    tp.load()

    # Pre-build uni lookup cache (before forking workers)
    tp._uni_by_template_idx = {r.index: r for r in tp.uni_reactions}

    # --- 2. Create multiprocessing pool (BEFORE CUDA init) ---
    mp_pool = None
    if args.cascade_workers > 0:
        import multiprocessing as mp
        import route.route as route_module
        route_module._mp_tp = tp
        ctx = mp.get_context('fork')
        mp_pool = ctx.Pool(args.cascade_workers)
        print(f"  MP pool: {args.cascade_workers} workers (fork)")

    # --- 3. Resolve device (CUDA init happens later on first .to()) ---
    device = args.device
    if device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif not torch.cuda.is_available():
        device = 'cpu'
    print(f"Device: {device}")

    # --- 4. Load initial molecules ---
    if args.init_mol_path:
        with open(args.init_mol_path) as f:
            init_mols = [line.strip() for line in f if line.strip()]
    else:
        init_mols = args.init_mol

    if args.num_molecules is not None:
        if args.num_molecules <= len(init_mols):
            init_mols = random.sample(init_mols, args.num_molecules)
        else:
            expanded = [init_mols[i % len(init_mols)]
                        for i in range(args.num_molecules)]
            init_mols = expanded

    print(f"\nInit molecules ({len(init_mols)}): {init_mols[:8]}"
          f"{'...' if len(init_mols) > 8 else ''}")

    # --- 5. Generate initial routes ---
    print("\n--- Generating initial routes ---")
    from route.retro_decompose import (
        build_random_routes, decompose_molecule, build_retro_routes,
        build_routes_from_aizynth, build_routes_from_paroutes)

    t0 = time.perf_counter()
    all_routes = []

    if args.decompose_method == 'aizynth':
        import json
        aizynth_path = args.aizynth_path
        if aizynth_path is None:
            aizynth_path = 'Experiments/aizynth_zinc128_results.json'
        with open(aizynth_path) as f:
            aizynth_data = json.load(f)
        # Filter to solved entries and respect num_molecules limit
        solved = [d for d in aizynth_data if d.get('is_solved', False)]
        n_mol = args.num_molecules or len(solved)
        solved = solved[:n_mol]
        print(f"  AiZynth: {len(solved)} solved molecules from {aizynth_path}")
        all_routes = build_routes_from_aizynth(solved, tp)
    elif args.decompose_method == 'paroutes':
        import json as _json
        paroutes_path = args.paroutes_path
        if paroutes_path is None:
            paroutes_path = 'Data/paroutes/n1_routes_matched.json'
        with open(paroutes_path) as f:
            paroutes_data = _json.load(f)
        n_mol = args.num_molecules or len(paroutes_data)
        paroutes_data = paroutes_data[:n_mol]
        print(f"  PaRoutes: {len(paroutes_data)} routes from {paroutes_path}")
        all_routes = build_routes_from_paroutes(paroutes_data, tp)
    else:
        for mol_smi in init_mols:
            if args.decompose_method == 'random':
                routes = build_random_routes(
                    mol_smi, tp,
                    n_routes=args.routes_per_mol,
                    n_steps=args.route_steps,
                    seed=args.seed,
                )
            elif args.decompose_method == 'retro':
                routes = build_retro_routes(
                    mol_smi, tp,
                    max_depth=args.route_steps,
                    max_routes=args.routes_per_mol,
                    min_steps=2,
                    seed=args.seed,
                    verify=True,
                )
            else:
                routes = decompose_molecule(
                    mol_smi, tp,
                    max_depth=args.route_steps,
                    max_routes=args.routes_per_mol,
                )
            all_routes.extend(routes)

    elapsed = time.perf_counter() - t0

    if not all_routes:
        print("ERROR: No valid routes generated! Try different molecules.")
        return

    print(f"Generated {len(all_routes)} routes in {elapsed:.1f}s")
    for i, route in enumerate(all_routes[:5]):
        print(f"  Route {i}: {len(route)} steps, "
              f"{route.n_modifiable} modifiable, "
              f"init={route.init_mol_smi}, "
              f"final={route.final_product_smi}")

    # --- 6. Create environment and DQN ---
    print("\n--- Creating environment and DQN ---")
    from route.route_environment import RouteEnvironment
    from route.route_dqn import RouteDQN

    env = RouteEnvironment(
        routes=all_routes,
        tp=tp,
        max_steps=args.max_steps,
        discount=args.discount,
        qed_weight=args.qed_weight,
        sa_weight=args.sa_weight,
    )

    dqn = RouteDQN(
        tp=tp,
        device=device,
        fp_dim=args.fp_dim,
        template_emb_dim=args.template_emb_dim,
        block_emb_dim=args.block_emb_dim,
        route_emb_dim=args.route_emb_dim,
        max_route_len=args.max_route_len,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        polyak=args.polyak,
        replay_size=args.replay_size,
        cascade_workers=args.cascade_workers,
        mp_pool=mp_pool,
    )

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        dqn.load_checkpoint(args.load_checkpoint)

    # --- 7. Training loop ---
    print(f"\n{'='*60}")
    print(f"Training: {args.episodes} episodes, {env.n_routes} routes, "
          f"{args.max_steps} steps/ep")
    print(f"{'='*60}\n")

    eps = args.eps_start
    best_mean_reward = -float('inf')
    all_rewards = []
    all_qeds = []
    all_sas = []
    best_products = []  # track best products seen (flat list, trimmed to top-20)

    # Per-route top-5 products with full synthesis paths
    route_top5: dict[int, list[dict]] = {}  # route_idx -> sorted list of top-5

    # Last N episodes' full route snapshots
    last_n_episodes = 5
    last_episodes_buf: list[dict] = []  # circular buffer of last N episodes

    # Timing accumulators (reset every log_freq episodes)
    phase_times = {
        'encode': 0.0, 'q_pos': 0.0, 'cascade': 0.0, 'q_bb': 0.0,
        'env_step': 0.0, 'next_encode': 0.0, 'store': 0.0, 'train': 0.0,
    }
    timing_steps = 0

    os.makedirs('Experiments/models', exist_ok=True)

    for episode in range(args.episodes):
        ep_start = time.perf_counter()

        routes = env.reset()
        episode_rewards = [0.0] * env.n_routes
        episode_qeds_final = [0.0] * env.n_routes

        for step in range(args.max_steps):
            # --- Batch action selection (1 encode + 1 Q_pos + parallel cascade + 1 Q_bb) ---
            batch_result = dqn.act_batch(routes, eps)
            positions = batch_result['positions']
            block_indices = batch_result['block_indices']
            valid_blocks_list = batch_result['valid_blocks_list']
            route_states_np = batch_result['route_states']
            pos_masks_np = batch_result['position_masks']
            pos_fps = batch_result['pos_fps']
            template_indices = batch_result['template_indices']

            # Accumulate act_batch timings
            for k, v in batch_result['timings'].items():
                phase_times[k] += v

            # --- Execute environment step ---
            t0 = time.perf_counter()
            result = env.step(positions, block_indices)
            phase_times['env_step'] += time.perf_counter() - t0

            # --- Batch encode next states (1 GPU forward) ---
            t0 = time.perf_counter()
            next_route_states = dqn.encode_routes(routes).cpu().numpy()
            phase_times['next_encode'] += time.perf_counter() - t0

            # --- Store transitions ---
            t0 = time.perf_counter()
            for i in range(len(routes)):
                next_pos_mask = np.zeros(args.max_route_len, dtype=bool)
                for p_idx, m in enumerate(routes[i].modifiable_mask):
                    if p_idx < args.max_route_len and m:
                        next_pos_mask[p_idx] = True

                transition = {
                    'route_state': route_states_np[i],
                    'position': positions[i],
                    'position_mask': pos_masks_np[i],
                    'block_idx': block_indices[i],
                    'block_mask': np.zeros(1, dtype=bool),  # unused in train_step
                    'position_fp': pos_fps[i],
                    'template_idx': template_indices[i],
                    'reward': result['rewards'][i],
                    'next_route_state': next_route_states[i],
                    'next_position_mask': next_pos_mask,
                    'done': result['done'],
                }
                dqn.store_transition(transition)

                episode_rewards[i] += result['rewards'][i]
                episode_qeds_final[i] = result['QED'][i]
            phase_times['store'] += time.perf_counter() - t0

            # --- DQN gradient steps ---
            t0 = time.perf_counter()
            n_updates = max(1, env.n_routes // args.batch_size)
            for _ in range(n_updates):
                dqn.train_step(args.batch_size)
            phase_times['train'] += time.perf_counter() - t0

            timing_steps += 1

        # Episode stats
        mean_reward = np.mean(episode_rewards)
        mean_qed = np.mean(episode_qeds_final)
        max_qed = max(episode_qeds_final)
        mean_sa = np.mean(result['SA_score'])

        all_rewards.append(mean_reward)
        all_qeds.append(mean_qed)
        all_sas.append(mean_sa)

        # Track best products (flat list for top-20) + per-route top-5
        ep_route_snapshots = []
        for i, route in enumerate(routes):
            rd = _route_to_dict(route, episode, i)
            if rd is None:
                continue
            qed_val = rd['qed']
            best_products.append({
                'smiles': route.final_product_smi,
                'qed': qed_val,
                'episode': episode,
                'route_idx': i,
                'init_mol': route.init_mol_smi,
                'n_steps': len(route),
            })
            ep_route_snapshots.append(rd)

            # Per-route top-5 with full path
            if i not in route_top5:
                route_top5[i] = []
            top_list = route_top5[i]
            # Insert if better than worst in top-5
            if len(top_list) < 5 or qed_val > top_list[-1]['qed']:
                # Deduplicate by smiles
                existing = {d['smiles'] for d in top_list}
                if rd['smiles'] not in existing:
                    top_list.append(rd)
                    top_list.sort(key=lambda x: x['qed'], reverse=True)
                    route_top5[i] = top_list[:5]

        # Circular buffer: keep last N episodes' full route snapshots
        last_episodes_buf.append({
            'episode': episode,
            'routes': ep_route_snapshots,
        })
        if len(last_episodes_buf) > last_n_episodes:
            last_episodes_buf.pop(0)

        # Epsilon decay
        eps = max(args.eps_min, eps * args.eps_decay)

        # Logging
        ep_time = time.perf_counter() - ep_start

        if (episode + 1) % args.log_freq == 0:
            n_success = sum(result['success'])
            print(f"Ep {episode+1:4d}/{args.episodes} | "
                  f"R: {mean_reward:.4f} | "
                  f"QED: {mean_qed:.3f} (max {max_qed:.3f}) | "
                  f"SA: {mean_sa:.2f} | "
                  f"Eps: {eps:.3f} | "
                  f"Success: {n_success}/{env.n_routes} | "
                  f"Buf: {len(dqn.replay_buffer)} | "
                  f"Time: {ep_time:.1f}s")

            # Show a couple of routes
            for i in range(min(2, env.n_routes)):
                route = routes[i]
                print(f"  Route {i}: {route.init_mol_smi} -> "
                      f"{route.final_product_smi} "
                      f"(QED={episode_qeds_final[i]:.3f})")

            # Timing breakdown (per step average over recent episodes)
            if timing_steps > 0:
                avg_ms = {k: v / timing_steps * 1000
                          for k, v in phase_times.items()}
                total_ms = sum(avg_ms.values())
                print(f"  Timing/step ({timing_steps} steps): "
                      f"encode={avg_ms['encode']:.1f}ms "
                      f"Q_pos={avg_ms['q_pos']:.1f}ms "
                      f"cascade={avg_ms['cascade']:.1f}ms "
                      f"Q_bb={avg_ms['q_bb']:.1f}ms "
                      f"env={avg_ms['env_step']:.1f}ms "
                      f"next_enc={avg_ms['next_encode']:.1f}ms "
                      f"store={avg_ms['store']:.1f}ms "
                      f"train={avg_ms['train']:.1f}ms "
                      f"total={total_ms:.1f}ms")
                # Reset accumulators
                phase_times = {k: 0.0 for k in phase_times}
                timing_steps = 0

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0 or mean_reward > best_mean_reward:
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

            ckpt_path = (f'Experiments/models/{args.experiment}_'
                         f'{args.trial}_checkpoint.pth')
            dqn.save(ckpt_path)

            # Save history
            best_products_sorted = sorted(
                best_products, key=lambda x: x['qed'], reverse=True)[:20]

            with open(f'Experiments/{args.experiment}_{args.trial}'
                      f'_history.pickle', 'wb') as f:
                pickle.dump({
                    'rewards': all_rewards,
                    'qeds': all_qeds,
                    'sas': all_sas,
                    'best_products': best_products_sorted,
                    'route_top5': dict(route_top5),
                    'last_episodes': list(last_episodes_buf),
                    'args': vars(args),
                }, f)

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"Training complete! Best mean reward: {best_mean_reward:.4f}")
    print(f"{'='*60}")

    # Top products
    best_products_sorted = sorted(
        best_products, key=lambda x: x['qed'], reverse=True)[:10]
    print("\nTop 10 products by QED:")
    for i, p in enumerate(best_products_sorted):
        print(f"  {i+1}. QED={p['qed']:.3f} | {p['smiles']} "
              f"(from {p['init_mol']}, ep {p['episode']})")

    # Final routes
    print("\nFinal routes:")
    for i, route in enumerate(routes[:5]):
        mol = Chem.MolFromSmiles(route.final_product_smi)
        qed_val = QED.qed(mol) if mol else 0
        print(f"  Route {i}: {route.init_mol_smi} -> "
              f"{route.final_product_smi} (QED={qed_val:.3f})")
        for j, step in enumerate(route.steps):
            rxn_type = "uni" if step.is_uni else "bi"
            bb_str = f" + {step.block_smi}" if step.block_smi else ""
            print(f"    Step {j}: [{rxn_type}] T{step.template_idx}"
                  f"{bb_str} → {step.intermediate_smi}")

    # Per-route top-5 summary
    print(f"\nPer-route top-5 products ({len(route_top5)} routes):")
    for ri in sorted(route_top5.keys())[:10]:
        tops = route_top5[ri]
        if tops:
            print(f"  Route {ri} ({tops[0]['init_mol']}):")
            for j, t in enumerate(tops):
                path_str = " → ".join(
                    s['intermediate_smi'][:40] for s in t['steps'])
                print(f"    {j+1}. QED={t['qed']:.3f} ep{t['episode']:4d} | "
                      f"{t['smiles'][:60]}")
                print(f"       Path: {path_str[:120]}")

    dqn.close()
    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
    return dqn, best_mean_reward


if __name__ == '__main__':
    args = parse_args()
    train(args)
