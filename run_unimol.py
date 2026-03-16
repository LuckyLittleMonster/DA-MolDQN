"""Training script for UniMol-DQN vs MolDQN baseline on QED optimization.

Usage:
    python run_unimol.py --model unimol --device cuda --iterations 200000
    python run_unimol.py --model moldqn --device cuda --iterations 200000
"""

import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.*")

import src.cenv as cenv
from unimol_dqn import UniMolDQN, UniMolEncoder, EmbeddingCache, MolDQNBaseline
from agent_unimol import DQNAgent
from conformer3d import ConformerManager


def parse_args():
    parser = argparse.ArgumentParser(description='UniMol-DQN vs MolDQN on QED')
    parser.add_argument('--model', type=str, default='unimol', choices=['unimol', 'moldqn'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--init_mol', type=str, default='C')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--replay_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=2000)
    parser.add_argument('--fp_radius', type=int, default=3)
    parser.add_argument('--fp_length', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='Experiments/unimol_dqn')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Path to embedding cache (UniMol only)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (0=disabled)')
    parser.add_argument('--double_dqn', action='store_true', default=True,
                        help='Use Double DQN (online selects, target evaluates)')
    return parser.parse_args()


def setup_cenv(fp_radius=3, fp_length=2048):
    """Initialize cenv environment."""
    atom_types = ["C", "O", "N"]
    ring_sizes = [3, 5, 6]
    flags = cenv.Flags()
    return cenv.Environment(atom_types, ring_sizes, fp_radius, fp_length, flags)


def to_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Output directory
    exp_name = f"{args.model}_qed_s{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and agent
    if args.model == 'unimol':
        cache_path = args.cache_path or os.path.join(output_dir, 'embedding_cache.npz')
        encoder = UniMolEncoder(use_gpu=(device == 'cuda'))
        cache = EmbeddingCache(cache_path)
        model = UniMolDQN(encoder, cache)
        model_type = 'unimol'
        print(f"UniMolDQN: encoder={encoder.dim}d, Q-head on {device}")
    else:
        model = MolDQNBaseline(input_length=args.fp_length + 1)
        model_type = 'moldqn'
        print(f"MolDQN baseline: input={args.fp_length + 1}")

    agent = DQNAgent(
        model, model_type=model_type, device=device,
        lr=args.lr, gamma=args.gamma, polyak=args.polyak,
        replay_size=args.replay_size, max_batch_size=args.batch_size,
        fp_radius=args.fp_radius, fp_length=args.fp_length,
        max_grad_norm=args.max_grad_norm, double_dqn=args.double_dqn
    )

    # Initialize cenv
    env = setup_cenv(args.fp_radius, args.fp_length)

    # Tracking
    max_episodes = args.iterations // args.max_steps
    top_k = 10
    top_molecules = []  # min-heap of (qed, smiles)
    episode_rewards = []
    episode_best_qeds = []
    losses = []

    init_mol = Chem.MolFromSmiles(args.init_mol)

    # 3D conformer manager (for UniMol mode)
    conf_mgr = None
    if model_type == 'unimol':
        conf_mgr = ConformerManager(atom_threshold=10)

    print(f"\nTraining: {args.model} on QED optimization")
    print(f"  iterations={args.iterations}, max_steps={args.max_steps}, episodes={max_episodes}")
    print(f"  device={device}, init_mol={args.init_mol}")
    if conf_mgr:
        print(f"  3D filter: distance({conf_mgr.distance_factor}x) + angle + steric")
        print(f"  ConstrainedEmbed threshold: {conf_mgr.atom_threshold} atoms")
    print(f"  output={output_dir}\n")

    global_step = 0
    t_start = time.time()
    filter_stats_total = {'total': 0, 'kept': 0,
                          'dist_reject': 0, 'angle_reject': 0, 'steric_reject': 0}

    for episode in range(max_episodes):
        state = Chem.RWMol(init_mol)
        episode_reward = 0.0
        episode_best_qed = 0.0

        # Initialize parent conformer for this episode
        if conf_mgr:
            conf_mgr.set_parent(state)

        for step in range(args.max_steps):
            done = (step == args.max_steps - 1)
            step_fraction = step / args.max_steps
            epsilon = agent.get_epsilon(
                global_step, args.eps_start, args.eps_end, args.eps_decay
            )

            # Get valid actions from cenv (Mol objects, last = no-modification)
            valid_actions, _ = env.get_valid_actions_and_fingerprint(
                state, 0, False
            )

            if len(valid_actions) == 0:
                break

            # 3D action filtering (UniMol only)
            conformers = None
            if conf_mgr and model_type == 'unimol':
                valid_actions, kept_idx, stats = conf_mgr.filter_actions_3d(
                    valid_actions
                )
                for k in filter_stats_total:
                    filter_stats_total[k] += stats.get(k, 0)

                # Generate conformers for filtered actions
                conformers = [
                    conf_mgr.generate_conformer(m) for m in valid_actions
                ]

            if len(valid_actions) == 0:
                break

            # Reward = QED of current state
            try:
                reward = QED.qed(state)
            except Exception:
                reward = 0.0

            # Agent selects action
            idx, is_greedy, action_data = agent.get_action(
                valid_actions, step_fraction, epsilon, conformers=conformers
            )

            # Store transition (skip step 0, same as DA-MolDQN)
            if step > 0:
                agent.store_transition(reward, done, action_data, step_fraction)

            # Training step
            loss = agent.training_step()
            if loss is not None:
                losses.append(loss)

            # Track best QED in this episode
            if reward > episode_best_qed:
                episode_best_qed = reward

            # Update top-K molecules
            state_smi = Chem.MolToSmiles(state)
            if len(top_molecules) < top_k:
                heapq.heappush(top_molecules, (reward, state_smi))
            elif reward > top_molecules[0][0]:
                heapq.heapreplace(top_molecules, (reward, state_smi))

            episode_reward += reward

            # Transition to next state and update parent conformer
            selected_mol = valid_actions[idx]
            selected_conf = conformers[idx] if conformers else None
            state = Chem.RWMol(selected_mol)

            if conf_mgr:
                conf_mgr.update_parent_from_action(selected_mol, selected_conf)

            global_step += 1

            if global_step >= args.iterations:
                break

        episode_rewards.append(episode_reward)
        episode_best_qeds.append(episode_best_qed)

        # Logging every 10 episodes
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - t_start
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            top_qed = max(q for q, _ in top_molecules) if top_molecules else 0.0
            avg_top10 = np.mean(sorted([q for q, _ in top_molecules], reverse=True)[:10])

            extra = ""
            if model_type == 'unimol':
                filt_pct = (1 - filter_stats_total['kept'] / max(1, filter_stats_total['total'])) * 100
                extra = (f" cache={cache.size} hit={cache.hit_rate:.1%}"
                         f" 3d_filt={filt_pct:.0f}%")

            print(
                f"Ep {episode+1:5d} | step {global_step:7d} | eps {epsilon:.3f} | "
                f"ep_best {episode_best_qed:.4f} | top1 {top_qed:.4f} | "
                f"top10 {avg_top10:.4f} | loss {avg_loss:.4f}{extra} | "
                f"{elapsed:.0f}s"
            )

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            agent.save(os.path.join(output_dir, f'checkpoint_ep{episode+1}.pt'))
            ckpt_results = {
                'episode': episode + 1,
                'global_step': global_step,
                'top_molecules': sorted(top_molecules, reverse=True),
                'args': vars(args),
            }
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                json.dump(ckpt_results, f, indent=2, default=to_json)

            if model_type == 'unimol':
                cache.save()

        if global_step >= args.iterations:
            break

    # Final save
    agent.save(os.path.join(output_dir, 'final_model.pt'))

    elapsed = time.time() - t_start
    top_sorted = sorted(top_molecules, reverse=True)

    print(f"\n=== Training Complete ===")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Steps: {global_step}")
    print(f"Top QED: {top_sorted[0][0]:.4f}")
    print(f"Top-10 avg: {np.mean([q for q, _ in top_sorted[:10]]):.4f}")
    print(f"\nTop molecules:")
    for qed_val, smi in top_sorted:
        print(f"  {qed_val:.4f}  {smi}")

    final_results = {
        'model': args.model,
        'total_time_s': elapsed,
        'total_steps': global_step,
        'top_molecules': top_sorted,
        'episode_rewards': episode_rewards,
        'episode_best_qeds': episode_best_qeds,
        'losses': losses[-1000:],
        'args': vars(args),
    }
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=to_json)

    if model_type == 'unimol':
        cache.save()
        print(f"Embedding cache: {cache.size} entries, hit rate {cache.hit_rate:.1%}")


if __name__ == '__main__':
    main()
