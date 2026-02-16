#!/usr/bin/env python
"""
Training script for molecular optimization with reaction-based actions.

Uses modular components:
- environment.py: BaseEnvironment / QEDEnvironment
- dqn.py: MLPDQN / BaseDQN / create_dqn
- replay_buffer.py: SimpleReplayBuffer
"""

import argparse
import os
import copy
import heapq
import hyp
import math
import numpy as np
import pickle
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as opt
from queue import PriorityQueue
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

from replay_buffer import SimpleReplayBuffer
from environment import QEDEnvironment, BaseEnvironment
from dqn import MLPDQN, DQN, BaseDQN, create_dqn

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


def load_init_mols(path):
    """Load initial molecules from file."""
    with open(path, 'r') as f:
        mols = [line.strip() for line in f if line.strip()]
    return mols


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description='Train MolDQN with reaction-based actions')

    parser.add_argument('--experiment', type=str, default="qed_hypergraph",
                        help="experiment name")
    parser.add_argument('--trial', type=int, default=1,
                        help="experiment trial number")
    parser.add_argument('--iteration', type=int, default=10000,
                        help="number of iterations")
    parser.add_argument('--max_steps_per_episode', type=int, default=5,
                        help="maximum steps per episode")

    parser.add_argument('--qed_weight', type=float, default=0.8,
                        help="QED weight in reward")
    parser.add_argument('--sa_weight', type=float, default=0.2,
                        help="SA score weight in reward")
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help="discount factor")

    parser.add_argument('--eps_start', type=float, default=1.0,
                        help="initial epsilon")
    parser.add_argument('--eps_decay', type=float, default=0.995,
                        help="epsilon decay rate")
    parser.add_argument('--eps_min', type=float, default=0.01,
                        help="minimum epsilon")

    parser.add_argument('--lr', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--replay_buffer_size', type=int, default=50000,
                        help="replay buffer size")

    parser.add_argument('--init_mol', type=str, nargs='+',
                        default=['CCO', 'c1ccccc1O', 'CC(=O)O'],
                        help="initial molecules (SMILES)")
    parser.add_argument('--init_mol_path', type=str, default=None,
                        help="path to file with initial molecules")

    parser.add_argument('--use_hypergraph', action='store_true', default=True,
                        help="use hypergraph actions")
    parser.add_argument('--hypergraph_top_k', type=int, default=20,
                        help="top k hypergraph actions")
    parser.add_argument('--reactant_method', type=str, default='hypergraph',
                        choices=['hypergraph', 'fingerprint', 'legacy', 'aio',
                                 'hybrid', 'template', 'template_2model'],
                        help="reactant prediction method")
    parser.add_argument('--hybrid_model_top_k', type=int, default=20,
                        help="2-model pipeline co-reactant count for "
                             "template_2model mode (pre-filter)")
    parser.add_argument('--hybrid_total_top_k', type=int, default=128,
                        help="total action count (model + template) for "
                             "template_2model mode")
    parser.add_argument('--product_num_beams', type=int, default=1,
                        help="beam size for product prediction")

    parser.add_argument('--agent_model', type=str, default='mlp',
                        choices=['mlp', 'gnn'],
                        help="DQN model type")
    parser.add_argument('--gnn_checkpoint', type=str,
                        default='/shared/data1/Users/l1062811/git/marl/'
                                'checkpoints/large_pretrained_qed_predictor.pt',
                        help="path to pretrained GNN encoder checkpoint")

    parser.add_argument('--save_freq', type=int, default=100,
                        help="save frequency (episodes)")
    parser.add_argument('--log_freq', type=int, default=10,
                        help="log frequency (episodes)")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU device id")

    parser.add_argument('--eval', action='store_true',
                        help="evaluation mode (greedy rollout, eps=0)")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="checkpoint path to load")
    parser.add_argument('--output', type=str, default=None,
                        help="output pickle path for eval mode")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--reaction_only', action='store_true',
                        help="only use reactions (no atom/bond mods)")
    parser.add_argument('--num_molecules', type=int, default=None,
                        help="number of molecules to run in parallel")
    parser.add_argument('--batch_actions', action='store_true', default=True,
                        help="use batch action prediction")

    # Plan B+C
    parser.add_argument('--expand_fragments', action='store_true',
                        help="keep all T5 product fragments as actions")
    parser.add_argument('--mw_penalty', type=float, default=0.0,
                        help="MW penalty alpha for co-reactant scoring")
    parser.add_argument('--mw_threshold', type=float, default=150.0,
                        help="MW below this gets no penalty")
    parser.add_argument('--co_reactant_oversample', type=int, default=1,
                        help="oversample factor for co-reactant retrieval")

    # Product filter
    parser.add_argument('--no_product_filter', action='store_true',
                        help="disable product validation filter")
    parser.add_argument('--filter_min_tanimoto', type=float, default=0.2,
                        help="min Tanimoto(reactant, product)")
    parser.add_argument('--filter_max_mw_delta', type=float, default=200.0,
                        help="max |MW(product) - MW(reactant)| in Da")

    # Reward shaping
    parser.add_argument('--reward_tanimoto_bonus', type=float, default=0.0,
                        help="bonus for structural similarity to prev mol")
    parser.add_argument('--reward_mw_penalty', type=float, default=0.0,
                        help="penalty for MW change")

    return parser.parse_args(argv)


def train(args):
    """Main training loop."""

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load initial molecules
    if args.init_mol_path:
        init_mols = load_init_mols(args.init_mol_path)
    else:
        init_mols = args.init_mol

    # Select num_molecules
    if args.num_molecules is not None:
        if args.num_molecules <= len(init_mols):
            init_mols = random.sample(init_mols, args.num_molecules)
        else:
            print(f"Warning: num_molecules={args.num_molecules} > "
                  f"available={len(init_mols)}, cycling")
            expanded = []
            for i in range(args.num_molecules):
                expanded.append(init_mols[i % len(init_mols)])
            init_mols = expanded

    print(f"Initial molecules ({len(init_mols)}): "
          f"{init_mols[:10]}{'...' if len(init_mols) > 10 else ''}")

    # For template_2model mode, override top_k
    effective_top_k = args.hypergraph_top_k
    effective_model_top_k = args.hybrid_model_top_k
    if args.reactant_method == 'template_2model':
        effective_top_k = args.hybrid_total_top_k

    # Create environment
    env = QEDEnvironment(
        init_mols=init_mols,
        qed_weight=args.qed_weight,
        sa_weight=args.sa_weight,
        discount_factor=args.discount_factor,
        max_steps=args.max_steps_per_episode,
        use_hypergraph=args.use_hypergraph,
        hypergraph_top_k=effective_top_k,
        device=str(device),
        reaction_only=args.reaction_only,
        reactant_method=args.reactant_method,
        product_num_beams=args.product_num_beams,
        expand_fragments=args.expand_fragments,
        mw_penalty=args.mw_penalty,
        mw_threshold=args.mw_threshold,
        co_reactant_oversample=args.co_reactant_oversample,
        filter_products=not args.no_product_filter,
        filter_min_tanimoto=args.filter_min_tanimoto,
        filter_max_mw_delta=args.filter_max_mw_delta,
        reward_tanimoto_bonus=args.reward_tanimoto_bonus,
        reward_mw_penalty=args.reward_mw_penalty,
        model_top_k=effective_model_top_k,
    )

    # Create DQN
    use_gnn = args.agent_model == 'gnn'
    if use_gnn:
        input_dim = None  # determined inside create_dqn
        dqn, target_dqn, opt_params = create_dqn(
            'gnn', device, gnn_checkpoint=args.gnn_checkpoint)
    else:
        input_dim = hyp.fingerprint_length + 1
        dqn, target_dqn, opt_params = create_dqn(
            'mlp', device, input_dim=input_dim)

    optimizer = torch.optim.Adam(opt_params, lr=args.lr)
    replay_buffer = SimpleReplayBuffer(args.replay_buffer_size)

    # Load checkpoint if specified
    if args.load_checkpoint or args.eval:
        checkpoint_path = args.load_checkpoint
        if checkpoint_path is None:
            checkpoint_path = (
                f'Experiments/models/{args.experiment}_{args.trial}'
                f'_checkpoint.pth')

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            dqn.load_state_dict(checkpoint['dqn_state'])
            target_dqn.load_state_dict(checkpoint['target_dqn_state'])
            if not args.eval:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded checkpoint from episode "
                  f"{checkpoint.get('episode', 'unknown')}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Evaluation mode
    if args.eval:
        return _run_eval(args, env, dqn, use_gnn, device)

    # Training
    eps = args.eps_start
    best_reward = -float('inf')
    all_rewards = []
    all_qeds = []
    all_sas = []

    top_paths = PriorityQueue()
    last_paths = []
    record_top_path = 5
    record_last_path = 5

    max_episodes = args.iteration // args.max_steps_per_episode
    n_mols = len(init_mols)

    n_updates_per_ep = max(
        1, n_mols * max(args.max_steps_per_episode - 1, 1) // args.batch_size)
    print(f"\n{'='*60}")
    print(f"Starting training: {max_episodes} episodes, {n_mols} molecules")
    print(f"  Gradient updates per episode: {n_updates_per_ep}")
    print(f"  Replay buffer: {args.replay_buffer_size}, "
          f"batch_size: {args.batch_size}")
    print(f"  Eps: {args.eps_start} -> {args.eps_min}, "
          f"decay={args.eps_decay:.6f}")
    print(f"{'='*60}\n")

    for episode in range(max_episodes):
        episode_start = time.time()

        states = env.reset()
        episode_reward = 0

        episode_path = [[Chem.MolToSmiles(m) for m in states]]
        episode_qeds = [[QED.qed(m) for m in states]]
        episode_sas = []

        prev_selected_obs = [None] * len(states)
        prev_rewards = None

        for step in range(args.max_steps_per_episode):
            all_valid_actions = env.get_valid_actions_batch(states)

            all_observations = env.compute_observations_batch(
                all_valid_actions, step,
                gnn_dqn=dqn if use_gnn else None)

            # Store transitions from previous step
            if prev_rewards is not None:
                for mol_idx in range(len(states)):
                    if prev_selected_obs[mol_idx] is not None:
                        with torch.no_grad():
                            next_obs_t = torch.tensor(
                                all_observations[mol_idx],
                                dtype=torch.float32, device=device)
                            max_next_q = target_dqn(next_obs_t).max().item()
                        replay_buffer.add({
                            'state_obs': prev_selected_obs[mol_idx],
                            'reward': prev_rewards[mol_idx],
                            'max_next_q': max_next_q,
                            'done': False,
                        })

            # Select actions
            actions = []
            for mol_idx in range(len(states)):
                valid_actions = all_valid_actions[mol_idx]
                observations = all_observations[mol_idx]

                if random.random() < eps:
                    action_idx = random.randint(0, len(valid_actions) - 1)
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(
                            observations, dtype=torch.float32, device=device)
                        q_values = dqn(obs_tensor)
                        action_idx = q_values.argmax().item()

                actions.append(valid_actions[action_idx])
                prev_selected_obs[mol_idx] = observations[action_idx]

            result = env.step(actions)
            states = env.states
            prev_rewards = result['reward']
            episode_reward = sum(result['reward'])

            episode_path.append([Chem.MolToSmiles(m) for m in states])
            episode_qeds.append(result.get('QED', [0.0] * n_mols))
            if 'SA_score' in result:
                episode_sas.append(result['SA_score'])

        # Terminal transitions
        for mol_idx in range(len(states)):
            if (prev_selected_obs[mol_idx] is not None
                    and prev_rewards is not None):
                replay_buffer.add({
                    'state_obs': prev_selected_obs[mol_idx],
                    'reward': prev_rewards[mol_idx],
                    'max_next_q': 0.0,
                    'done': True,
                })

        # Record stats
        mean_reward = episode_reward / n_mols

        # Generic property metrics
        metrics = {'reward': mean_reward}
        for key in result:
            if key not in ('reward', 'done'):
                metrics[key] = sum(result[key]) / n_mols

        all_rewards.append(mean_reward)
        all_qeds.append(metrics.get('QED', 0.0))
        all_sas.append(metrics.get('SA_score', 0.0))

        # Save top paths (use QED if available, else reward)
        sort_metric_vals = result.get('QED', result['reward'])
        for i in range(n_mols):
            final_metric = sort_metric_vals[i]
            if (top_paths.qsize() < record_top_path
                    or final_metric > top_paths.queue[0][0]):
                path_data = {
                    'path': [p[i] if i < len(p) else p[0]
                             for p in episode_path],
                    'qeds': [q[i] if i < len(q) else q[0]
                             for q in episode_qeds],
                    'sas': ([s[i] if i < len(s) else s[0]
                             for s in episode_sas] if episode_sas else []),
                    'final_qed': (result['QED'][i]
                                  if 'QED' in result else None),
                    'final_reward': result['reward'][i],
                    'episode': episode,
                }
                top_paths.put((final_metric, episode, i, path_data))
                if top_paths.qsize() > record_top_path:
                    top_paths.get()

        # Last paths
        if episode >= max_episodes - record_last_path:
            last_paths.append({
                'episode': episode,
                'paths': episode_path,
                'qeds': episode_qeds,
                'sas': episode_sas,
                'rewards': result['reward'],
            })

        # Training step
        n_new_experiences = n_mols * args.max_steps_per_episode
        n_updates = max(1, n_new_experiences // args.batch_size)

        if len(replay_buffer) >= args.batch_size:
            for _ in range(n_updates):
                batch = replay_buffer.sample(args.batch_size)

                state_obs_batch = torch.tensor(
                    np.array([exp['state_obs'] for exp in batch]),
                    dtype=torch.float32, device=device)
                rewards_batch = torch.tensor(
                    [exp['reward'] for exp in batch],
                    dtype=torch.float32, device=device)
                max_next_q_batch = torch.tensor(
                    [exp['max_next_q'] for exp in batch],
                    dtype=torch.float32, device=device)
                done_batch = torch.tensor(
                    [float(exp['done']) for exp in batch],
                    dtype=torch.float32, device=device)

                q_values = dqn(state_obs_batch).squeeze(-1)
                targets = (rewards_batch
                           + (1.0 - done_batch) * hyp.gamma
                           * max_next_q_batch)
                loss = ((q_values - targets) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Soft update target network
            if use_gnn:
                for p, p_targ in zip(dqn.q_head.parameters(),
                                     target_dqn.q_head.parameters()):
                    p_targ.data.mul_(hyp.polyak)
                    p_targ.data.add_((1 - hyp.polyak) * p.data)
            else:
                for p, p_targ in zip(dqn.parameters(),
                                     target_dqn.parameters()):
                    p_targ.data.mul_(hyp.polyak)
                    p_targ.data.add_((1 - hyp.polyak) * p.data)

        # Decay epsilon
        eps = max(args.eps_min, eps * args.eps_decay)

        # Logging
        if (episode + 1) % args.log_freq == 0:
            episode_time = time.time() - episode_start
            metric_str = " | ".join(
                f"{k}: {v:.3f}" for k, v in metrics.items()
                if k != 'reward')
            print(f"Episode {episode+1}/{max_episodes} | "
                  f"Reward: {mean_reward:.4f} | {metric_str} | "
                  f"Eps: {eps:.3f} | Time: {episode_time:.2f}s")

            for i, mol in enumerate(states[:3]):
                smiles = Chem.MolToSmiles(mol)
                qed_val = (result['QED'][i] if 'QED' in result
                           else result['reward'][i])
                print(f"  Mol {i}: {smiles} (QED={qed_val:.3f})")

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0 or mean_reward > best_reward:
            if mean_reward > best_reward:
                best_reward = mean_reward

            os.makedirs('Experiments/models', exist_ok=True)
            torch.save({
                'episode': episode,
                'dqn_state': dqn.state_dict(),
                'target_dqn_state': target_dqn.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'eps': eps,
                'best_reward': best_reward,
            }, f'Experiments/models/{args.experiment}_{args.trial}'
               f'_checkpoint.pth')

            with open(f'Experiments/{args.experiment}_{args.trial}'
                      f'_history.pickle', 'wb') as f:
                pickle.dump({
                    'rewards': all_rewards,
                    'qeds': all_qeds,
                    'sas': all_sas,
                }, f)

            top_paths_list = sorted(
                [item[3] for item in list(top_paths.queue)],
                key=lambda x: x.get('final_qed') or x['final_reward'],
                reverse=True)
            with open(f'Experiments/{args.experiment}_{args.trial}'
                      f'_paths.pickle', 'wb') as f:
                pickle.dump({
                    'top_paths': top_paths_list,
                    'last_paths': last_paths,
                }, f)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best reward: {best_reward:.4f}")
    print(f"{'='*60}")

    print("\nFinal molecules:")
    for i, mol in enumerate(states):
        smiles = Chem.MolToSmiles(mol)
        qed_val = QED.qed(mol)
        print(f"  {i}: {smiles} (QED={qed_val:.3f})")

    return dqn, best_reward


def _run_eval(args, env, dqn, use_gnn, device):
    """Run evaluation mode (greedy rollout, eps=0).

    Runs all molecules through max_steps with pure Q-value greedy action
    selection. Produces per-molecule statistics, sorted tables, and
    comprehensive pickle output.
    """
    dqn.eval()
    n_mols = len(env.init_mols)

    # SA scorer
    from rdkit import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    states = env.reset()

    paths = []
    qeds_history = []
    sas_history = []
    action_info_history = []

    init_smiles = [Chem.MolToSmiles(m) for m in states]
    init_qeds = [QED.qed(m) for m in states]
    init_sas = [sascorer.calculateScore(m) for m in states]
    paths.append(init_smiles)
    qeds_history.append(init_qeds)
    sas_history.append(init_sas)

    print(f"\n{'='*60}")
    print(f"EVALUATION MODE (greedy, eps=0)")
    print(f"  {n_mols} molecules, {args.max_steps_per_episode} steps")
    print(f"{'='*60}")

    total_start = time.time()

    for step in range(args.max_steps_per_episode):
        step_start = time.time()

        all_valid_actions, all_action_info = env.get_valid_actions_batch(
            states, return_info=True)

        all_observations = env.compute_observations_batch(
            all_valid_actions, step,
            gnn_dqn=dqn if use_gnn else None)

        actions = []
        step_action_info = []
        for mol_idx in range(n_mols):
            valid_actions = all_valid_actions[mol_idx]
            observations = all_observations[mol_idx]

            with torch.no_grad():
                obs_tensor = torch.tensor(
                    observations, dtype=torch.float32, device=device)
                q_values = dqn(obs_tensor)
                action_idx = q_values.argmax().item()

            actions.append(valid_actions[action_idx])

            if (mol_idx < len(all_action_info)
                    and action_idx < len(all_action_info[mol_idx])):
                info = all_action_info[mol_idx][action_idx]
                info['q_value'] = q_values[action_idx].item()
                info['q_values_all'] = q_values.squeeze(-1).cpu().tolist()
                step_action_info.append(info)
            else:
                step_action_info.append(
                    {'q_value': q_values[action_idx].item()})

        action_info_history.append(step_action_info)

        result = env.step(actions)
        states = env.states

        step_smiles = [Chem.MolToSmiles(m) for m in states]
        paths.append(step_smiles)
        qeds_history.append(result.get('QED', [0.0] * n_mols))
        sas_history.append(result.get('SA_score',
                           [sascorer.calculateScore(m) for m in states]))

        step_time = time.time() - step_start
        mean_qed = np.mean(qeds_history[-1])
        mean_sa = np.mean(sas_history[-1])
        n_changed = sum(1 for s0, s1 in zip(paths[-2], paths[-1])
                        if s0 != s1)
        print(f"  Step {step+1}/{args.max_steps_per_episode} | "
              f"QED: {mean_qed:.3f} | SA: {mean_sa:.2f} | "
              f"Changed: {n_changed}/{n_mols} | Time: {step_time:.2f}s")

    total_time = time.time() - total_start

    # --- QED Summary ---
    init_qeds_arr = np.array(qeds_history[0])
    final_qeds_arr = np.array(qeds_history[-1])
    improvements = final_qeds_arr - init_qeds_arr

    print(f"\n{'='*60}")
    print(f"Evaluation complete in {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"\nQED Summary:")
    print(f"  Initial:  mean={init_qeds_arr.mean():.3f}, "
          f"median={np.median(init_qeds_arr):.3f}")
    print(f"  Final:    mean={final_qeds_arr.mean():.3f}, "
          f"median={np.median(final_qeds_arr):.3f}")
    print(f"  Improved: {(improvements > 0.01).sum()}/{n_mols} molecules")
    print(f"  Degraded: {(improvements < -0.01).sum()}/{n_mols} molecules")
    print(f"  QED > 0.8: {(final_qeds_arr > 0.8).sum()}/{n_mols}")
    print(f"  QED > 0.9: {(final_qeds_arr > 0.9).sum()}/{n_mols}")

    # --- Per-molecule results ---
    mol_results = []
    for i in range(n_mols):
        mol_result = {
            'mol_idx': i,
            'init_smiles': paths[0][i],
            'final_smiles': paths[-1][i],
            'init_qed': qeds_history[0][i],
            'final_qed': qeds_history[-1][i],
            'init_sa': sas_history[0][i],
            'final_sa': sas_history[-1][i],
            'qed_improvement': qeds_history[-1][i] - qeds_history[0][i],
            'path_smiles': [paths[s][i] for s in range(len(paths))],
            'path_qeds': [qeds_history[s][i]
                          for s in range(len(qeds_history))],
            'path_sas': [sas_history[s][i] for s in range(len(sas_history))],
            'path_actions': [action_info_history[s][i]
                             for s in range(len(action_info_history))],
        }
        effective_steps = sum(
            1 for s in range(1, len(paths))
            if paths[s][i] != paths[s-1][i])
        mol_result['effective_steps'] = effective_steps
        mol_results.append(mol_result)

    mol_results_sorted = sorted(
        mol_results, key=lambda x: x['final_qed'], reverse=True)

    # --- Top 10 / Bottom 5 tables ---
    print(f"\nTop 10 molecules by final QED:")
    print(f"  {'Idx':>4} {'Init QED':>9} {'Final QED':>10} "
          f"{'Delta':>7} {'Steps':>5}  Final SMILES")
    for r in mol_results_sorted[:10]:
        delta = r['qed_improvement']
        sign = '+' if delta >= 0 else ''
        print(f"  {r['mol_idx']:4d} {r['init_qed']:9.3f} "
              f"{r['final_qed']:10.3f} "
              f"{sign}{delta:6.3f} {r['effective_steps']:5d}  "
              f"{r['final_smiles'][:80]}")

    if n_mols > 10:
        print(f"\nBottom 5 molecules by final QED:")
        for r in mol_results_sorted[-5:]:
            delta = r['qed_improvement']
            sign = '+' if delta >= 0 else ''
            print(f"  {r['mol_idx']:4d} {r['init_qed']:9.3f} "
                  f"{r['final_qed']:10.3f} "
                  f"{sign}{delta:6.3f} {r['effective_steps']:5d}  "
                  f"{r['final_smiles'][:80]}")

    # --- Best molecule synthesis path ---
    best = mol_results_sorted[0]
    print(f"\nBest molecule synthesis path (mol {best['mol_idx']}):")
    for s in range(len(best['path_smiles'])):
        marker = ('*' if s > 0 and
                  best['path_smiles'][s] != best['path_smiles'][s-1]
                  else ' ')
        action_src = ''
        if s > 0 and s - 1 < len(best['path_actions']):
            act = best['path_actions'][s-1]
            action_src = f" [{act.get('source', '?')}]"
            if act.get('co_reactant'):
                action_src += f" + {act['co_reactant']}"
        print(f"  {marker} Step {s}: QED={best['path_qeds'][s]:.3f} "
              f"SA={best['path_sas'][s]:.2f}{action_src}")
        print(f"         {best['path_smiles'][s]}")

    # --- Save pickle ---
    output_path = args.output
    if output_path is None:
        output_path = (f'Experiments/{args.experiment}_{args.trial}'
                       f'_eval.pickle')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_data = {
        'args': vars(args),
        'mol_results': mol_results,
        'mol_results_sorted': mol_results_sorted,
        'paths': paths,
        'qeds_history': qeds_history,
        'sas_history': sas_history,
        'action_info_history': action_info_history,
        'summary': {
            'n_mols': n_mols,
            'max_steps': args.max_steps_per_episode,
            'init_qed_mean': float(init_qeds_arr.mean()),
            'final_qed_mean': float(final_qeds_arr.mean()),
            'init_qed_median': float(np.median(init_qeds_arr)),
            'final_qed_median': float(np.median(final_qeds_arr)),
            'n_improved': int((improvements > 0.01).sum()),
            'n_degraded': int((improvements < -0.01).sum()),
            'n_qed_above_0.8': int((final_qeds_arr > 0.8).sum()),
            'n_qed_above_0.9': int((final_qeds_arr > 0.9).sum()),
            'total_time': total_time,
        }
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to: {output_path}")

    return dqn, float(final_qeds_arr.max())


# ======================================================================
# Backward-compatible re-exports
# ======================================================================

def _compute_observations_batch(all_valid_actions, step, max_steps, env,
                                gnn_dqn=None):
    """Legacy wrapper — delegates to env.compute_observations_batch()."""
    return env.compute_observations_batch(all_valid_actions, step,
                                          gnn_dqn=gnn_dqn)


if __name__ == '__main__':
    args = parse_args()
    train(args)
