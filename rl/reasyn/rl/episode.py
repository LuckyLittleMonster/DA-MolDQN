"""ReaSyn RL episode runner."""

import random
import torch

from dqn import make_observation
from reward import compute_reward
from .actions import get_reasyn_actions


def run_episode(smiles, models, fpindex, rxn_matrix, dqn, target_dqn,
                eps, max_steps, gamma, cfg_method, cfg_reward,
                device_str='cpu', dock_scorer=None, action_cache=None):
    """Run one RL episode for a single molecule (ReaSyn method).

    Returns dict with transitions, path, rewards, scores, etc.
    """
    device = torch.device(device_str)
    top_k = cfg_method.top_k
    transitions = []
    path = [smiles]
    rewards = []
    scores = []
    synthesis_per_step = []  # ReaSyn synthesis route for each step's selected mol
    synthesis_map = {}  # product_smiles -> synthesis_string
    cache_updates = {}
    reasyn_calls = 0
    cache_hits = 0

    current_smiles = smiles
    prev_obs = None
    prev_reward = None

    for step in range(max_steps):
        step_frac = step / max_steps

        # 1. Get candidates: cache-first, ReaSyn on miss/explore
        use_reasyn = True
        if action_cache is not None and not action_cache.should_call_reasyn(current_smiles):
            cached = action_cache.get_actions(current_smiles, top_k=top_k)
            if cached:
                reasyn_cands = [s for s, _ in cached]
                use_reasyn = False
                cache_hits += 1

        if use_reasyn:
            reasyn_calls += 1
            scored = get_reasyn_actions(
                current_smiles, models, fpindex, rxn_matrix, cfg_method,
                top_k=top_k, return_scores=True,
            )
            reasyn_cands = [s for s, _, _syn in scored]
            # Store synthesis routes
            for s, _sc, syn in scored:
                if syn:
                    synthesis_map[s] = syn
            # Strip synthesis for cache (cache stores (smiles, score) only)
            scored_for_cache = [(s, sc) for s, sc, _syn in scored]
            if action_cache is not None and scored_for_cache:
                action_cache.update(current_smiles, scored_for_cache)
                if current_smiles not in cache_updates:
                    cache_updates[current_smiles] = []
                cache_updates[current_smiles].extend(scored_for_cache)

        candidates = [current_smiles]
        seen = {current_smiles}
        for c in reasyn_cands:
            if c not in seen:
                candidates.append(c)
                seen.add(c)

        # 2. Compute observations for all candidates
        obs_list = [make_observation(c, step_frac) for c in candidates]
        obs_batch = torch.stack(obs_list).to(device)

        # 3. Epsilon-greedy action selection
        if random.random() < eps:
            action_idx = random.randrange(len(candidates))
        else:
            with torch.no_grad():
                q_values = dqn(obs_batch).squeeze(-1)
                action_idx = q_values.argmax().item()

        # 4. Execute action
        selected_smiles = candidates[action_idx]
        selected_obs = obs_list[action_idx]
        synthesis_per_step.append(synthesis_map.get(selected_smiles, ''))

        # 5. Compute reward
        rdict = compute_reward(
            selected_smiles, step, max_steps, gamma, cfg_reward,
            dock_scorer=dock_scorer)
        reward = rdict['reward']
        score = rdict['dock_score'] if cfg_reward.name in ('dock', 'dock_deprecated', 'multi', 'multi_deprecated') else rdict['qed']
        rewards.append(reward)
        scores.append(score)

        # 6. Delayed storage -- store next_obs for fresh Q computation at training
        if prev_obs is not None:
            transitions.append({
                'obs': prev_obs.cpu(),
                'reward': prev_reward,
                'next_obs': selected_obs.cpu(),
                'done': False,
            })

        prev_obs = selected_obs
        prev_reward = reward
        current_smiles = selected_smiles
        path.append(current_smiles)

    # Terminal transition (next_obs=None for terminal states)
    if prev_obs is not None:
        transitions.append({
            'obs': prev_obs.cpu(),
            'reward': prev_reward,
            'next_obs': None,
            'done': True,
        })

    return {
        'transitions': transitions,
        'path': path,
        'rewards': rewards,
        'scores': scores,
        'synthesis': synthesis_per_step,
        'final_smiles': current_smiles,
        'init_smiles': smiles,
        'cache_updates': cache_updates,
        'reasyn_calls': reasyn_calls,
        'cache_hits': cache_hits,
    }
