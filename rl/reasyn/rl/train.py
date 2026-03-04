"""ReaSyn DQN training — ReaSynTrainer(RLTrainer).

Supports three collection modes:
1. Sync dock: step-by-step with batch docking in main process
2. Async workers: full episodes in worker processes (QED mode)
3. Sequential: single-process mode with ReaSyn loaded in main
"""

import json
import os
import pathlib
import pickle
import random
import shutil
import tempfile
import time
from collections import deque

import multiprocessing as _mp

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from dqn import MLPDQN, make_observation
from rl.trainer import RLTrainer
from training_utils import (
    save_checkpoint, save_pickle, load_molecules,
)
from reward import compute_reward, make_dock_config, make_dock_scorer
from reward.core import _load_sascorer
from reward.admet.reward import _get_admet_model

from .replay_buffer import ReplayBuffer
from .action_cache import IncrementalActionCache
from .actions import get_reasyn_actions, get_reasyn_actions_full
from .episode import run_episode
from .workers import RLWorker, ReaSynActionWorker

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class ReaSynTrainer(RLTrainer):
    """ReaSyn DQN trainer with synthesis route generation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dock = cfg.reward.name in (
            'dock', 'dock_rxnflow', 'dock_deprecated',
            'multi', 'multi_deprecated')

        # Models
        self.dqn = None
        self.target_dqn = None
        self.optimizer = None
        self.replay = None
        self.dock_scorer_main = None
        self.dock_config = None

        # Data
        self.all_smiles = []
        self.model_paths = []

        # Workers
        self._workers_shutdown = False
        self.reasyn_action_queue = None
        self.reasyn_result_queue = None
        self.reasyn_action_workers = []
        self.task_queue = None
        self.result_queue = None
        self.workers = []
        self.seq_models = None
        self.seq_fpindex = None
        self.seq_rxn_matrix = None

        # Cache
        self.global_cache = None
        self.weight_dir = None
        self.num_workers = cfg.method.num_workers
        self.use_sync_dock = False  # set in setup()

        # Tracking
        self.reasyn_history = None  # dict of lists, set in setup()
        self.best_paths = []
        self.recent_episodes = deque(maxlen=5)
        self.synthesis_map = {}
        self.oracle_counts = []  # Track oracle consumption per episode

    # ── Setup ─────────────────────────────────────────────────────

    def setup(self):
        cfg = self.cfg

        mode_str = (
            f"Docking({cfg.reward.target})"
            if 'dock' in cfg.reward.name and 'multi' not in cfg.reward.name
            else f"Multi({cfg.reward.target})"
            if 'multi' in cfg.reward.name
            else cfg.reward.name.upper())

        print(f"\n{'='*70}")
        print(f"ReaSyn DQN Training -- {mode_str}")
        print(f"  episodes={cfg.episodes}, max_steps={cfg.max_steps}, "
              f"workers={cfg.method.num_workers}")
        print(f"  gamma={cfg.gamma}, lr={cfg.lr}, top_k={cfg.method.top_k}")
        print(f"  eps: {cfg.eps_start} (decay={cfg.eps_decay})")
        print(f"{'='*70}\n")

        # Model paths
        model_dir = PROJECT_ROOT / cfg.method.model_dir
        self.model_paths = [model_dir / f for f in cfg.method.model_files]

        # Load molecules
        self._load_molecules(model_dir)

        # Determine sync dock mode from config (no CUDA objects yet)
        scoring_method = cfg.reward.get('scoring_method', 'dock')
        self.dock_config = (make_dock_config(cfg.reward)
                            if scoring_method != 'proxy' else None)
        # Always use sync collection with lightweight ReaSynActionWorkers.
        # Even QED benefits: avoids heavy RLWorker (each loads full model
        # copies + action_cache → OOM at ~140GB with 16 workers).
        self.use_sync_dock = True

        # Weight sharing directory + cache (no CUDA)
        self.weight_dir = tempfile.mkdtemp(prefix="reasyn_dqn_")
        print(f"  Weight sync dir: {self.weight_dir}")
        self._setup_cache()

        # Workers MUST start BEFORE CUDA init (fork+CUDA deadlock)
        self._setup_workers(model_dir)

        # DQN (initializes CUDA — after workers forked)
        input_dim = cfg.method.get('input_dim', 4097)
        hidden_dim = cfg.method.get('hidden_dim', 256)
        self.dqn = MLPDQN(input_dim, hidden_dim).to(self.device)
        self.target_dqn = MLPDQN(input_dim, hidden_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)

        # Dock scorer (may use CUDA for proxy — safe after workers forked)
        self.dock_scorer_main = make_dock_scorer(cfg.reward)

        # History tracking
        self.reasyn_history = {
            'episode': [],
            'mean_reward': [],
            'mean_score': [],
            'best_score': [],
            'best_score_ever': [],
            'eps': [],
            'time_s': [],
            'replay_size': [],
            'loss': [],
            'cache_mols': [],
            'cache_edges': [],
            'reward_mode': cfg.reward.name,
            'config': OmegaConf.to_container(cfg, resolve=True),
        }

        # Print header
        score_label = "Dock" if self.is_dock else "QED"
        cache_hdr = (" | CacheHit | Mols | Edges"
                     if self.global_cache else "")
        print(f"\n{'='*70}")
        print(f"{'Ep':>4s} | {'Reward':>7s} | {score_label:>5s} | "
              f"{'Best':>6s} | {'BstEvr':>7s} | {'Eps':>5s} | "
              f"{'Loss':>7s} | {'Buf':>6s} | {'Time':>6s}{cache_hdr}")
        print(f"{'-'*70}")

    def _load_molecules(self, model_dir):
        cfg = self.cfg
        if cfg.mol_json or cfg.get('init_mol_path') or cfg.get('init_mol'):
            self.all_smiles = load_molecules(cfg, PROJECT_ROOT)
            print(f"  Loaded {len(self.all_smiles)} molecules")
        else:
            zinc_path = model_dir.parent / 'zinc_first64.txt'
            with open(zinc_path) as f:
                lines = f.read().strip().split("\n")
            if lines[0].upper() == "SMILES":
                lines = lines[1:]
            num_mols = cfg.num_molecules or 64
            self.all_smiles = [s.strip() for s in lines[:num_mols]]
            print(f"  Loaded {len(self.all_smiles)} molecules "
                  f"from {zinc_path.name}")

    def _setup_cache(self):
        cfg = self.cfg
        if not cfg.method.use_cache:
            return
        cache_config = {
            'min_cached_actions': cfg.method.min_cached_actions,
            'max_neighbors': cfg.method.max_neighbors,
            'explore_prob': cfg.method.explore_prob,
        }
        self.global_cache = IncrementalActionCache(**cache_config)
        if cfg.method.cache_path and os.path.exists(cfg.method.cache_path):
            self.global_cache.load(cfg.method.cache_path)
            print(f"  Loaded cache: {self.global_cache.num_molecules} mols, "
                  f"{self.global_cache.num_edges} edges")
        print(f"  Cache: explore_prob={cfg.method.explore_prob}, "
              f"min_actions={cfg.method.min_cached_actions}")

    def _setup_workers(self, model_dir):
        cfg = self.cfg
        cfg_method_dict = OmegaConf.to_container(cfg.method, resolve=True)
        cfg_reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
        cache_config = None
        if cfg.method.use_cache:
            cache_config = {
                'min_cached_actions': cfg.method.min_cached_actions,
                'max_neighbors': cfg.method.max_neighbors,
                'explore_prob': cfg.method.explore_prob,
            }

        if self.use_sync_dock and self.num_workers > 0:
            # Sync dock mode: lightweight ReaSyn action workers
            self.reasyn_action_queue = _mp.Queue()
            self.reasyn_result_queue = _mp.Queue()
            for _ in range(self.num_workers):
                w = ReaSynActionWorker(
                    model_paths=[str(p) for p in self.model_paths],
                    task_queue=self.reasyn_action_queue,
                    result_queue=self.reasyn_result_queue,
                    cfg_method_dict=cfg_method_dict,
                )
                w.start()
                self.reasyn_action_workers.append(w)
            for _ in range(self.num_workers):
                self.reasyn_result_queue.get()  # 'ready' signal
            print(f"  Started {self.num_workers} ReaSyn action workers "
                  f"(dock in main process)")

        elif self.num_workers > 0:
            # QED mode: full episode workers
            self.task_queue = _mp.JoinableQueue()
            self.result_queue = _mp.Queue()
            for _ in range(self.num_workers):
                w = RLWorker(
                    model_paths=[str(p) for p in self.model_paths],
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    max_steps=cfg.max_steps,
                    gamma=cfg.gamma,
                    top_k=cfg.method.top_k,
                    weight_dir=self.weight_dir,
                    cfg_method_dict=cfg_method_dict,
                    cfg_reward_dict=cfg_reward_dict,
                    dock_config=self.dock_config,
                    cache_config=cache_config,
                )
                w.start()
                self.workers.append(w)
            print(f"  Started {len(self.workers)} workers")

        else:
            # Sequential mode: load ReaSyn in main process
            print("  Sequential mode: loading ReaSyn models...")
            from ..models.reasyn import ReaSyn

            seq_device = "cuda"
            self.seq_models = []
            reasyn_config = None
            for p in self.model_paths:
                ckpt = torch.load(p, map_location="cpu")
                reasyn_config = OmegaConf.create(
                    ckpt["hyper_parameters"]["config"])
                m = ReaSyn(reasyn_config.model)
                m.load_state_dict(
                    {k[6:]: v for k, v in ckpt["state_dict"].items()})
                m = m.half().to(seq_device)
                m.eval()
                self.seq_models.append(m)
            _reasyn_root = model_dir.parent.parent
            _fpindex_path = _reasyn_root / reasyn_config.chem.fpindex
            _rxn_matrix_path = _reasyn_root / reasyn_config.chem.rxn_matrix
            self.seq_fpindex = pickle.load(open(_fpindex_path, "rb"))
            self.seq_rxn_matrix = pickle.load(open(_rxn_matrix_path, "rb"))
            print("  Models loaded.")

    # ── Checkpoint ────────────────────────────────────────────────

    def try_load_checkpoint(self):
        cfg = self.cfg
        if not cfg.load_checkpoint or not os.path.exists(cfg.load_checkpoint):
            return
        ckpt = torch.load(cfg.load_checkpoint, map_location=self.device,
                          weights_only=False)
        self.dqn.load_state_dict(ckpt['dqn'])
        self.target_dqn.load_state_dict(ckpt['target_dqn'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        eps_val = ckpt.get('eps')
        if eps_val is not None:
            self.eps = eps_val
        best = ckpt.get('best_score_ever')
        if best is not None:
            self.best_metric = best
        print(f"  Loaded checkpoint: {cfg.load_checkpoint}")
        print(f"    (ep={ckpt.get('episode', '?')}, "
              f"eps={self.eps:.4f}, "
              f"best={self.best_metric})")

    def save(self, episode, history):
        if not self.cfg.get('no_save_checkpoint', False):
            save_checkpoint(
                self.model_save_dir / f'{self.prefix}_checkpoint.pth',
                dqn=self.dqn.state_dict(),
                target_dqn=self.target_dqn.state_dict(),
                optimizer=self.optimizer.state_dict(),
                episode=episode,
                eps=self.eps,
                best_score_ever=(self.best_metric
                                 if self.best_metric is not None else 0.0),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
        save_pickle(self.exp_dir / f'{self.prefix}_history.pickle',
                     self.reasyn_history)
        save_pickle(self.exp_dir / f'{self.prefix}_paths.pickle',
                     self.best_paths)
        save_pickle(self.exp_dir / f'{self.prefix}_recent_episodes.pickle',
                     list(self.recent_episodes))

        if self.global_cache is not None:
            if self.cfg.method.cache_path:
                self.global_cache.save(self.cfg.method.cache_path)
            self.global_cache.save(
                str(self.exp_dir / f'{self.prefix}_cache.pickle'))

    def should_save(self, episode, improved):
        return ((episode + 1) % self.cfg.save_freq == 0
                or episode == self.cfg.episodes - 1)

    def is_improvement(self, metrics):
        score = metrics.get('best_score')
        if score is None:
            return False
        if self.is_dock:
            if self.best_metric is None or score < self.best_metric:
                self.best_metric = score
                return True
        else:
            if self.best_metric is None or score > self.best_metric:
                self.best_metric = score
                return True
        return False

    # ── Episode ───────────────────────────────────────────────────

    def run_episode(self, episode):
        # 1. Collect results
        if self.use_sync_dock:
            results = self._collect_sync_dock(episode)
        else:
            results = self._collect_async(episode)

        # 2. Merge cache updates
        if self.global_cache is not None and self.num_workers > 0:
            for result in results:
                cu = result.get('cache_updates', {})
                if cu:
                    self.global_cache.merge(cu)
            cache_path = os.path.join(self.weight_dir, "action_cache.pkl")
            self.global_cache.save(cache_path)

        # 3. Add transitions to replay + compute stats
        n_transitions = 0
        ep_rewards = []
        ep_scores = []
        ep_best_score = None
        for result in results:
            for t in result['transitions']:
                self.replay.push(
                    t['obs'], t['reward'], t['next_obs'], t['done'])
                n_transitions += 1
            if result['rewards']:
                ep_rewards.append(sum(result['rewards']))
            if result['scores']:
                if self.is_dock:
                    best_s = min(result['scores'])
                else:
                    best_s = max(result['scores'])
                ep_scores.append(best_s)

                if ep_best_score is None:
                    ep_best_score = best_s
                elif self.is_dock:
                    ep_best_score = min(ep_best_score, best_s)
                else:
                    ep_best_score = max(ep_best_score, best_s)

                self.best_paths.append({
                    'episode': episode,
                    'path': result['path'],
                    'scores': result['scores'],
                    'rewards': result['rewards'],
                    'synthesis': result.get('synthesis', []),
                    'best_score': best_s,
                    'init_smiles': result['init_smiles'],
                    'final_smiles': result['final_smiles'],
                })

        self.best_paths.sort(key=lambda x: x['best_score'],
                             reverse=(not self.is_dock))
        self.best_paths = self.best_paths[:5]

        # Save per-molecule paths for recent episodes
        self.recent_episodes.append({
            'episode': episode,
            'molecules': [{
                'init_smiles': r['init_smiles'],
                'final_smiles': r['final_smiles'],
                'path': r['path'],
                'scores': r['scores'],
                'rewards': r['rewards'],
                'synthesis': r.get('synthesis', []),
            } for r in results],
        })

        # 4. Gradient updates
        total_loss = 0.0
        n_updates = 0
        cfg = self.cfg
        if len(self.replay) >= cfg.batch_size:
            n_grad = min(cfg.grad_steps, max(1, n_transitions))
            for _ in range(n_grad):
                obs, rew, next_obs, done = self.replay.sample(cfg.batch_size)
                obs = obs.to(self.device)
                rew = rew.to(self.device)
                next_obs = next_obs.to(self.device)
                done = done.to(self.device)

                q_pred = self.dqn(obs).squeeze(-1)
                with torch.no_grad():
                    next_q = self.target_dqn(next_obs).squeeze(-1)
                    q_target = rew + cfg.gamma * next_q * (1 - done)
                loss = F.mse_loss(q_pred, q_target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        # 5. Soft update target network
        self.soft_update(cfg.polyak, (self.dqn, self.target_dqn))

        # 6. Build metrics
        mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_score = np.mean(ep_scores) if ep_scores else 0.0
        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0

        # Track oracle consumption (unique molecules scored by proxy)
        oracle_count = (self.dock_scorer_main.cache_size
                        if self.dock_scorer_main else 0)
        self.oracle_counts.append(oracle_count)

        # Update history
        best_ever = (self.best_metric
                     if self.best_metric is not None else 0.0)
        self.reasyn_history['episode'].append(episode)
        self.reasyn_history['mean_reward'].append(float(mean_reward))
        self.reasyn_history['mean_score'].append(float(mean_score))
        self.reasyn_history['best_score'].append(
            float(ep_best_score if ep_best_score is not None else 0.0))
        self.reasyn_history['replay_size'].append(len(self.replay))
        self.reasyn_history['loss'].append(float(avg_loss))
        self.reasyn_history['cache_mols'].append(
            self.global_cache.num_molecules if self.global_cache else 0)
        self.reasyn_history['cache_edges'].append(
            self.global_cache.num_edges if self.global_cache else 0)
        # Note: eps, time_s, best_score_ever are appended in on_episode_end
        # (after base class sets metrics['eps'] and metrics['elapsed'])

        return {
            'reward': mean_reward,
            'score': mean_score,
            'best_score': ep_best_score,
            'loss': avg_loss,
            'replay_size': len(self.replay),
            'results': results,  # for cache hit stats in log_episode
            'oracle_count': oracle_count,
        }

    # ── Collection modes ──────────────────────────────────────────

    def _collect_sync_dock(self, episode):
        """Step-by-step collection with batch docking in main process."""
        cfg = self.cfg
        sascorer = _load_sascorer()
        n_mols = len(self.all_smiles)
        current_smiles = list(self.all_smiles)
        paths = [[smi] for smi in current_smiles]
        all_transitions = [[] for _ in range(n_mols)]
        all_rewards = [[] for _ in range(n_mols)]
        all_scores = [[] for _ in range(n_mols)]
        all_synthesis = [[] for _ in range(n_mols)]
        prev_obs = [None] * n_mols
        prev_reward = [None] * n_mols

        self.dqn.cpu()
        self.target_dqn.cpu()

        for step in range(cfg.max_steps):
            step_frac = step / cfg.max_steps

            # 1. Generate candidates for all molecules
            all_candidates = [None] * n_mols
            cache_misses = []

            for i, smi in enumerate(current_smiles):
                use_reasyn = True
                if (self.global_cache is not None
                        and not self.global_cache.should_call_reasyn(smi)):
                    cached = self.global_cache.get_actions(
                        smi, top_k=cfg.method.top_k)
                    if cached:
                        cands = [s for s, _ in cached]
                        use_reasyn = False

                if use_reasyn:
                    cache_misses.append((i, smi))
                else:
                    candidates = [smi]
                    seen = {smi}
                    for c in cands:
                        if c not in seen:
                            candidates.append(c)
                            seen.add(c)
                    all_candidates[i] = candidates

            # Process cache misses
            if cache_misses and self.reasyn_action_queue is not None:
                for idx, smi in cache_misses:
                    self.reasyn_action_queue.put((idx, smi))
                for _ in range(len(cache_misses)):
                    idx, smi, scored = self.reasyn_result_queue.get()
                    cands = [s for s, _, _syn in scored]
                    for s, _sc, syn in scored:
                        if syn:
                            self.synthesis_map[s] = syn
                    scored_for_cache = [(s, sc) for s, sc, _syn in scored]
                    if self.global_cache is not None and scored_for_cache:
                        self.global_cache.update(smi, scored_for_cache)
                    candidates = [smi]
                    seen = {smi}
                    for c in cands:
                        if c not in seen:
                            candidates.append(c)
                            seen.add(c)
                    all_candidates[idx] = candidates
            elif cache_misses:
                for idx, smi in cache_misses:
                    scored = get_reasyn_actions(
                        smi, self.seq_models, self.seq_fpindex,
                        self.seq_rxn_matrix,
                        cfg.method, top_k=cfg.method.top_k,
                        return_scores=True,
                    )
                    cands = [s for s, _, _syn in scored]
                    for s, _sc, syn in scored:
                        if syn:
                            self.synthesis_map[s] = syn
                    scored_for_cache = [(s, sc) for s, sc, _syn in scored]
                    if self.global_cache is not None and scored_for_cache:
                        self.global_cache.update(smi, scored_for_cache)
                    candidates = [smi]
                    seen = {smi}
                    for c in cands:
                        if c not in seen:
                            candidates.append(c)
                            seen.add(c)
                    all_candidates[idx] = candidates

            for i in range(n_mols):
                if all_candidates[i] is None:
                    all_candidates[i] = [current_smiles[i]]

            # 2. Select actions (epsilon-greedy)
            selected_smiles = []
            selected_obs = []
            for i in range(n_mols):
                candidates = all_candidates[i]
                obs_list = [make_observation(c, step_frac)
                            for c in candidates]
                obs_batch = torch.stack(obs_list)

                if random.random() < self.eps:
                    idx = random.randrange(len(candidates))
                else:
                    with torch.no_grad():
                        q_values = self.dqn(obs_batch).squeeze(-1)
                        idx = q_values.argmax().item()

                selected_smiles.append(candidates[idx])
                selected_obs.append(obs_list[idx])
                all_synthesis[i].append(
                    self.synthesis_map.get(candidates[idx], ''))

            # 3. Parse mols once — reuse for docking + reward
            selected_mols = [Chem.MolFromSmiles(smi) for smi in selected_smiles]

            # 3a. Batch dock all selected molecules (skip for QED-only)
            if self.dock_scorer_main is not None:
                dock_results = self.dock_scorer_main.batch_dock(
                    selected_smiles, mols=selected_mols)
            else:
                dock_results = [0.0] * n_mols

            # 3b. Batch ADMET prediction
            admet_batch = {}
            if cfg.reward.name == 'admet':
                admet_model = _get_admet_model()
                batch_df = admet_model.predict_properties(selected_smiles)
                if hasattr(batch_df, 'iloc'):
                    for j in range(len(selected_smiles)):
                        admet_batch[j] = batch_df.iloc[j].to_dict()
                else:
                    admet_batch[0] = batch_df

            # 4. Compute rewards
            for i in range(n_mols):
                smi = selected_smiles[i]
                dock_val = dock_results[i]

                rdict = compute_reward(
                    smi, step, cfg.max_steps, cfg.gamma,
                    cfg_reward=cfg.reward,
                    dock_scorer=None,
                    dock_score=dock_val,
                    admet_preds=admet_batch.get(i),
                    mol=selected_mols[i])
                all_rewards[i].append(rdict['reward'])
                score_val = dock_val if self.is_dock else rdict.get('qed', 0.0)
                all_scores[i].append(score_val)

                # Delayed storage
                if prev_obs[i] is not None:
                    all_transitions[i].append({
                        'obs': prev_obs[i].cpu(),
                        'reward': prev_reward[i],
                        'next_obs': selected_obs[i].cpu(),
                        'done': False,
                    })

                prev_obs[i] = selected_obs[i]
                prev_reward[i] = rdict['reward']
                paths[i].append(smi)

            current_smiles = selected_smiles

        # Terminal transitions
        for i in range(n_mols):
            if prev_obs[i] is not None:
                all_transitions[i].append({
                    'obs': prev_obs[i].cpu(),
                    'reward': prev_reward[i],
                    'next_obs': None,
                    'done': True,
                })

        self.dqn.to(self.device)
        self.target_dqn.to(self.device)

        # Build results
        results = []
        for i in range(n_mols):
            results.append({
                'transitions': all_transitions[i],
                'path': paths[i],
                'rewards': all_rewards[i],
                'scores': all_scores[i],
                'synthesis': all_synthesis[i],
                'final_smiles': current_smiles[i],
                'init_smiles': self.all_smiles[i],
                'cache_updates': {},
                'reasyn_calls': 0,
                'cache_hits': 0,
            })
        return results

    def _collect_async(self, episode):
        """Async worker or sequential collection (QED mode)."""
        cfg = self.cfg
        results = []

        if self.num_workers > 0:
            # Save DQN weights for workers
            weight_path = os.path.join(self.weight_dir, "dqn_weights.pth")
            tmp_path = weight_path + ".tmp"
            torch.save({
                'dqn': self.dqn.cpu().state_dict(),
                'target_dqn': self.target_dqn.cpu().state_dict(),
            }, tmp_path)
            os.replace(tmp_path, weight_path)
            self.dqn.to(self.device)
            self.target_dqn.to(self.device)

            # Dispatch
            for smi in self.all_smiles:
                self.task_queue.put((smi, self.eps))
            for _ in range(len(self.all_smiles)):
                results.append(self.result_queue.get())
        else:
            # Sequential
            for smi in self.all_smiles:
                result = run_episode(
                    smi, self.seq_models, self.seq_fpindex,
                    self.seq_rxn_matrix,
                    self.dqn.cpu(), self.target_dqn.cpu(), self.eps,
                    cfg.max_steps, cfg.gamma, cfg.method, cfg.reward,
                    device_str='cpu',
                    dock_scorer=None,
                    action_cache=(self.global_cache
                                  if self.global_cache else None),
                )
                results.append(result)
            self.dqn.to(self.device)
            self.target_dqn.to(self.device)

        return results

    def on_episode_end(self, episode, metrics, improved):
        """Append post-episode values to reasyn_history."""
        best_ever = (self.best_metric
                     if self.best_metric is not None else 0.0)
        self.reasyn_history['eps'].append(float(metrics['eps']))
        self.reasyn_history['time_s'].append(float(metrics['elapsed']))
        self.reasyn_history['best_score_ever'].append(float(best_ever))

    # ── Logging ───────────────────────────────────────────────────

    def log_episode(self, episode, metrics):
        mean_reward = metrics['reward']
        mean_score = metrics['score']
        ep_best = metrics.get('best_score')
        _ep_best = ep_best if ep_best is not None else 0.0
        best_ever = (self.best_metric
                     if self.best_metric is not None else 0.0)
        avg_loss = metrics['loss']

        cache_str = ""
        if self.global_cache is not None:
            results = metrics.get('results', [])
            if self.num_workers == 0:
                hr = self.global_cache.hit_rate()
                self.global_cache.stats = {
                    'hits': 0, 'misses': 0, 'explores': 0}
            else:
                total_hits = sum(r.get('cache_hits', 0) for r in results)
                total_calls = sum(r.get('reasyn_calls', 0) for r in results)
                total_all = total_hits + total_calls
                hr = total_hits / total_all if total_all > 0 else 0.0
            oracle_info = (f" | Orc:{self.dock_scorer_main.cache_size}"
                          if self.dock_scorer_main else "")
            cache_str = (f" | {hr:6.1%}  | "
                         f"{self.global_cache.num_molecules:4d} | "
                         f"{self.global_cache.num_edges:5d}{oracle_info}")

        if self.is_dock:
            print(f"{episode:4d} | {mean_reward:7.3f} | "
                  f"{mean_score:5.1f} | {_ep_best:6.1f} | "
                  f"{best_ever:7.1f} | {self.eps:.3f} | "
                  f"{avg_loss:7.4f} | {metrics['replay_size']:6d} | "
                  f"{metrics['elapsed']:5.1f}s{cache_str}")
        else:
            print(f"{episode:4d} | {mean_reward:7.3f} | "
                  f"{mean_score:.3f} | {_ep_best:6.3f} | "
                  f"{best_ever:7.3f} | {self.eps:.3f} | "
                  f"{avg_loss:7.4f} | {metrics['replay_size']:6d} | "
                  f"{metrics['elapsed']:5.1f}s{cache_str}")

    # ── Finalize ──────────────────────────────────────────────────

    def finalize(self):
        # Shutdown workers first (free GPU memory for full synthesis)
        self._shutdown_workers()

        # Full-mode synthesis routes for final molecules
        self._run_full_synthesis()

        # Save oracle counts to history
        self.reasyn_history['oracle_counts'] = self.oracle_counts

        # Final save
        save_pickle(self.exp_dir / f'{self.prefix}_history.pickle',
                     self.reasyn_history)
        save_pickle(self.exp_dir / f'{self.prefix}_paths.pickle',
                     self.best_paths)
        save_pickle(self.exp_dir / f'{self.prefix}_recent_episodes.pickle',
                     list(self.recent_episodes))

        # Print summary
        best_ever = (self.best_metric
                     if self.best_metric is not None else 0.0)
        print(f"\n{'='*70}")
        print(f"Training complete!")
        if self.is_dock:
            print(f"  Best dock score ever: {best_ever:.2f} kcal/mol")
        else:
            print(f"  Max QED ever: {best_ever:.3f}")
        print(f"  Final epsilon: {self.eps:.4f}")
        print(f"  Replay buffer: {len(self.replay)}")
        if self.global_cache is not None:
            print(f"  Cache: {self.global_cache.num_molecules} molecules, "
                  f"{self.global_cache.num_edges} edges")
        print(f"  Saved: {self.prefix}_history.pickle, "
              f"{self.prefix}_paths.pickle")

        if self.best_paths:
            print(f"\nTop-5 best paths:")
            for i, bp in enumerate(self.best_paths):
                if self.is_dock:
                    print(f"  {i+1}. Dock={bp['best_score']:.2f} | "
                          f"{bp['init_smiles'][:30]} -> "
                          f"{bp['final_smiles'][:30]} "
                          f"(ep {bp['episode']})")
                else:
                    print(f"  {i+1}. QED={bp['best_score']:.3f} | "
                          f"{bp['init_smiles'][:30]} -> "
                          f"{bp['final_smiles'][:30]} "
                          f"(ep {bp['episode']})")

        # ── Oracle consumption summary ───────────────────────────
        if self.oracle_counts:
            total_oracle = self.oracle_counts[-1]
            print(f"\n{'='*60}")
            print(f"Oracle Consumption")
            print(f"{'='*60}")
            print(f"  Total unique proxy calls: {total_oracle}")
            for ep_idx in [9, 24, 49, 74, 99, 199, 299, 399, 499]:
                if ep_idx < len(self.oracle_counts):
                    print(f"  After ep {ep_idx+1:3d}: "
                          f"{self.oracle_counts[ep_idx]} oracles")

        # ── Hypervolume computation ──────────────────────────────
        self._compute_hypervolume()

    def _compute_hypervolume(self):
        """Compute HV from all scored molecules using per-target proxy scores.

        4-obj: (GSK3B, JNK3, QED, SA_norm), ref=(0,0,0,0)
        2-obj: (GSK3B, JNK3), ref=(0,0)
        """
        if (not self.dock_scorer_main
                or not hasattr(self.dock_scorer_main, 'get_all_scored')):
            print("\n  [HV] No multi-target scorer available, skipping HV.")
            return

        try:
            from botorch.utils.multi_objective.hypervolume import Hypervolume
            from botorch.utils.multi_objective.pareto import is_non_dominated
            import torch as th
        except ImportError:
            print("\n  [HV] botorch not installed, skipping HV computation.")
            return

        per_target = self.dock_scorer_main.get_all_scored()
        if not per_target:
            print("\n  [HV] No scored molecules found.")
            return

        sascorer = _load_sascorer()

        points_4d = []
        points_2d = []
        for smi, scores in per_target.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            gsk3b = scores.get('gsk3b', 0.0)
            jnk3 = scores.get('jnk3', 0.0)
            qed = QEDModule.qed(mol)
            sa_raw = sascorer.calculateScore(mol)
            sa_norm = max(0.0, min(1.0, (10.0 - sa_raw) / 9.0))
            points_4d.append([gsk3b, jnk3, qed, sa_norm])
            points_2d.append([gsk3b, jnk3])

        if not points_4d:
            print("\n  [HV] No valid molecules for HV computation.")
            return

        pts_4d = th.tensor(points_4d)
        pts_2d = th.tensor(points_2d)

        ref_4d = th.zeros(4)
        hv4 = Hypervolume(ref_4d)
        pareto_4d = pts_4d[is_non_dominated(pts_4d)]
        hv_4d = hv4.compute(pareto_4d) if len(pareto_4d) > 0 else 0.0

        ref_2d = th.zeros(2)
        hv2 = Hypervolume(ref_2d)
        pareto_2d = pts_2d[is_non_dominated(pts_2d)]
        hv_2d = hv2.compute(pareto_2d) if len(pareto_2d) > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Hypervolume (ref=(0,...,0))")
        print(f"{'='*60}")
        print(f"  Total scored molecules: {len(points_4d)}")
        print(f"  4-obj Pareto front: {len(pareto_4d)} molecules")
        print(f"  4-obj HV: {hv_4d:.6f}")
        print(f"  2-obj Pareto front: {len(pareto_2d)} molecules")
        print(f"  2-obj HV: {hv_2d:.6f}")
        print(f"  (Baseline 4-obj: Genetic-GFN=0.642, HN-GFN=0.416)")
        print(f"  (Baseline 2-obj: Genetic-GFN=0.718, HN-GFN=0.669)")

        if len(pareto_4d) > 0:
            print(f"\n  4-obj Pareto extremes:")
            for dim, name in enumerate(['GSK3B', 'JNK3', 'QED', 'SA']):
                best_idx = pareto_4d[:, dim].argmax()
                vals = pareto_4d[best_idx]
                print(f"    Best {name}: [{vals[0]:.3f}, {vals[1]:.3f}, "
                      f"{vals[2]:.3f}, {vals[3]:.3f}]")

    def _run_full_synthesis(self):
        """Run Full ReaSyn (8x4 + editflow, fp32) on final molecules."""
        if not self.recent_episodes:
            return
        last_ep = self.recent_episodes[-1]
        final_smiles_set = set()
        for m in last_ep['molecules']:
            final_smiles_set.add(m['final_smiles'])
        final_smiles_list = sorted(final_smiles_set)
        if not final_smiles_list:
            return

        print(f"\n--- Full synthesis for {len(final_smiles_list)} "
              f"unique final molecules (8x4 + editflow) ---")

        # Load fp32 models
        print("  Loading ReaSyn models (fp32)...")
        from ..models.reasyn import ReaSyn as _ReaSyn
        _full_device = "cuda"
        full_models = []
        _reasyn_cfg = None
        for p in self.model_paths:
            _ckpt = torch.load(p, map_location="cpu")
            _reasyn_cfg = OmegaConf.create(
                _ckpt["hyper_parameters"]["config"])
            _m = _ReaSyn(_reasyn_cfg.model)
            _m.load_state_dict(
                {k[6:]: v for k, v in _ckpt["state_dict"].items()})
            _m = _m.to(_full_device)
            _m.eval()
            full_models.append(_m)

        model_dir = PROJECT_ROOT / self.cfg.method.model_dir
        if self.seq_fpindex is not None:
            full_fpindex = self.seq_fpindex
            full_rxn_matrix = self.seq_rxn_matrix
        else:
            _reasyn_root = model_dir.parent.parent
            full_fpindex = pickle.load(
                open(_reasyn_root / _reasyn_cfg.chem.fpindex, "rb"))
            full_rxn_matrix = pickle.load(
                open(_reasyn_root / _reasyn_cfg.chem.rxn_matrix, "rb"))
        print("  Models loaded.")

        full_synthesis_map = {}
        t0_all = time.perf_counter()
        for si, smi in enumerate(final_smiles_list):
            t0_full = time.perf_counter()
            try:
                scored = get_reasyn_actions_full(
                    smi, full_models, full_fpindex, full_rxn_matrix,
                    top_k=10)
            except Exception as e:
                print(f"  [{si+1}/{len(final_smiles_list)}] "
                      f"{smi[:40]}... ERROR: {e}")
                scored = []
            elapsed_full = time.perf_counter() - t0_full
            best_syn = ''
            best_steps = 0
            exact_syn = ''
            for s, sc, syn in scored:
                if not syn:
                    continue
                n_rxn = sum(1 for tok in syn.split(';') if tok.startswith('R'))
                if s == smi:
                    exact_syn = syn
                if n_rxn > best_steps:
                    best_syn = syn
                    best_steps = n_rxn
            chosen = exact_syn if exact_syn else best_syn
            full_synthesis_map[smi] = chosen
            n_chosen = (sum(1 for tok in chosen.split(';')
                            if tok.startswith('R'))
                        if chosen else 0)
            print(f"  [{si+1}/{len(final_smiles_list)}] "
                  f"{smi[:40]}... -> {len(scored)} prods, "
                  f"best={best_steps} steps, chosen={n_chosen}, "
                  f"{elapsed_full:.1f}s")
        total_full = time.perf_counter() - t0_all
        print(f"  Total Full synthesis time: {total_full:.0f}s "
              f"({total_full/60:.1f} min)")

        # Update recent_episodes and best_paths
        for m in last_ep['molecules']:
            fsmi = m['final_smiles']
            if fsmi in full_synthesis_map and full_synthesis_map[fsmi]:
                if m['synthesis']:
                    m['synthesis'][-1] = full_synthesis_map[fsmi]
                else:
                    m['synthesis'] = [full_synthesis_map[fsmi]]
                self.synthesis_map[fsmi] = full_synthesis_map[fsmi]

        for bp in self.best_paths:
            fsmi = bp['final_smiles']
            if fsmi in full_synthesis_map and full_synthesis_map[fsmi]:
                if bp['synthesis']:
                    bp['synthesis'][-1] = full_synthesis_map[fsmi]
                else:
                    bp['synthesis'] = [full_synthesis_map[fsmi]]

        del full_models
        torch.cuda.empty_cache()

        n_updated = sum(1 for v in full_synthesis_map.values() if v)
        print(f"  Full synthesis updated: {n_updated}/"
              f"{len(full_synthesis_map)} molecules")

    # ── Worker management ─────────────────────────────────────────

    def _shutdown_workers(self):
        """Shutdown all worker processes (idempotent)."""
        if self._workers_shutdown:
            return
        self._workers_shutdown = True

        if self.reasyn_action_workers:
            for _ in self.reasyn_action_workers:
                self.reasyn_action_queue.put(None)
            for w in self.reasyn_action_workers:
                w.join(timeout=10)
                if w.is_alive():
                    w.terminate()
            self.reasyn_action_workers = []

        if self.workers:
            for _ in self.workers:
                self.task_queue.put(None)
            self.task_queue.join()
            for w in self.workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()
            self.workers = []

    # ── Cleanup ───────────────────────────────────────────────────

    def cleanup(self):
        # Ensure workers are shut down (idempotent)
        self._shutdown_workers()
        # Clean temp dir
        if self.weight_dir:
            try:
                shutil.rmtree(self.weight_dir, ignore_errors=True)
            except Exception:
                pass
            self.weight_dir = None


# ── Backward-compatible entry point ──────────────────────────────────

def train_reasyn(cfg):
    """ReaSyn DQN training entry point."""
    ReaSynTrainer(cfg).train()
