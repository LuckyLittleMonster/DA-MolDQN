"""Unified training entry point for DA-MolDQN.

Supports two RL methods (Route-DQN and ReaSyn) and three reward modes
(QED, docking, multi-objective) via Hydra configuration.

Usage:
    # Route-DQN with QED (default)
    python main.py

    # Route-DQN smoke test
    python main.py +experiment=smoke

    # ReaSyn with docking
    python main.py method=reasyn reward=dock reward.target=3pbl

    # Route-DQN with multi-objective
    python main.py reward=multi reward.target=3pbl

    # Override individual params
    python main.py episodes=100 max_steps=3 method.decompose_method=aizynth
"""

# ReaSyn requires spawn for multiprocessing — must be set before any other
# multiprocessing usage.  Route-DQN uses fork (created explicitly later).
import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import hydra
from omegaconf import DictConfig, OmegaConf

import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import tempfile
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED as QEDModule, AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# sklearn compat: fpindex.pkl was pickled with older sklearn where ManhattanDistance
# existed; in sklearn >=1.8 it was renamed to ManhattanDistance64.
try:
    import sklearn.metrics._dist_metrics as _dm
    if not hasattr(_dm, 'ManhattanDistance'):
        _dm.ManhattanDistance = _dm.ManhattanDistance64
except Exception:
    pass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Ensure PROJECT_ROOT is on sys.path so spawned workers can find local packages
# (e.g., reasyn/, route/, template/, docking/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# SA Score helper (lazy loaded)
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


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------

def make_observation(smiles, step_frac):
    """Morgan FP (4096) + step_fraction -> tensor (4097,)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fp = np.zeros(4096, dtype=np.float32)
    else:
        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096),
            dtype=np.float32,
        )
    obs = np.append(fp, np.float32(step_frac))
    return torch.from_numpy(obs)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _check_lipinski(mol, cfg_reward):
    """Return True if mol passes Lipinski constraints from cfg."""
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = 0
    if mw > cfg_reward.max_mw:
        violations += 1
    if logp > cfg_reward.max_logp:
        violations += 1
    if hbd > cfg_reward.max_hbd:
        violations += 1
    if hba > cfg_reward.max_hba:
        violations += 1
    return violations <= 1  # Lipinski allows 1 violation


def compute_reward_qed(smiles, step, max_steps, gamma,
                       qed_weight=0.8, sa_weight=0.2):
    """QED-based reward. Returns (discounted_reward, qed, sa)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 10.0
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)
    raw = qed_weight * qed + sa_weight * sa_norm
    discount = gamma ** (max_steps - step - 1)
    return raw * discount, qed, sa


def compute_reward_dock(smiles, step, max_steps, gamma,
                        dock_scorer, dock_weight=0.7, sa_weight=0.2):
    """Docking-based reward. Returns (discounted_reward, dock_score, sa)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 10.0
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    scores = dock_scorer.batch_dock([smiles])
    dock_score = scores[0]
    dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
    raw = dock_weight * dock_norm + sa_weight * sa_norm
    discount = gamma ** (max_steps - step - 1)
    return raw * discount, dock_score, sa


def compute_reward_multi(smiles, step, max_steps, gamma,
                         cfg_reward, dock_scorer=None):
    """Multi-objective reward with layered filtering.

    Strategy: hard constraints -> soft penalties -> primary objective.
    Returns (discounted_reward, metrics_dict).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, {'qed': 0.0, 'sa': 10.0, 'dock': 0.0, 'valid': False}

    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)

    metrics = {'qed': qed, 'sa': sa, 'dock': 0.0, 'valid': True}

    # Layer 1: Hard constraints
    penalty = 0.0

    # SA threshold
    if sa > cfg_reward.sa_threshold:
        penalty += 0.3

    # Lipinski
    if cfg_reward.lipinski and not _check_lipinski(mol, cfg_reward):
        penalty += 0.2

    # Layer 2: Compute objectives
    dock_norm = 0.0
    if dock_scorer is not None:
        scores = dock_scorer.batch_dock([smiles])
        dock_score = scores[0]
        dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
        metrics['dock'] = dock_score

    # Layer 3: Scalarize
    primary = cfg_reward.get('primary', 'qed')
    if primary == 'dock' and dock_scorer is not None:
        raw = (cfg_reward.dock_weight * dock_norm +
               cfg_reward.sa_weight * sa_norm +
               cfg_reward.get('primary_weight', 0.6) * qed)
    else:
        raw = (cfg_reward.get('primary_weight', 0.6) * qed +
               cfg_reward.sa_weight * sa_norm)
        if dock_scorer is not None:
            raw += cfg_reward.dock_weight * dock_norm

    raw = max(0.0, raw - penalty)
    discount = gamma ** (max_steps - step - 1)
    return raw * discount, metrics


def compute_reward(smiles, step, max_steps, gamma, cfg_reward,
                   dock_scorer=None):
    """Unified reward dispatch based on cfg_reward.name.

    Returns (discounted_reward, primary_score, sa_score).
    For multi: primary_score is QED, sa_score is SA.
    """
    name = cfg_reward.name
    if name == 'qed':
        return compute_reward_qed(
            smiles, step, max_steps, gamma,
            qed_weight=cfg_reward.qed_weight,
            sa_weight=cfg_reward.sa_weight)
    elif name == 'dock':
        return compute_reward_dock(
            smiles, step, max_steps, gamma,
            dock_scorer=dock_scorer,
            dock_weight=cfg_reward.dock_weight,
            sa_weight=cfg_reward.sa_weight)
    elif name == 'multi':
        r, metrics = compute_reward_multi(
            smiles, step, max_steps, gamma,
            cfg_reward=cfg_reward,
            dock_scorer=dock_scorer)
        return r, metrics['qed'], metrics['sa']
    else:
        raise ValueError(f"Unknown reward mode: {name}")


# ---------------------------------------------------------------------------
# Docking scorer factory
# ---------------------------------------------------------------------------

def make_dock_scorer(cfg_reward, num_workers=4):
    """Create UniDockScorer from reward config. Returns None if not needed."""
    if cfg_reward.name == 'qed':
        return None
    target = cfg_reward.get('target', None)
    if target is None:
        if cfg_reward.name == 'dock':
            raise ValueError("reward.target must be set for dock mode")
        return None  # multi without dock

    target_dir = PROJECT_ROOT / 'docking' / 'targets' / target
    config_path = target_dir / 'config.json'
    receptor_path = target_dir / 'receptor.pdbqt'
    if not config_path.exists():
        raise FileNotFoundError(f"Target config not found: {config_path}")

    with open(config_path) as f:
        tgt_cfg = json.load(f)

    dock_config = {
        'receptor_pdbqt': str(receptor_path),
        'center_x': cfg_reward.get('center_x') or tgt_cfg['center_x'],
        'center_y': cfg_reward.get('center_y') or tgt_cfg['center_y'],
        'center_z': cfg_reward.get('center_z') or tgt_cfg['center_z'],
        'size_x': cfg_reward.get('size_x', 22.5),
        'size_y': cfg_reward.get('size_y', 22.5),
        'size_z': cfg_reward.get('size_z', 22.5),
    }

    from docking.dock_scorer import UniDockScorer
    scorer = UniDockScorer(num_workers=num_workers, **dock_config)
    print(f"  Dock scorer: target={target}, "
          f"center=({dock_config['center_x']}, "
          f"{dock_config['center_y']}, {dock_config['center_z']})")
    return scorer


# ---------------------------------------------------------------------------
# Checkpoint / History utilities
# ---------------------------------------------------------------------------

def save_checkpoint(path, **kwargs):
    """Atomic checkpoint save."""
    tmp = str(path) + '.tmp'
    torch.save(kwargs, tmp)
    os.replace(tmp, str(path))


def save_pickle(path, data):
    """Atomic pickle save."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# ReaSyn-specific: Inlined MLPDQN
# ---------------------------------------------------------------------------

class MLPDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# ReaSyn-specific: Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, reward, next_max_q, done):
        self.buffer.append((obs, reward, next_max_q, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, rewards, next_max_qs, dones = zip(*batch)
        return (
            torch.stack(obs),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_max_qs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# ReaSyn-specific: Incremental Action Cache
# ---------------------------------------------------------------------------

class IncrementalActionCache:
    """Cache ReaSyn-discovered neighbors for incremental action generation."""

    def __init__(self, min_cached_actions=5, max_neighbors=200, explore_prob=0.1):
        self.cache = {}
        self.min_cached_actions = min_cached_actions
        self.max_neighbors = max_neighbors
        self.explore_prob = explore_prob
        self.stats = {'hits': 0, 'misses': 0, 'explores': 0}

    def should_call_reasyn(self, smiles):
        cached = self.cache.get(smiles, [])
        if len(cached) < self.min_cached_actions:
            self.stats['misses'] += 1
            return True
        if random.random() < self.explore_prob:
            self.stats['explores'] += 1
            return True
        self.stats['hits'] += 1
        return False

    def get_actions(self, smiles, top_k=20):
        cached = self.cache.get(smiles, [])
        if not cached:
            return []
        sorted_c = sorted(cached, key=lambda x: x[1], reverse=True)
        return sorted_c[:top_k]

    def update(self, smiles, neighbors_with_scores):
        if smiles not in self.cache:
            self.cache[smiles] = []
        existing = {n for n, _ in self.cache[smiles]}
        new_entries = []
        for n, s in neighbors_with_scores:
            if n != smiles and n not in existing:
                self.cache[smiles].append((n, s))
                existing.add(n)
                new_entries.append((n, s))
        if len(self.cache[smiles]) > self.max_neighbors:
            self.cache[smiles].sort(key=lambda x: x[1], reverse=True)
            self.cache[smiles] = self.cache[smiles][:self.max_neighbors]
        return new_entries

    def merge(self, updates):
        for smiles, neighbors in updates.items():
            if smiles not in self.cache:
                self.cache[smiles] = []
            existing = {n for n, _ in self.cache[smiles]}
            for n, s in neighbors:
                if n != smiles and n not in existing:
                    self.cache[smiles].append((n, s))
                    existing.add(n)
            if len(self.cache[smiles]) > self.max_neighbors:
                self.cache[smiles].sort(key=lambda x: x[1], reverse=True)
                self.cache[smiles] = self.cache[smiles][:self.max_neighbors]

    def save(self, path):
        tmp = str(path) + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'cache': self.cache}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.merge(data['cache'])
        except (FileNotFoundError, EOFError, KeyError):
            pass

    @property
    def num_molecules(self):
        return len(self.cache)

    @property
    def num_edges(self):
        return sum(len(v) for v in self.cache.values())

    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses'] + self.stats['explores']
        return self.stats['hits'] / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# ReaSyn-specific: Action generation
# ---------------------------------------------------------------------------

def get_reasyn_actions(smiles, models, fpindex, rxn_matrix, cfg_method,
                       top_k=20, return_scores=False):
    """Run FastSampler L1a on one molecule, return deduplicated candidate SMILES."""
    from reasyn.chem.mol import Molecule
    from reasyn.sampler.sampler_fast import FastSampler
    from reasyn.utils.sample_utils import TimeLimit

    sampler_kwargs = {
        'use_fp16': cfg_method.use_fp16,
        'max_branch_states': cfg_method.max_branch_states,
        'skip_editflow': cfg_method.skip_editflow,
        'rxn_product_limit': cfg_method.rxn_product_limit,
    }
    evolve_kwargs = {
        'max_evolve_steps': cfg_method.max_evolve_steps,
        'num_cycles': cfg_method.num_cycles,
        'num_editflow_samples': cfg_method.num_editflow_samples,
        'num_editflow_steps': cfg_method.num_editflow_steps,
    }

    mol = Molecule(smiles)
    sampler = FastSampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix,
        mol=mol, model=models,
        factor=cfg_method.search_width,
        max_active_states=cfg_method.expansion_width,
        **sampler_kwargs,
    )
    tl = TimeLimit(30)
    sampler.evolve(gpu_lock=None, time_limit=tl, **evolve_kwargs)
    torch.cuda.synchronize()

    df = sampler.get_dataframe()
    if len(df) == 0:
        return []

    df = df.drop_duplicates(subset='smiles')
    df = df.sort_values('score', ascending=False).head(top_k)
    if return_scores:
        return list(zip(df['smiles'].tolist(), df['score'].tolist()))
    return df['smiles'].tolist()


# ---------------------------------------------------------------------------
# ReaSyn-specific: Episode runner
# ---------------------------------------------------------------------------

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
            reasyn_cands = [s for s, _ in scored]
            if action_cache is not None and scored:
                action_cache.update(current_smiles, scored)
                if current_smiles not in cache_updates:
                    cache_updates[current_smiles] = []
                cache_updates[current_smiles].extend(scored)

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

        # 5. Compute reward
        reward, score, sa = compute_reward(
            selected_smiles, step, max_steps, gamma, cfg_reward,
            dock_scorer=dock_scorer)
        rewards.append(reward)
        scores.append(score)

        # 6. Delayed storage
        if prev_obs is not None:
            with torch.no_grad():
                next_q = target_dqn(obs_batch).squeeze(-1)
                next_max_q = next_q.max().item()
            transitions.append({
                'obs': prev_obs.cpu(),
                'reward': prev_reward,
                'next_max_q': next_max_q,
                'done': False,
            })

        prev_obs = selected_obs
        prev_reward = reward
        current_smiles = selected_smiles
        path.append(current_smiles)

    # Terminal transition
    if prev_obs is not None:
        transitions.append({
            'obs': prev_obs.cpu(),
            'reward': prev_reward,
            'next_max_q': 0.0,
            'done': True,
        })

    return {
        'transitions': transitions,
        'path': path,
        'rewards': rewards,
        'scores': scores,
        'final_smiles': current_smiles,
        'init_smiles': smiles,
        'cache_updates': cache_updates,
        'reasyn_calls': reasyn_calls,
        'cache_hits': cache_hits,
    }


# ---------------------------------------------------------------------------
# ReaSyn-specific: Worker process
# ---------------------------------------------------------------------------

class RLWorker(_mp.Process):
    """Persistent worker: loads ReaSyn models once, runs episodes on demand."""

    def __init__(self, model_paths, task_queue, result_queue,
                 max_steps, gamma, top_k, weight_dir,
                 cfg_method_dict, cfg_reward_dict,
                 dock_config=None, cache_config=None):
        super().__init__(daemon=True)
        self.model_paths = model_paths
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.max_steps = max_steps
        self.gamma = gamma
        self.top_k = top_k
        self.weight_dir = weight_dir
        self.cfg_method_dict = cfg_method_dict
        self.cfg_reward_dict = cfg_reward_dict
        self.dock_config = dock_config
        self.cache_config = cache_config

    def run(self):
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        device = "cuda"

        # Reconstruct OmegaConf in worker process
        cfg_method = OmegaConf.create(self.cfg_method_dict)
        cfg_reward = OmegaConf.create(self.cfg_reward_dict)

        from reasyn.chem.fpindex import FingerprintIndex
        from reasyn.chem.matrix import ReactantReactionMatrix
        from reasyn.models.reasyn import ReaSyn

        # Load ReaSyn models (fp16)
        models = []
        config = None
        for p in self.model_paths:
            ckpt = torch.load(p, map_location="cpu")
            config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
            m = ReaSyn(config.model)
            m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
            m = m.half().to(device)
            m.eval()
            models.append(m)
        # fpindex/rxn_matrix paths in checkpoint are relative to ReaSyn root
        # model_path = .../refs/ReaSyn/data/trained_model/xxx.ckpt
        # → parent.parent.parent = .../refs/ReaSyn
        _reasyn_root = pathlib.Path(self.model_paths[0]).resolve().parent.parent.parent
        _fpindex_path = _reasyn_root / config.chem.fpindex
        _rxn_matrix_path = _reasyn_root / config.chem.rxn_matrix
        fpindex = pickle.load(open(_fpindex_path, "rb"))
        rxn_matrix = pickle.load(open(_rxn_matrix_path, "rb"))

        # Local DQN (CPU inference)
        input_dim = cfg_method.get('input_dim', 4097)
        hidden_dim = cfg_method.get('hidden_dim', 256)
        dqn = MLPDQN(input_dim, hidden_dim).to('cpu')
        target_dqn = MLPDQN(input_dim, hidden_dim).to('cpu')

        # Local action cache
        action_cache = None
        cache_mtime = 0.0
        if self.cache_config is not None:
            action_cache = IncrementalActionCache(**self.cache_config)

        # Dock scorer
        dock_scorer = None
        if self.dock_config is not None:
            from docking.dock_scorer import UniDockScorer
            dock_scorer = UniDockScorer(num_workers=0, **self.dock_config)

        while True:
            item = self.task_queue.get()
            if item is None:
                self.task_queue.task_done()
                break

            mol_smiles, eps = item

            # Load latest DQN weights
            weight_path = os.path.join(self.weight_dir, "dqn_weights.pth")
            if os.path.exists(weight_path):
                try:
                    state = torch.load(weight_path, map_location='cpu',
                                       weights_only=True)
                    dqn.load_state_dict(state['dqn'])
                    target_dqn.load_state_dict(state['target_dqn'])
                except Exception:
                    pass

            # Load latest global cache
            if action_cache is not None:
                cache_path = os.path.join(self.weight_dir, "action_cache.pkl")
                try:
                    mt = os.path.getmtime(cache_path)
                    if mt > cache_mtime:
                        action_cache.load(cache_path)
                        cache_mtime = mt
                except (FileNotFoundError, OSError):
                    pass

            try:
                result = run_episode(
                    mol_smiles, models, fpindex, rxn_matrix,
                    dqn, target_dqn, eps,
                    self.max_steps, self.gamma, cfg_method, cfg_reward,
                    device_str='cpu',
                    dock_scorer=dock_scorer,
                    action_cache=action_cache,
                )
                self.result_queue.put(result)
            except Exception as e:
                print(f"  Worker {self.name} ERROR on {mol_smiles[:30]}: {e}")
                self.result_queue.put({
                    'transitions': [],
                    'path': [mol_smiles],
                    'rewards': [],
                    'scores': [],
                    'final_smiles': mol_smiles,
                    'init_smiles': mol_smiles,
                    'cache_updates': {},
                    'reasyn_calls': 0,
                    'cache_hits': 0,
                    'error': str(e),
                })
            self.task_queue.task_done()


# ===========================================================================
# Route-DQN training
# ===========================================================================

def _route_to_dict(route, episode, route_idx):
    """Serialize a SynthesisRoute to a picklable dict."""
    mol = Chem.MolFromSmiles(route.final_product_smi)
    if mol is None:
        return None
    qed_val = QEDModule.qed(mol)
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


def train_route(cfg):
    """Route-DQN training loop."""
    print(f"\n{'='*60}")
    print(f"Route-DQN Training")
    print(f"  reward={cfg.reward.name}, episodes={cfg.episodes}, "
          f"max_steps={cfg.max_steps}")
    print(f"  decompose={cfg.method.decompose_method}, "
          f"batch_size={cfg.batch_size}")
    print(f"{'='*60}\n")

    # --- 1. Load template predictor (BEFORE CUDA / mp pool) ---
    print("--- Loading template predictor ---")
    from template.template_predictor import TemplateReactionPredictor

    _data_dir = PROJECT_ROOT / 'template' / 'data'

    if cfg.method.decompose_method == 'aizynth':
        bb_path = (cfg.method.bb_library
                   or str(_data_dir / 'building_blocks_merged.smi'))
        tp = TemplateReactionPredictor(
            template_path=str(_data_dir / 'templates_aizynth.txt'),
            building_block_path=bb_path,
            num_workers=0,
        )
    elif cfg.method.decompose_method == 'paroutes':
        bb_path = (cfg.method.bb_library
                   or str(_data_dir / 'building_blocks_paroutes.smi'))
        tp = TemplateReactionPredictor(
            template_path=str(_data_dir / 'templates_paroutes.txt'),
            building_block_path=bb_path,
            num_workers=0,
        )
    else:
        if cfg.method.bb_library:
            tp = TemplateReactionPredictor(
                building_block_path=cfg.method.bb_library, num_workers=0)
        else:
            tp = TemplateReactionPredictor(num_workers=0)
    tp.load()

    # Pre-build uni lookup cache (before forking workers)
    tp._uni_by_template_idx = {r.index: r for r in tp.uni_reactions}

    # --- 2. Create multiprocessing pool (BEFORE CUDA init) ---
    mp_pool = None
    cascade_workers = cfg.method.cascade_workers
    if cascade_workers > 0:
        import multiprocessing as mp
        import route.route as route_module
        route_module._mp_tp = tp
        ctx = mp.get_context('fork')
        mp_pool = ctx.Pool(cascade_workers)
        print(f"  MP pool: {cascade_workers} workers (fork)")

    # --- 3. Resolve device ---
    device = cfg.device
    if device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{cfg.gpu}'
    elif not torch.cuda.is_available():
        device = 'cpu'
    print(f"  Device: {device}")

    # --- 4. Load initial molecules ---
    if cfg.init_mol_path:
        with open(cfg.init_mol_path) as f:
            init_mols = [line.strip() for line in f if line.strip()]
    else:
        init_mols = list(cfg.init_mol)

    if cfg.num_molecules is not None:
        if cfg.num_molecules <= len(init_mols):
            init_mols = random.sample(init_mols, cfg.num_molecules)
        else:
            expanded = [init_mols[i % len(init_mols)]
                        for i in range(cfg.num_molecules)]
            init_mols = expanded

    print(f"  Init molecules ({len(init_mols)}): "
          f"{init_mols[:8]}{'...' if len(init_mols) > 8 else ''}")

    # --- 5. Generate initial routes ---
    print("\n--- Generating initial routes ---")
    from route.retro_decompose import (
        build_random_routes, decompose_molecule, build_retro_routes,
        build_routes_from_aizynth, build_routes_from_paroutes)

    t0 = time.perf_counter()
    all_routes = []

    if cfg.method.decompose_method == 'aizynth':
        aizynth_path = cfg.method.aizynth_path
        if aizynth_path is None:
            aizynth_path = str(PROJECT_ROOT / 'Experiments' /
                               'aizynth_zinc128_results.json')
        with open(aizynth_path) as f:
            aizynth_data = json.load(f)
        solved = [d for d in aizynth_data if d.get('is_solved', False)]
        n_mol = cfg.num_molecules or len(solved)
        solved = solved[:n_mol]
        print(f"  AiZynth: {len(solved)} solved molecules")
        all_routes = build_routes_from_aizynth(solved, tp)
    elif cfg.method.decompose_method == 'paroutes':
        paroutes_path = cfg.method.paroutes_path
        if paroutes_path is None:
            paroutes_path = str(PROJECT_ROOT / 'Data' / 'paroutes' /
                                'n1_routes_matched.json')
        with open(paroutes_path) as f:
            paroutes_data = json.load(f)
        n_mol = cfg.num_molecules or len(paroutes_data)
        paroutes_data = paroutes_data[:n_mol]
        print(f"  PaRoutes: {len(paroutes_data)} routes")
        all_routes = build_routes_from_paroutes(paroutes_data, tp)
    else:
        for mol_smi in init_mols:
            if cfg.method.decompose_method == 'random':
                routes = build_random_routes(
                    mol_smi, tp,
                    n_routes=cfg.method.routes_per_mol,
                    n_steps=cfg.method.route_steps,
                    seed=cfg.seed,
                )
            elif cfg.method.decompose_method == 'retro':
                routes = build_retro_routes(
                    mol_smi, tp,
                    max_depth=cfg.method.route_steps,
                    max_routes=cfg.method.routes_per_mol,
                    min_steps=2,
                    seed=cfg.seed,
                    verify=True,
                )
            else:
                routes = decompose_molecule(
                    mol_smi, tp,
                    max_depth=cfg.method.route_steps,
                    max_routes=cfg.method.routes_per_mol,
                )
            all_routes.extend(routes)

    elapsed = time.perf_counter() - t0

    if not all_routes:
        print("ERROR: No valid routes generated! Try different molecules.")
        if mp_pool is not None:
            mp_pool.close()
            mp_pool.join()
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

    # Dock scorer (for dock/multi reward modes)
    dock_scorer = make_dock_scorer(cfg.reward) if cfg.reward.name != 'qed' else None

    env = RouteEnvironment(
        routes=all_routes,
        tp=tp,
        max_steps=cfg.max_steps,
        discount=cfg.gamma,
        qed_weight=cfg.reward.get('qed_weight', 0.8),
        sa_weight=cfg.reward.sa_weight,
        dock_scorer=dock_scorer,
        dock_weight=cfg.reward.get('dock_weight', 0.7),
    )

    dqn = RouteDQN(
        tp=tp,
        device=device,
        fp_dim=cfg.method.fp_dim,
        template_emb_dim=cfg.method.template_emb_dim,
        block_emb_dim=cfg.method.block_emb_dim,
        route_emb_dim=cfg.method.route_emb_dim,
        max_route_len=cfg.method.max_route_len,
        hidden_dim=cfg.method.hidden_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        polyak=cfg.polyak,
        replay_size=cfg.replay_size,
        cascade_workers=cascade_workers,
        mp_pool=mp_pool,
    )

    if cfg.load_checkpoint and os.path.exists(cfg.load_checkpoint):
        dqn.load_checkpoint(cfg.load_checkpoint)

    # --- 7. Training loop ---
    print(f"\n{'='*60}")
    print(f"Training: {cfg.episodes} episodes, {env.n_routes} routes, "
          f"{cfg.max_steps} steps/ep")
    print(f"{'='*60}\n")

    eps = cfg.eps_start
    best_mean_reward = -float('inf')
    all_rewards = []
    all_qeds = []
    all_sas = []
    best_products = []
    route_top5 = {}
    last_n_episodes = 5
    last_episodes_buf = []

    phase_times = {
        'encode': 0.0, 'q_pos': 0.0, 'cascade': 0.0, 'q_bb': 0.0,
        'env_step': 0.0, 'next_encode': 0.0, 'store': 0.0, 'train': 0.0,
    }
    timing_steps = 0

    exp_dir = PROJECT_ROOT / 'Experiments'
    exp_dir.mkdir(exist_ok=True)
    model_save_dir = exp_dir / 'models'
    model_save_dir.mkdir(exist_ok=True)

    prefix = f"{cfg.exp_name}_{cfg.trial}"

    for episode in range(cfg.episodes):
        ep_start = time.perf_counter()

        routes = env.reset()
        episode_rewards = [0.0] * env.n_routes
        episode_qeds_final = [0.0] * env.n_routes

        for step in range(cfg.max_steps):
            batch_result = dqn.act_batch(
                routes, eps,
                cascade_top_k=cfg.method.get('cascade_topk', 64))
            positions = batch_result['positions']
            block_indices = batch_result['block_indices']
            valid_blocks_list = batch_result['valid_blocks_list']
            route_states_np = batch_result['route_states']
            pos_masks_np = batch_result['position_masks']
            pos_fps = batch_result['pos_fps']
            template_indices = batch_result['template_indices']

            for k, v in batch_result['timings'].items():
                phase_times[k] += v

            t0 = time.perf_counter()
            result = env.step(positions, block_indices)
            phase_times['env_step'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_route_states = dqn.encode_routes(routes).cpu().numpy()
            phase_times['next_encode'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            for i in range(len(routes)):
                next_pos_mask = np.zeros(cfg.method.max_route_len, dtype=bool)
                for p_idx, m in enumerate(routes[i].modifiable_mask):
                    if p_idx < cfg.method.max_route_len and m:
                        next_pos_mask[p_idx] = True

                transition = {
                    'route_state': route_states_np[i],
                    'position': positions[i],
                    'position_mask': pos_masks_np[i],
                    'block_idx': block_indices[i],
                    'block_mask': np.zeros(1, dtype=bool),
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

            t0 = time.perf_counter()
            n_updates = max(1, env.n_routes // cfg.batch_size)
            for _ in range(n_updates):
                dqn.train_step(cfg.batch_size)
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

            if i not in route_top5:
                route_top5[i] = []
            top_list = route_top5[i]
            if len(top_list) < 5 or qed_val > top_list[-1]['qed']:
                existing = {d['smiles'] for d in top_list}
                if rd['smiles'] not in existing:
                    top_list.append(rd)
                    top_list.sort(key=lambda x: x['qed'], reverse=True)
                    route_top5[i] = top_list[:5]

        last_episodes_buf.append({
            'episode': episode,
            'routes': ep_route_snapshots,
        })
        if len(last_episodes_buf) > last_n_episodes:
            last_episodes_buf.pop(0)

        # Epsilon decay
        eps = max(cfg.eps_min, eps * cfg.eps_decay)

        # Logging
        ep_time = time.perf_counter() - ep_start

        if (episode + 1) % cfg.log_freq == 0:
            n_success = sum(result['success'])
            print(f"Ep {episode+1:4d}/{cfg.episodes} | "
                  f"R: {mean_reward:.4f} | "
                  f"QED: {mean_qed:.3f} (max {max_qed:.3f}) | "
                  f"SA: {mean_sa:.2f} | "
                  f"Eps: {eps:.3f} | "
                  f"Success: {n_success}/{env.n_routes} | "
                  f"Buf: {len(dqn.replay_buffer)} | "
                  f"Time: {ep_time:.1f}s")

            for i in range(min(2, env.n_routes)):
                route = routes[i]
                print(f"  Route {i}: {route.init_mol_smi} -> "
                      f"{route.final_product_smi} "
                      f"(QED={episode_qeds_final[i]:.3f})")

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
                phase_times = {k: 0.0 for k in phase_times}
                timing_steps = 0

        # Save checkpoint
        if (episode + 1) % cfg.save_freq == 0 or mean_reward > best_mean_reward:
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

            ckpt_path = model_save_dir / f'{prefix}_checkpoint.pth'
            dqn.save(str(ckpt_path))

            best_products_sorted = sorted(
                best_products, key=lambda x: x['qed'], reverse=True)[:20]

            save_pickle(
                exp_dir / f'{prefix}_history.pickle',
                {
                    'rewards': all_rewards,
                    'qeds': all_qeds,
                    'sas': all_sas,
                    'best_products': best_products_sorted,
                    'route_top5': dict(route_top5),
                    'last_episodes': list(last_episodes_buf),
                    'config': OmegaConf.to_container(cfg, resolve=True),
                })

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"Training complete! Best mean reward: {best_mean_reward:.4f}")
    print(f"{'='*60}")

    best_products_sorted = sorted(
        best_products, key=lambda x: x['qed'], reverse=True)[:10]
    print("\nTop 10 products by QED:")
    for i, p in enumerate(best_products_sorted):
        print(f"  {i+1}. QED={p['qed']:.3f} | {p['smiles']} "
              f"(from {p['init_mol']}, ep {p['episode']})")

    print("\nFinal routes:")
    for i, route in enumerate(routes[:5]):
        mol = Chem.MolFromSmiles(route.final_product_smi)
        qed_val = QEDModule.qed(mol) if mol else 0
        print(f"  Route {i}: {route.init_mol_smi} -> "
              f"{route.final_product_smi} (QED={qed_val:.3f})")
        for j, step_obj in enumerate(route.steps):
            rxn_type = "uni" if step_obj.is_uni else "bi"
            bb_str = f" + {step_obj.block_smi}" if step_obj.block_smi else ""
            print(f"    Step {j}: [{rxn_type}] T{step_obj.template_idx}"
                  f"{bb_str} -> {step_obj.intermediate_smi}")

    print(f"\nPer-route top-5 products ({len(route_top5)} routes):")
    for ri in sorted(route_top5.keys())[:10]:
        tops = route_top5[ri]
        if tops:
            print(f"  Route {ri} ({tops[0]['init_mol']}):")
            for j, t in enumerate(tops):
                path_str = " -> ".join(
                    s['intermediate_smi'][:40] for s in t['steps'])
                print(f"    {j+1}. QED={t['qed']:.3f} ep{t['episode']:4d} | "
                      f"{t['smiles'][:60]}")
                print(f"       Path: {path_str[:120]}")

    dqn.close()
    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()


# ===========================================================================
# ReaSyn training
# ===========================================================================

def train_reasyn(cfg):
    """ReaSyn DQN training loop."""
    is_dock = cfg.reward.name in ('dock', 'multi')
    mode_str = (f"Docking({cfg.reward.target})" if cfg.reward.name == 'dock'
                else cfg.reward.name.upper())

    print(f"\n{'='*70}")
    print(f"ReaSyn DQN Training -- {mode_str}")
    print(f"  episodes={cfg.episodes}, max_steps={cfg.max_steps}, "
          f"workers={cfg.method.num_workers}")
    print(f"  gamma={cfg.gamma}, lr={cfg.lr}, top_k={cfg.method.top_k}")
    print(f"  eps: {cfg.eps_start} -> {cfg.eps_min} (decay={cfg.eps_decay})")
    print(f"{'='*70}\n")

    # Paths
    model_dir = PROJECT_ROOT / cfg.method.model_dir
    model_files = cfg.method.model_files
    model_paths = [model_dir / f for f in model_files]

    exp_dir = PROJECT_ROOT / 'Experiments'
    exp_dir.mkdir(exist_ok=True)
    model_save_dir = exp_dir / 'models'
    model_save_dir.mkdir(exist_ok=True)

    # Load molecules
    if cfg.mol_json:
        mol_json_path = pathlib.Path(cfg.mol_json)
        if not mol_json_path.is_absolute():
            mol_json_path = PROJECT_ROOT / mol_json_path
        with open(mol_json_path) as f:
            mol_data = json.load(f)
        num_mols = cfg.num_molecules or len(mol_data)
        all_smiles = [d['smiles'] for d in mol_data[:num_mols]]
        print(f"  Loaded {len(all_smiles)} molecules from {mol_json_path.name}")
    else:
        zinc_path = model_dir.parent / 'zinc_first64.txt'
        with open(zinc_path) as f:
            lines = f.read().strip().split("\n")
        if lines[0].upper() == "SMILES":
            lines = lines[1:]
        num_mols = cfg.num_molecules or 64
        all_smiles = [s.strip() for s in lines[:num_mols]]
        print(f"  Loaded {len(all_smiles)} molecules from {zinc_path.name}")

    # Docking setup
    dock_config = None
    dock_scorer_main = None
    if cfg.reward.name != 'qed' and cfg.reward.get('target', None) is not None:
        target = cfg.reward.target
        target_dir = PROJECT_ROOT / 'docking' / 'targets' / target
        config_path = target_dir / 'config.json'
        receptor_path = target_dir / 'receptor.pdbqt'
        if not config_path.exists():
            raise FileNotFoundError(f"Target config not found: {config_path}")
        with open(config_path) as f:
            tgt_cfg = json.load(f)
        dock_config = {
            'receptor_pdbqt': str(receptor_path),
            'center_x': cfg.reward.get('center_x') or tgt_cfg['center_x'],
            'center_y': cfg.reward.get('center_y') or tgt_cfg['center_y'],
            'center_z': cfg.reward.get('center_z') or tgt_cfg['center_z'],
            'size_x': cfg.reward.get('size_x', 22.5),
            'size_y': cfg.reward.get('size_y', 22.5),
            'size_z': cfg.reward.get('size_z', 22.5),
        }
        print(f"  Docking target: {target}")

    # DQN on main process
    device = torch.device(cfg.device)
    input_dim = cfg.method.get('input_dim', 4097)
    hidden_dim = cfg.method.get('hidden_dim', 256)
    dqn = MLPDQN(input_dim, hidden_dim).to(device)
    target_dqn = MLPDQN(input_dim, hidden_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    # Weight sharing directory
    weight_dir = tempfile.mkdtemp(prefix="reasyn_dqn_")
    print(f"  Weight sync dir: {weight_dir}")

    # Incremental action cache
    cache_config = None
    global_cache = None
    if cfg.method.use_cache:
        cache_config = {
            'min_cached_actions': cfg.method.min_cached_actions,
            'max_neighbors': cfg.method.max_neighbors,
            'explore_prob': cfg.method.explore_prob,
        }
        global_cache = IncrementalActionCache(**cache_config)
        if cfg.method.cache_path and os.path.exists(cfg.method.cache_path):
            global_cache.load(cfg.method.cache_path)
            print(f"  Loaded cache: {global_cache.num_molecules} mols, "
                  f"{global_cache.num_edges} edges")
        print(f"  Cache: explore_prob={cfg.method.explore_prob}, "
              f"min_actions={cfg.method.min_cached_actions}")

    # Serializable config dicts for workers
    cfg_method_dict = OmegaConf.to_container(cfg.method, resolve=True)
    cfg_reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)

    # Score label
    is_dock_score = cfg.reward.name == 'dock'
    score_label = "Dock" if is_dock_score else "QED"

    # Training history
    history = {
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
    best_paths = []
    best_score_ever = 0.0
    eps = cfg.eps_start
    prefix = f"{cfg.exp_name}_{cfg.trial}"

    num_workers = cfg.method.num_workers

    if num_workers > 0:
        task_queue = _mp.JoinableQueue()
        result_queue = _mp.Queue()
        workers = []
        for _ in range(num_workers):
            w = RLWorker(
                model_paths=[str(p) for p in model_paths],
                task_queue=task_queue,
                result_queue=result_queue,
                max_steps=cfg.max_steps,
                gamma=cfg.gamma,
                top_k=cfg.method.top_k,
                weight_dir=weight_dir,
                cfg_method_dict=cfg_method_dict,
                cfg_reward_dict=cfg_reward_dict,
                dock_config=dock_config,
                cache_config=cache_config,
            )
            w.start()
            workers.append(w)
        print(f"  Started {len(workers)} workers")
    else:
        # Sequential mode: load ReaSyn in main process
        print("  Sequential mode: loading ReaSyn models...")
        from reasyn.models.reasyn import ReaSyn

        seq_device = "cuda"
        seq_models = []
        reasyn_config = None
        for p in model_paths:
            ckpt = torch.load(p, map_location="cpu")
            reasyn_config = OmegaConf.create(
                ckpt["hyper_parameters"]["config"])
            m = ReaSyn(reasyn_config.model)
            m.load_state_dict(
                {k[6:]: v for k, v in ckpt["state_dict"].items()})
            m = m.half().to(seq_device)
            m.eval()
            seq_models.append(m)
        # fpindex/rxn_matrix paths in checkpoint are relative to ReaSyn root
        # model_dir = .../refs/ReaSyn/data/trained_model → parent.parent = .../refs/ReaSyn
        _reasyn_root = model_dir.parent.parent
        _fpindex_path = _reasyn_root / reasyn_config.chem.fpindex
        _rxn_matrix_path = _reasyn_root / reasyn_config.chem.rxn_matrix
        seq_fpindex = pickle.load(open(_fpindex_path, "rb"))
        seq_rxn_matrix = pickle.load(open(_rxn_matrix_path, "rb"))
        seq_action_cache = global_cache

        # Sequential dock scorer
        if dock_config is not None:
            from docking.dock_scorer import UniDockScorer
            dock_scorer_main = UniDockScorer(num_workers=4, **dock_config)
            print("  Dock scorer initialized.")
        print("  Models loaded.")

    # Print header
    cache_hdr = " | CacheHit | Mols | Edges" if global_cache else ""
    print(f"\n{'='*70}")
    print(f"{'Ep':>4s} | {'Reward':>7s} | {score_label:>5s} | {'Best':>6s} | "
          f"{'BstEvr':>7s} | {'Eps':>5s} | {'Loss':>7s} | {'Buf':>6s} | "
          f"{'Time':>6s}{cache_hdr}")
    print(f"{'-'*70}")

    for episode in range(cfg.episodes):
        t0_ep = time.perf_counter()

        # 1. Save DQN weights for workers
        if num_workers > 0:
            weight_path = os.path.join(weight_dir, "dqn_weights.pth")
            tmp_path = weight_path + ".tmp"
            torch.save({
                'dqn': dqn.cpu().state_dict(),
                'target_dqn': target_dqn.cpu().state_dict(),
            }, tmp_path)
            os.replace(tmp_path, weight_path)
            dqn.to(device)
            target_dqn.to(device)

        # 2. Collect episodes
        results = []
        if num_workers > 0:
            for smi in all_smiles:
                task_queue.put((smi, eps))
            for _ in range(len(all_smiles)):
                results.append(result_queue.get())
        else:
            for smi in all_smiles:
                result = run_episode(
                    smi, seq_models, seq_fpindex, seq_rxn_matrix,
                    dqn.cpu(), target_dqn.cpu(), eps,
                    cfg.max_steps, cfg.gamma, cfg.method, cfg.reward,
                    device_str='cpu',
                    dock_scorer=dock_scorer_main,
                    action_cache=seq_action_cache,
                )
                results.append(result)
            dqn.to(device)
            target_dqn.to(device)

        # 2b. Merge cache updates
        if global_cache is not None and num_workers > 0:
            for result in results:
                cu = result.get('cache_updates', {})
                if cu:
                    global_cache.merge(cu)
            cache_path = os.path.join(weight_dir, "action_cache.pkl")
            global_cache.save(cache_path)

        # 3. Add transitions to replay buffer
        n_transitions = 0
        ep_rewards = []
        ep_scores = []
        ep_best_score = None
        for result in results:
            for t in result['transitions']:
                replay.push(t['obs'], t['reward'], t['next_max_q'], t['done'])
                n_transitions += 1
            if result['rewards']:
                ep_rewards.append(sum(result['rewards']))
            if result['scores']:
                if is_dock_score:
                    best_s = min(result['scores'])
                else:
                    best_s = max(result['scores'])
                ep_scores.append(best_s)

                if ep_best_score is None:
                    ep_best_score = best_s
                elif is_dock_score:
                    ep_best_score = min(ep_best_score, best_s)
                else:
                    ep_best_score = max(ep_best_score, best_s)

                if is_dock_score:
                    if best_s < best_score_ever:
                        best_score_ever = best_s
                else:
                    if best_s > best_score_ever:
                        best_score_ever = best_s

                best_paths.append({
                    'episode': episode,
                    'path': result['path'],
                    'scores': result['scores'],
                    'rewards': result['rewards'],
                    'best_score': best_s,
                    'init_smiles': result['init_smiles'],
                    'final_smiles': result['final_smiles'],
                })

        best_paths.sort(key=lambda x: x['best_score'],
                        reverse=(not is_dock_score))
        best_paths = best_paths[:5]

        # 4. Gradient updates
        total_loss = 0.0
        n_updates = 0
        if len(replay) >= cfg.batch_size:
            n_grad = min(cfg.grad_steps, max(1, n_transitions))
            for _ in range(n_grad):
                obs, rew, next_max_q, done = replay.sample(cfg.batch_size)
                obs = obs.to(device)
                rew = rew.to(device)
                next_max_q = next_max_q.to(device)
                done = done.to(device)

                q_pred = dqn(obs).squeeze(-1)
                q_target = rew + cfg.gamma * next_max_q * (1 - done)
                loss = F.mse_loss(q_pred, q_target.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10.0)
                optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        # 5. Soft update target network
        with torch.no_grad():
            for p, tp in zip(dqn.parameters(), target_dqn.parameters()):
                tp.data.mul_(cfg.polyak).add_(p.data, alpha=1 - cfg.polyak)

        # 6. Decay epsilon
        eps = max(cfg.eps_min, eps * cfg.eps_decay)

        elapsed = time.perf_counter() - t0_ep
        mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_score = np.mean(ep_scores) if ep_scores else 0.0
        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0

        # Log
        history['episode'].append(episode)
        history['mean_reward'].append(float(mean_reward))
        history['mean_score'].append(float(mean_score))
        history['best_score'].append(
            float(ep_best_score if ep_best_score is not None else 0.0))
        history['best_score_ever'].append(float(best_score_ever))
        history['eps'].append(float(eps))
        history['time_s'].append(float(elapsed))
        history['replay_size'].append(len(replay))
        history['loss'].append(float(avg_loss))
        history['cache_mols'].append(
            global_cache.num_molecules if global_cache else 0)
        history['cache_edges'].append(
            global_cache.num_edges if global_cache else 0)

        cache_str = ""
        if global_cache is not None:
            if num_workers == 0:
                hr = global_cache.hit_rate()
                global_cache.stats = {'hits': 0, 'misses': 0, 'explores': 0}
            else:
                total_hits = sum(r.get('cache_hits', 0) for r in results)
                total_calls = sum(r.get('reasyn_calls', 0) for r in results)
                total_all = total_hits + total_calls
                hr = total_hits / total_all if total_all > 0 else 0.0
            cache_str = (f" | {hr:6.1%}  | {global_cache.num_molecules:4d} | "
                         f"{global_cache.num_edges:5d}")

        _ep_best = ep_best_score if ep_best_score is not None else 0.0
        if is_dock_score:
            print(f"{episode:4d} | {mean_reward:7.3f} | {mean_score:5.1f} | "
                  f"{_ep_best:6.1f} | {best_score_ever:7.1f} | "
                  f"{eps:.3f} | {avg_loss:7.4f} | {len(replay):6d} | "
                  f"{elapsed:5.1f}s{cache_str}")
        else:
            print(f"{episode:4d} | {mean_reward:7.3f} | {mean_score:.3f} | "
                  f"{_ep_best:6.3f} | {best_score_ever:7.3f} | "
                  f"{eps:.3f} | {avg_loss:7.4f} | {len(replay):6d} | "
                  f"{elapsed:5.1f}s{cache_str}")

        # Save checkpoints periodically
        if (episode + 1) % cfg.save_freq == 0 or episode == cfg.episodes - 1:
            save_checkpoint(
                model_save_dir / f'{prefix}_checkpoint.pth',
                dqn=dqn.state_dict(),
                target_dqn=target_dqn.state_dict(),
                optimizer=optimizer.state_dict(),
                episode=episode,
                eps=eps,
                best_score_ever=best_score_ever,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            save_pickle(exp_dir / f'{prefix}_history.pickle', history)
            save_pickle(exp_dir / f'{prefix}_paths.pickle', best_paths)

            if global_cache is not None:
                if cfg.method.cache_path:
                    global_cache.save(cfg.method.cache_path)
                global_cache.save(
                    str(exp_dir / f'{prefix}_cache.pickle'))

    # Shutdown workers
    if num_workers > 0:
        for _ in workers:
            task_queue.put(None)
        task_queue.join()
        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

    # Final save
    save_pickle(exp_dir / f'{prefix}_history.pickle', history)
    save_pickle(exp_dir / f'{prefix}_paths.pickle', best_paths)

    # Cleanup temp dir
    try:
        shutil.rmtree(weight_dir, ignore_errors=True)
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"Training complete!")
    if is_dock_score:
        print(f"  Best dock score ever: {best_score_ever:.2f} kcal/mol")
    else:
        print(f"  Max QED ever: {best_score_ever:.3f}")
    print(f"  Final epsilon: {eps:.4f}")
    print(f"  Replay buffer: {len(replay)}")
    if global_cache is not None:
        print(f"  Cache: {global_cache.num_molecules} molecules, "
              f"{global_cache.num_edges} edges")
    print(f"  Saved: {prefix}_history.pickle, {prefix}_paths.pickle")

    if best_paths:
        print(f"\nTop-5 best paths:")
        for i, bp in enumerate(best_paths):
            if is_dock_score:
                print(f"  {i+1}. Dock={bp['best_score']:.2f} | "
                      f"{bp['init_smiles'][:30]} -> {bp['final_smiles'][:30]} "
                      f"(ep {bp['episode']})")
            else:
                print(f"  {i+1}. QED={bp['best_score']:.3f} | "
                      f"{bp['init_smiles'][:30]} -> {bp['final_smiles'][:30]} "
                      f"(ep {bp['episode']})")


# ===========================================================================
# Dispatcher
# ===========================================================================

def train(cfg):
    """Dispatch to the appropriate training function based on method."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    method = cfg.method.name
    if method == 'route':
        train_route(cfg)
    elif method == 'reasyn':
        train_reasyn(cfg)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'route' or 'reasyn'.")


# ===========================================================================
# Entry point
# ===========================================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Hydra changes cwd to output dir; change back to project root
    os.chdir(hydra.utils.get_original_cwd())

    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    train(cfg)


if __name__ == "__main__":
    main()
