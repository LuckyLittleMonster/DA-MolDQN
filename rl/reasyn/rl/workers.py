"""ReaSyn RL worker processes."""

import multiprocessing as _mp

# Workers need forkserver to avoid fork+CUDA deadlock (Python 3.13+).
# Must be called before creating any Queue or Process objects.
try:
    _mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

import os
import pathlib
import pickle

import torch
from omegaconf import OmegaConf

from dqn import MLPDQN
from .action_cache import IncrementalActionCache
from .episode import run_episode
from .actions import get_reasyn_actions

# sklearn compat: fpindex.pkl was pickled with older sklearn where ManhattanDistance
# existed; in sklearn >=1.8 it was renamed to ManhattanDistance64.
try:
    import sklearn.metrics._dist_metrics as _dm
    if not hasattr(_dm, 'ManhattanDistance'):
        _dm.ManhattanDistance = _dm.ManhattanDistance64
except Exception:
    pass


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

        from ..chem.fpindex import FingerprintIndex
        from ..chem.matrix import ReactantReactionMatrix
        from ..models.reasyn import ReaSyn

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
        # -> parent.parent.parent = .../refs/ReaSyn
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

        # Dock scorer (proxy or UniDock)
        dock_scorer = None
        scoring_method = cfg_reward.get('scoring_method', 'dock')
        if scoring_method == 'proxy':
            target = cfg_reward.get('target', None)
            if target is not None:
                from reward.docking_score.proxy import ProxyDockAdapter
                dock_scorer = ProxyDockAdapter(target, device='cuda')
        elif self.dock_config is not None:
            from reward.docking_score.unidock import UniDockScorer
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


class ReaSynActionWorker(_mp.Process):
    """Lightweight worker: generates ReaSyn actions only (no DQN, no docking).

    Used in sync-dock mode so that ReaSyn inference runs in parallel across
    workers while the main process handles DQN selection and batch docking.
    """

    def __init__(self, model_paths, task_queue, result_queue, cfg_method_dict):
        super().__init__(daemon=True)
        self.model_paths = model_paths
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cfg_method_dict = cfg_method_dict

    def run(self):
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        from omegaconf import OmegaConf
        from ..models.reasyn import ReaSyn

        cfg_method = OmegaConf.create(self.cfg_method_dict)
        device = "cuda"

        # Load models (fp16)
        models = []
        reasyn_config = None
        for p in self.model_paths:
            ckpt = torch.load(p, map_location="cpu")
            reasyn_config = OmegaConf.create(
                ckpt["hyper_parameters"]["config"])
            m = ReaSyn(reasyn_config.model)
            m.load_state_dict(
                {k[6:]: v for k, v in ckpt["state_dict"].items()})
            m = m.half().to(device)
            m.eval()
            models.append(m)

        _model_dir = pathlib.Path(self.model_paths[0]).parent
        _reasyn_root = _model_dir.parent.parent
        fpindex = pickle.load(
            open(_reasyn_root / reasyn_config.chem.fpindex, "rb"))
        rxn_matrix = pickle.load(
            open(_reasyn_root / reasyn_config.chem.rxn_matrix, "rb"))

        # Signal ready
        self.result_queue.put(('ready', None, None))

        while True:
            item = self.task_queue.get()
            if item is None:
                break
            idx, smiles = item
            try:
                scored = get_reasyn_actions(
                    smiles, models, fpindex, rxn_matrix,
                    cfg_method, top_k=cfg_method.top_k,
                    return_scores=True,
                )
            except Exception:
                scored = []
            self.result_queue.put((idx, smiles, scored))
