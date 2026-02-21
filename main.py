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

import os
import pathlib
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Trainer registry — maps method name → RLTrainer subclass (lazy import)
# ===========================================================================

_TRAINERS = {
    'route': 'rl.route.train.RouteTrainer',
    'reasyn': 'rl.reasyn.rl.train.ReaSynTrainer',
}


def _get_trainer_cls(method: str):
    """Lazy-import and return the RLTrainer subclass for *method*."""
    path = _TRAINERS.get(method)
    if path is None:
        avail = ', '.join(sorted(_TRAINERS))
        raise ValueError(f"Unknown method: {method}. Available: {avail}")
    module_path, cls_name = path.rsplit('.', 1)
    from importlib import import_module
    mod = import_module(module_path)
    return getattr(mod, cls_name)


def train(cfg):
    """Instantiate the appropriate RLTrainer subclass and run training."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainer_cls = _get_trainer_cls(cfg.method.name)
    trainer = trainer_cls(cfg)
    trainer.train()


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
