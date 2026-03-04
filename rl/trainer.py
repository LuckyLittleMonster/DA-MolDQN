"""Base RLTrainer class for DQN-based molecular optimization.

RouteTrainer and ReaSynTrainer inherit from this class, sharing:
- Device/directory/epsilon setup
- Training loop skeleton (episode iteration, timing, epsilon decay)
- Soft target network update
- Checkpoint save/load triggers
- History tracking
"""

import os
import pathlib
import time
from abc import ABC, abstractmethod

import torch
from omegaconf import OmegaConf

from training_utils import (
    resolve_device, setup_experiment_dirs, save_checkpoint, save_pickle,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class RLTrainer(ABC):
    """Base DQN trainer for molecular optimization.

    Subclasses implement setup(), run_episode(), log_episode(), finalize(),
    and cleanup().  The shared train() loop handles epsilon decay, checkpoint
    management, and history tracking.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.exp_dir, self.model_save_dir, self.prefix = (
            setup_experiment_dirs(PROJECT_ROOT, cfg))
        self.eps = cfg.eps_start
        self.best_metric = None  # subclass tracks its "best" value

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self):
        """Create models, environment, replay buffer, load molecules, etc."""

    @abstractmethod
    def run_episode(self, episode: int) -> dict:
        """Run one episode of collection + training.

        Returns a metrics dict.  Must contain at least:
          'reward': float  (mean episode reward)
        Subclass may add 'score', 'loss', 'replay_size', etc.
        """

    @abstractmethod
    def log_episode(self, episode: int, metrics: dict):
        """Print one line of episode summary."""

    @abstractmethod
    def finalize(self):
        """Print final training summary (top products, best paths, etc.)."""

    @abstractmethod
    def cleanup(self):
        """Release resources: worker pools, temp dirs, etc.

        Must be idempotent — may be called on error before all resources exist.
        """

    def get_checkpoint_state(self) -> dict:
        """Return model/optimizer state dicts for saving.

        Default implementation returns empty dict.  Override if using the
        base class save() method.
        """
        return {}

    def load_checkpoint_state(self, ckpt: dict):
        """Restore model/optimizer from loaded checkpoint dict.

        Default is a no-op.  Override if using the base class
        try_load_checkpoint() method.
        """

    def is_improvement(self, metrics: dict) -> bool:
        """Return True if this episode is a new best (triggers checkpoint).

        Default: track metrics['reward'] (maximize).
        Override for dock (minimize) etc.
        """
        reward = metrics.get('reward', 0.0)
        if self.best_metric is None or reward > self.best_metric:
            self.best_metric = reward
            return True
        return False

    def on_episode_end(self, episode: int, metrics: dict, improved: bool):
        """Called after each episode with final metrics (eps, elapsed set).

        Override to update custom tracking that depends on post-episode values.
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def soft_update(polyak: float, *network_pairs):
        """Polyak-average target networks.

        Usage::

            self.soft_update(cfg.polyak,
                             (dqn, target_dqn),
                             (encoder, target_encoder))
        """
        with torch.no_grad():
            for net, target_net in network_pairs:
                for p, tp in zip(net.parameters(), target_net.parameters()):
                    tp.data.mul_(polyak).add_(p.data, alpha=1 - polyak)

    def try_load_checkpoint(self):
        """Load checkpoint if cfg.load_checkpoint points to an existing file."""
        path = self.cfg.load_checkpoint
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device,
                              weights_only=False)
            self.load_checkpoint_state(ckpt)
            ep = ckpt.get('episode', '?')
            eps_val = ckpt.get('eps', None)
            if eps_val is not None:
                self.eps = eps_val
            print(f"  Loaded checkpoint: {path} (ep={ep})")

    def save(self, episode: int, history: list):
        """Atomic save: model checkpoint + history pickle.

        Subclasses typically override this to save in their own format.
        """
        if not self.cfg.get('no_save_checkpoint', False):
            state = self.get_checkpoint_state()
            state['episode'] = episode
            state['eps'] = self.eps
            state['best_metric'] = self.best_metric
            state['config'] = OmegaConf.to_container(self.cfg, resolve=True)
            save_checkpoint(
                self.model_save_dir / f'{self.prefix}_checkpoint.pth', **state)

        save_pickle(
            self.exp_dir / f'{self.prefix}_history.pickle',
            {'history': history,
             'config': OmegaConf.to_container(self.cfg, resolve=True)})

    def should_save(self, episode: int, improved: bool) -> bool:
        """Decide whether to save checkpoint this episode."""
        return (improved
                or (episode + 1) % self.cfg.save_freq == 0
                or episode == self.cfg.episodes - 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Shared training loop: setup → episodes → finalize → cleanup."""
        try:
            self.setup()
            self.try_load_checkpoint()

            # Finetune: override checkpoint eps/best with config values
            if self.cfg.get('finetune'):
                self.eps = self.cfg.eps_start
                self.best_metric = None
                print(f"  Finetune mode: eps reset to {self.eps}, best_metric cleared")

            history = []

            for episode in range(self.cfg.episodes):
                t0 = time.perf_counter()

                metrics = self.run_episode(episode)

                # Common bookkeeping
                elapsed = time.perf_counter() - t0
                self.eps *= self.cfg.eps_decay
                metrics['elapsed'] = elapsed
                metrics['eps'] = self.eps
                history.append(metrics)

                improved = self.is_improvement(metrics)
                self.on_episode_end(episode, metrics, improved)

                if (episode + 1) % self.cfg.log_freq == 0:
                    self.log_episode(episode, metrics)

                if self.should_save(episode, improved):
                    self.save(episode, history)

            self.finalize()
        finally:
            self.cleanup()
