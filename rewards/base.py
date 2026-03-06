from abc import ABC, abstractmethod


class RewardCalculator(ABC):
    @abstractmethod
    def compute(self, molecules, **kwargs):
        """Compute rewards for current molecules. Returns {key: [values], 'reward': [weighted_sum]}."""

    def compute_overlap(self, molecules, prefetch_fn=None, cleanup_fn=None, **kwargs):
        """Compute with GPU/CPU overlap (default falls back to compute)."""
        if cleanup_fn:
            cleanup_fn()
        reward = self.compute(molecules)
        prefetch_result = prefetch_fn() if prefetch_fn else None
        return reward, prefetch_result

    @property
    @abstractmethod
    def reward_keys(self):
        """List of reward component names, e.g. ['bde', 'ip', 'rrab']."""
