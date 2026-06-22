"""Abstract cache interface for property-prediction memoization."""
from abc import ABC, abstractmethod


class Cache(ABC):
    """Key -> value cache for memoizing pure predictions.

    Implementations are swappable (``LRUCache`` today; future: TTL, persistent,
    cross-rank shared) — inject a different ``Cache`` into a ``CachedPredictor``
    without touching the predictor.
    """

    @abstractmethod
    def get(self, key):
        """Return ``(value, hit)``. On a miss ``hit`` is False (value ignored)."""

    @abstractmethod
    def put(self, key, value):
        """Insert / update ``key`` -> ``value``."""

    @abstractmethod
    def hit_rate(self, episode=False):
        """Cumulative (or per-episode if ``episode``) hit rate in [0, 1]."""

    def reset_episode_hit_rate(self):
        """Reset the per-episode hit/total counters. Optional; default no-op."""
