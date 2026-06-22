"""Cache interface as a structural Protocol for property-prediction memoization."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Key -> value cache. Structural typing: any object exposing these methods
    *is* a ``Cache`` — no inheritance required. Swap the algorithm (``LRUCache``
    today; future: TTL, persistent, cross-rank) by passing a different object.
    """

    def get(self, key) -> tuple[object, bool]:
        """Return ``(value, hit)``. On a miss ``hit`` is False (value ignored)."""
        ...

    def put(self, key, value) -> None:
        """Insert / update ``key`` -> ``value``."""
        ...

    def hit_rate(self, episode: bool = False) -> float:
        """Cumulative (or per-episode if ``episode``) hit rate in [0, 1]."""
        ...

    def reset_episode_hit_rate(self) -> None:
        """Reset the per-episode hit/total counters."""
        ...
