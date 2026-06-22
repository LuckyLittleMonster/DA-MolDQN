"""Swappable caching for property-prediction memoization.

``Cache`` is the structural interface; ``LRUCache`` is the current implementation;
``cached`` wraps a pure predictor with dedup + an optional ``Cache``.
"""
from src.cache.base import Cache
from src.cache.lru import LRUCache
from src.cache.cached import cached

__all__ = ["Cache", "LRUCache", "cached"]
