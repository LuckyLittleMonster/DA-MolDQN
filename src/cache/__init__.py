"""Swappable caching for property-prediction memoization.

``Cache`` is the abstract interface; ``LRUCache`` is the current implementation;
``CachedPredictor`` wraps a pure predictor with dedup + an optional ``Cache``.
"""
from src.cache.base import Cache
from src.cache.lru import LRUCache
from src.cache.cached_predictor import CachedPredictor

__all__ = ["Cache", "LRUCache", "CachedPredictor"]
