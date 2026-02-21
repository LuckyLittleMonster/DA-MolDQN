"""Incremental action cache for ReaSyn RL training."""

import os
import pickle
import random


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
