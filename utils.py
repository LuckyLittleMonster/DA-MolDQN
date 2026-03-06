"""Utility functions for DA-MolDQN."""

import numpy as np
import random
from collections import OrderedDict

# Fingerprint generator — lazily initialized via init_fingerprint_gen()
_FP_LENGTH = 2048

def init_fingerprint_gen(fingerprint_length=2048, fingerprint_radius=3):
    """Initialize the module-level Morgan fingerprint generator."""
    global _FP_LENGTH
    _FP_LENGTH = fingerprint_length


def get_observations_from_list(fp_list, remaining_steps):
    a = np.zeros(_FP_LENGTH + 1, dtype='uint8')
    for f in fp_list:
        a[f] = 1
    a[-1] = remaining_steps
    return a


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return [self._storage[i] for i in idxes]


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hit_count = 0
        self.total_count = 0
        self.hit_count_episode = 0
        self.total_count_episode = 0

    def get(self, key):
        self.total_count += 1
        self.total_count_episode += 1
        if key not in self.cache:
            return -1000, False
        else:
            self.hit_count += 1
            self.hit_count_episode += 1
            self.cache.move_to_end(key)
            return self.cache[key], True

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def hit_rate(self, episode=False):
        if episode:
            if self.total_count_episode == 0:
                return 0.0
            return self.hit_count_episode / self.total_count_episode
        else:
            if self.total_count == 0:
                return 0.0
            return self.hit_count / self.total_count

    def reset_episode_hit_rate(self):
        self.hit_count_episode = 0
        self.total_count_episode = 0
