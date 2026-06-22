"""LRU cache implementation of the :class:`Cache` interface."""
from collections import OrderedDict


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
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count

    def reset_episode_hit_rate(self):
        self.hit_count_episode = 0
        self.total_count_episode = 0
