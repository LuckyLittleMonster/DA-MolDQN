"""Experience replay buffer for DQN training."""

import random

import numpy as np
import torch

from src.config import ENV_DEFAULTS
from src.features.fingerprints import get_observations_from_list


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.setseed = True

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, next_states, rewards, dones = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            fingerprints, steps, reward, done, obs_t, obs_tp1 = data

            if self.use_cxx_incremental_fingerprint == 0:
                states.append(obs_t)
                next_states.append(obs_tp1)
            elif self.use_cxx_incremental_fingerprint == 1:
                states.append(get_observations_from_list(fingerprints[-1], steps))
                observations = np.vstack([get_observations_from_list(fp, steps) for fp in fingerprints])
                next_states.append(torch.FloatTensor(observations).reshape(-1, ENV_DEFAULTS.fingerprint_length + 1))
            elif self.use_cxx_incremental_fingerprint == 2:
                states.append(obs_t)
                next_states.append(obs_tp1)

            rewards.append(reward)
            dones.append(done)

        return states, next_states, rewards, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        states, next_states, rewards, dones
        """
        if self.setseed :
            random.seed(0)
            self.setseed = False
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return [self._storage[i] for i in idxes]
