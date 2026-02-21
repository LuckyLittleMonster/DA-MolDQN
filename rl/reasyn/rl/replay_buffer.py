"""Replay buffer for ReaSyn DQN training."""

import random
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, reward, next_obs, done):
        """Store transition. next_obs is the selected next-step observation
        (single tensor), or None for terminal transitions."""
        self.buffer.append((obs, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, rewards, next_obs_list, dones = zip(*batch)
        # For terminal states (next_obs=None), use zeros with same shape as obs
        obs_dim = obs[0].shape[0]
        next_obs_tensors = [
            no if no is not None else torch.zeros(obs_dim)
            for no in next_obs_list
        ]
        return (
            torch.stack(obs),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_obs_tensors),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
