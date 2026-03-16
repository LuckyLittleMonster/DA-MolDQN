"""Agent for UniMol-DQN and MolDQN baseline.

Key design:
  - Replay buffer stores pre-computed observation arrays (not SMILES)
  - Both UniMol and MolDQN share the same replay buffer + training loop
  - Embedding cache in UniMolDQN avoids redundant Uni-Mol calls at inference
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from collections import deque


class ReplayBuffer:
    """Replay buffer storing pre-computed observations.

    Each entry: (reward, done, observations)
    observations: np.ndarray (n_actions, obs_dim) — variable-length
    observations[-1] is the current state (allow_no_modification).
    """

    def __init__(self, capacity):
        self._storage = deque(maxlen=capacity)

    def add(self, reward, done, observations):
        self._storage.append((reward, done, observations))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        return [self._storage[i] for i in indices]

    def __len__(self):
        return len(self._storage)


class DQNAgent:
    """Unified DQN agent supporting both UniMolDQN and MolDQN.

    Both modes store pre-computed observation arrays in the replay buffer
    and share the same training loop.
    """

    def __init__(self, model, model_type='unimol', device='cpu',
                 lr=1e-4, gamma=0.95, polyak=0.995,
                 replay_size=5000, max_batch_size=128,
                 fp_radius=3, fp_length=2048,
                 max_grad_norm=1.0, double_dqn=False):
        self.model_type = model_type
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn

        if model_type == 'unimol':
            self.dqn = model
            self.dqn.q_head = self.dqn.q_head.to(device)
            self.target_dqn = copy.deepcopy(model)
            self.target_dqn.q_head = self.target_dqn.q_head.to(device)
            for p in self.target_dqn.q_head.parameters():
                p.requires_grad = False
        else:
            self.dqn = model.to(device)
            self.target_dqn = copy.deepcopy(model).to(device)
            for p in self.target_dqn.parameters():
                p.requires_grad = False

        self.replay_buffer = ReplayBuffer(replay_size)

        self.optimizer = torch.optim.Adam(
            [p for p in self.dqn.parameters() if p.requires_grad], lr=lr
        )
        self.gamma = gamma
        self.polyak = polyak
        self.max_batch_size = max_batch_size
        self.fp_radius = fp_radius
        self.fp_length = fp_length

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, valid_actions, step_fraction, epsilon, conformers=None):
        """Select action via epsilon-greedy.

        Always computes observations (needed for replay buffer storage).

        Args:
            valid_actions: list of RDKit Mol objects
            step_fraction: float
            epsilon: exploration rate
            conformers: optional list of (atoms, coords) tuples for UniMol
                        (from ConformerManager.generate_conformer)

        Returns:
            (action_index, is_greedy, observations_np)
        """
        n_actions = len(valid_actions)
        action_smiles = [Chem.MolToSmiles(m) for m in valid_actions]

        # Always compute observations for replay buffer
        if self.model_type == 'unimol':
            obs = self.dqn.encode_actions(
                action_smiles, step_fraction, conformers=conformers
            )
        else:
            obs = self._compute_fp_observations(action_smiles, step_fraction)

        if np.random.uniform() < epsilon:
            idx = np.random.randint(0, n_actions)
            return idx, False, obs
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.dqn(obs_t).squeeze(-1)
            idx = q_values.argmax().item()
            return idx, True, obs

    def store_transition(self, reward, done, observations, step_fraction=None):
        """Store transition in replay buffer."""
        self.replay_buffer.add(reward, done, observations)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self):
        """Sample from replay buffer and do one DQN update. Returns loss."""
        buf_len = len(self.replay_buffer)
        if buf_len < self.max_batch_size:
            return None

        batch_size = min(buf_len, self.max_batch_size)
        batch = self.replay_buffer.sample(batch_size)

        states = []
        next_states_list = []
        rewards = []
        dones = []

        for reward, done, observations in batch:
            # observations[-1] = current state (no-modification action)
            states.append(observations[-1])
            next_states_list.append(observations)
            rewards.append(reward)
            dones.append(done)

        # Batch Q-values for current states
        states_t = torch.tensor(
            np.stack(states), dtype=torch.float32, device=self.device
        )
        q_values = self.dqn(states_t).squeeze(-1)

        # Next state Q-values
        max_next_q = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            for i, obs in enumerate(next_states_list):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if self.double_dqn:
                    # Double DQN: online selects action, target evaluates
                    best_idx = self.dqn(obs_t).squeeze(-1).argmax()
                    max_next_q[i] = self.target_dqn(obs_t).squeeze(-1)[best_idx]
                else:
                    max_next_q[i] = self.target_dqn(obs_t).squeeze(-1).max()

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        target = rewards_t + self.gamma * (1 - dones_t) * max_next_q
        td_error = q_values - target

        # Huber loss
        loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error ** 2,
            torch.abs(td_error) - 0.5,
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents large updates from exploding Q-values
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.dqn.parameters() if p.requires_grad],
                self.max_grad_norm
            )
        self.optimizer.step()
        self._update_target()

        return loss.item()

    def _update_target(self):
        """Polyak average update of target network."""
        if self.model_type == 'unimol':
            params = zip(self.dqn.q_head.parameters(), self.target_dqn.q_head.parameters())
        else:
            params = zip(self.dqn.parameters(), self.target_dqn.parameters())

        with torch.no_grad():
            for p, p_targ in params:
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_fp_observations(self, smiles_list, step_fraction):
        """Compute Morgan FP observations for MolDQN baseline."""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_length)
            arr = np.zeros(self.fp_length, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(np.append(arr, step_fraction))
        return np.stack(fps)

    def get_epsilon(self, step, eps_start=1.0, eps_end=0.01, eps_decay=2000):
        """Exponential epsilon decay."""
        return eps_end + (eps_start - eps_end) * np.exp(-step / eps_decay)

    def save(self, path):
        if self.model_type == 'unimol':
            torch.save(self.dqn.q_head.state_dict(), path)
        else:
            torch.save(self.dqn.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if self.model_type == 'unimol':
            self.dqn.q_head.load_state_dict(state)
            self.target_dqn.q_head.load_state_dict(state)
        else:
            self.dqn.load_state_dict(state)
            self.target_dqn.load_state_dict(state)
