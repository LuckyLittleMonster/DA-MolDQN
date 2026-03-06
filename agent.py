import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import utils
from dqn import MolDQN

# Re-export for backward compatibility
from molecules import MultiMolecules


class DistributedAgent(object):
    """DQN agent with DDP support."""
    def __init__(self, input_length, gpu_index, device, args, rank):
        super(DistributedAgent, self).__init__()
        self.gpu_index = gpu_index
        self.device = device
        torch.cuda.set_device(gpu_index)

        self.max_batch_size = args.max_batch_size
        self.gamma = args.dqn.gamma
        self.polyak = args.dqn.polyak

        self.dqn = MolDQN(input_length, 1).to(self.device)
        self.target_dqn = MolDQN(input_length, 1).to(self.device)
        if args.checkpoint is not None:
            if rank == 0:
                dqn_checkpoint = torch.load(f'./Experiments/models/{args.checkpoint}_best_model_dqn.pth')
                dqn_model_state = dqn_checkpoint['model_state_dict']
                self.dqn.load_state_dict(dqn_model_state)

            target_dqn_checkpoint = torch.load(f'./Experiments/models/{args.checkpoint}_best_model_target_dqn.pth')
            target_dqn_model_state = target_dqn_checkpoint['model_state_dict']
            self.target_dqn.load_state_dict(target_dqn_model_state)
            self.eps_threshold = target_dqn_checkpoint['eps_threshold']

        self.dqn = nn.parallel.DistributedDataParallel(self.dqn,
            device_ids=[self.gpu_index],
            output_device = self.gpu_index)

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        n_mols = args.num_init_mol if args.num_init_mol is not None else 1
        self.replay_buffer = utils.ReplayBuffer(args.dqn.replay_buffer_size * n_mols)
        self.optimizer = getattr(opt, args.dqn.optimizer)(
            self.dqn.parameters(), lr=args.dqn.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):
        isGreedy = True

        if np.random.uniform() < epsilon_threshold:
            if isinstance(observations, list):
                al = observations[0].shape[0]
            else:
                al = observations.shape[0]
            action = np.random.randint(0, al)
            isGreedy = False
        else:
            q_value = self.dqn.forward(observations).cpu()
            action = torch.argmax(q_value).numpy()
        return action, isGreedy

    def training_step(self):
        batch_size = min(len(self.replay_buffer), self.max_batch_size)
        data_batch = self.replay_buffer.sample(batch_size)

        # --- Phase 1: Extract data from replay buffer ---
        rewards_np = np.empty(batch_size, dtype=np.float32)
        dones_np = np.empty(batch_size, dtype=np.float32)
        obs_list = [None] * batch_size
        next_splits = np.empty(batch_size, dtype=np.int64)

        for i, (reward, done, saved_obs) in enumerate(data_batch):
            obs = saved_obs if isinstance(saved_obs, np.ndarray) else np.asarray(saved_obs)
            obs_list[i] = obs
            next_splits[i] = obs.shape[0]
            rewards_np[i] = reward
            dones_np[i] = done

        # --- Phase 2: Single numpy concatenation ---
        next_np = np.vstack(obs_list)       # [total_actions, 2049] uint8
        cum_splits = np.cumsum(next_splits)
        states_np = next_np[cum_splits - 1]  # [batch_size, 2049] — last row per block

        # --- Phase 3: GPU transfer + forward ---
        states_t = torch.from_numpy(np.ascontiguousarray(states_np)).to(
            self.device, dtype=torch.float32)
        q = self.dqn(states_t)

        next_t = torch.from_numpy(next_np).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            all_q_next = self.target_dqn(next_t)  # [total_actions, 1]

        # --- Phase 4: Vectorized segment max ---
        splits_t = torch.from_numpy(next_splits).to(self.device)
        seg_ids = torch.arange(batch_size, device=self.device).repeat_interleave(splits_t)
        max_q = torch.full((batch_size,), float('-inf'), device=self.device)
        max_q.scatter_reduce_(0, seg_ids, all_q_next[:, 0], reduce='amax')
        max_q = max_q.unsqueeze(1)

        # --- Phase 5: Loss + backward + optimizer ---
        rewards_t = torch.from_numpy(rewards_np).to(self.device).reshape(q.shape)
        dones_t = torch.from_numpy(dones_np).to(self.device).reshape(q.shape)

        target = rewards_t + self.gamma * (1 - dones_t) * max_q
        td_error = q - target
        loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.distributed.barrier()
        self.optimizer.step()
        with torch.no_grad():
            for p, p_targ in zip(self.dqn.module.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss.item()
