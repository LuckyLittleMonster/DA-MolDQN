import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
from src import utils
from src.config import ObservationType
from src.models.dqn import MolDQN


class DistributedAgent(object):
    """docstring for DistributedAgent"""
    def __init__(self, input_length, gpu_index, device, config, rank):
        super(DistributedAgent, self).__init__()
        self.gpu_index = gpu_index
        self.device = device
        self.config = config
        self.gamma = config.train.gamma
        self.polyak = config.train.polyak
        # print(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(gpu_index)

        self.observation = config.env.observation
        self.max_batch_size = config.optim.max_batch_size

        # Single model-construction site. To add another model in the future,
        # branch on self.observation here and build it instead of MolDQN.
        # todo: initialize the same weights in target_dqn and dqn?
        # MolDQN didn't do that, I am not sure if I need to add it.
        # --Huanyi
        # GNN mode swaps in the residual prior head: Q starts EXACTLY at the frozen
        # teacher's reward prediction (zero-initialised last layer) instead of at random.
        # See src/models/gnn_teacher.py for the measurements motivating this.
        self.use_gnn = self.observation is ObservationType.GNN
        if self.use_gnn:
            # lazy: keeps torch_geometric off the fingerprint path, which every rank pays
            # for at startup otherwise
            from src.models.gnn_teacher import MolDQNPrior
            _Net = MolDQNPrior
        else:
            _Net = MolDQN
        self.aux_distill = float(getattr(config.train, "aux_distill", 0.0) or 0.0)
        self.dqn = _Net(input_length, 1).to(self.device)
        self.target_dqn = _Net(input_length, 1).to(self.device)
        checkpoint = config.ckpt.checkpoint
        if checkpoint is not None:
            if rank == 0:
                # load pre-trained models from the run folder produced by Recorder:
                # ./Experiments/{checkpoint}/checkpoints/model_{dqn,target_dqn}.pth
                dqn_checkpoint = torch.load(f'./Experiments/{checkpoint}/checkpoints/model_dqn.pth')
                dqn_model_state = dqn_checkpoint['model_state_dict']
                self.dqn.load_state_dict(dqn_model_state)

            target_dqn_checkpoint = torch.load(f'./Experiments/{checkpoint}/checkpoints/model_target_dqn.pth')
            target_dqn_model_state = target_dqn_checkpoint['model_state_dict']
            self.target_dqn.load_state_dict(target_dqn_model_state)
            self.eps_threshold = target_dqn_checkpoint['eps_threshold']

        if self.device.type == 'cuda':
            self.dqn = nn.parallel.DistributedDataParallel(self.dqn,
                device_ids=[self.gpu_index],
                output_device = self.gpu_index)
        else:
            # CPU module: DDP must not receive device_ids/output_device.
            self.dqn = nn.parallel.DistributedDataParallel(self.dqn)

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.use_cxx_incremental_fingerprint = self.observation.cenv_fp_mode

        self.replay_buffer = utils.ReplayBuffer(config.optim.replay_buffer_size)
        # the original replay buffer is confusing and inefficient,
        # use torchrl.data.ReplayBuffer instead
        # self.replay_buffer = torchrl.data.ReplayBuffer(config.optim.replay_buffer_size)
        self.optimizer = getattr(opt, config.optim.optimizer.value)(
            self.dqn.parameters(), lr=config.optim.lr
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
            # Action selection is pure inference — run the UNWRAPPED module under
            # no_grad. Going through the DDP wrapper invokes DDP's forward hooks
            # (a gloo collective) on every call; get_action runs a DATA-DEPENDENT
            # number of times per step (eps-greedy skips it on random actions), so
            # that is wasted sync and a cross-rank desync risk. module() gives
            # identical output (same weights) with no DDP sync.
            with torch.no_grad():
                q_value = self.dqn.module(observations).cpu()
            action = torch.argmax(q_value).numpy()
        return action, isGreedy


    def training_step(self):
        batch_size = min(self.replay_buffer.__len__(), self.max_batch_size)
        states, next_states, rewards, dones = [], [], [], []

        data_batch = self.replay_buffer.sample(batch_size)

        for data in data_batch:
            # data = (reward, float(done), saved_observations)
            reward, done, saved_observations = data

            if self.observation is ObservationType.RDKIT:
                observations = torch.tensor(saved_observations, device = self.device).float()
            elif self.observation is ObservationType.LIST:
                st, fingerprints = saved_observations
                observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                observations = torch.tensor(observations, device = self.device).float()
            elif self.observation in (ObservationType.NUMPY, ObservationType.GNN):
                observations = torch.tensor(saved_observations, device = self.device).float()
            states.append(observations[-1])
            next_states.append(observations)

            rewards.append(reward)
            dones.append(done)

        # q = torch.zeros(batch_size, 1, requires_grad=False)
        # state = torch.FloatTensor(states).reshape(batch_size, fingerprint_length + 1).to(self.device)
        states = torch.stack(states, dim = 0)
        q = self.dqn(states) #.to(self.device)

        # v_tp1 -> max_q #_in_next_states
        # Vectorized target-Q: each next_states[i] is (n_i, feat) with ragged n_i.
        # Concatenate into one (sum_i n_i, feat) tensor, run a SINGLE batched
        # forward, then segment-max back to per-sample maxima. This replaces a
        # Python loop of batch_size tiny sequential GPU forwards.
        counts = [ns.shape[0] for ns in next_states]
        big = torch.cat(next_states, dim=0)
        q_all = self.target_dqn(big).reshape(-1)  # (sum_i n_i,)
        # segment ids: sample index for each candidate row
        seg_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=q_all.device),
            torch.tensor(counts, device=q_all.device),
        )
        seg_max = torch.full((batch_size,), float('-inf'),
                             dtype=q_all.dtype, device=q_all.device)
        seg_max = seg_max.scatter_reduce(0, seg_ids, q_all, reduce='amax')
        max_q = seg_max.reshape(batch_size, 1)

        max_q = max_q.to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32, device = self.device).reshape(q.shape)
        dones = torch.tensor(dones, dtype = torch.float32, device = self.device).reshape(q.shape)

        mask = (1 - dones) * max_q
        target = rewards + self.gamma * mask
        td_error = q - target
        loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )

        loss = loss.mean()

        # --- candidate-set distillation (no extra oracle calls) ---
        # The Bellman gradient only ever reaches states[i] = one molecule per transition,
        # while the argmax has to rank the WHOLE candidate set. `big` already holds every
        # candidate, and its last column is the frozen teacher's reward prediction, so this
        # term supplies a free label for all of them and is what repairs the coverage hole.
        if self.use_gnn and self.aux_distill > 0.0:
            aux = torch.nn.functional.smooth_l1_loss(
                self.dqn(big).reshape(-1), big[:, -1].detach())
            loss = loss + self.aux_distill * aux

        self.optimizer.zero_grad()
        loss.backward()
        torch.distributed.barrier()
        self.optimizer.step()
        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return loss.item()


# Re-export so existing call sites (e.g. src/trainer.py) keep working unchanged:
#   from src.agent import DistributedAgent, MultiMolecules
from src.env.multi_molecules import MultiMolecules  # noqa: E402,F401
