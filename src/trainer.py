"""RL training loop, ported verbatim from main_hpc.train()/main().

Behaviour is preserved exactly; only organisation changes:
  * device is resolved (cuda if available, else cpu) instead of hard-coded cuda;
  * scattered pickle.dump/torch.save are routed through ``Recorder``;
  * mode in {train, finetune, test} replaces the --test / --checkpoint flags.

RED LINE — minimal communication: the DQN is synced once per ``update_episodes``
episodes via ``agent.training_step()`` (default: once per episode). That call site
is copied unchanged; do not move sync anywhere else in the loop.
"""
from __future__ import annotations

import os
import time
from queue import PriorityQueue

import numpy as np
import psutil
import torch
import torch.distributed as dist
from rdkit import Chem

from src import utils
from src.config import ENV_DEFAULTS, ObservationType, RewardType
from src.agent import DistributedAgent, MultiMolecules
from src.persistence import Recorder


def should_save(episode, freq):
    if freq <= 0:
        return False
    return episode % freq == freq - 1


def resolve_device(config, rank):
    gpu_list = [int(g) for g in config.mols.gpu_list]
    if torch.cuda.is_available() and len(gpu_list) > 0:
        gpu_id = gpu_list[rank % len(gpu_list)]
        return torch.device(f"cuda:{gpu_id}"), gpu_id
    return torch.device("cpu"), 0


class Trainer:
    def __init__(self, config, rank: int, world_size: int, init_mols,
                 mode: str = "train", config_yaml: str | None = None):
        self.config = config
        self.config_yaml = config_yaml
        self.rank = rank
        self.world_size = world_size
        self.init_mols = init_mols
        self.mode = mode  # train | finetune | test
        self.is_test = mode == "test"
        self.device, self.gpu_index = resolve_device(config, rank)
        base = os.path.join(".", "Experiments")
        self.recorder = Recorder(base, config.experiment.experiment,
                                 config.experiment.trial, rank, world_size)

    def run(self):
        config = self.config
        rank = self.rank
        if rank == 0 and self.config_yaml is not None:
            self.recorder.save_config(self.config_yaml)

        if config.dist.torch_num_threads > 0:
            torch.set_num_threads(config.dist.torch_num_threads)

        gnn_teacher = None
        input_length = ENV_DEFAULTS.fingerprint_length + 1
        if config.env.observation is ObservationType.GNN:
            from src.models.gnn_teacher import GNNTeacher, make_corrections
            gnn_teacher = GNNTeacher(config.env.gnn_ckpt, self.device,
                                     reward_kind=config.reward.type.value)
            # r_hat MUST reproduce the FULL reward. The teacher was trained on the property
            # only, so any reward term it never saw has to be added here -- otherwise the
            # better-ranking agent optimises the modelled part and drifts on the rest
            # (measured: r_hat vs production QED reward Spearman only +0.363; molecules grew
            # 16.6 -> 19.5 heavy atoms on QED and shrank 13.9 -> 9.9 with n_unique 0.59 ->
            # 0.19 on BDE_IP). See make_corrections for what each reward needs.
            gnn_teacher.scale_fn, gnn_teacher.extra_fn = make_corrections(
                config.reward.type.value, config, gnn_teacher)
            input_length = gnn_teacher.obs_dim
            if rank == 0:
                print(f"[gnn] teacher {config.env.gnn_ckpt} -> obs_dim {input_length} "
                      f"(aux_distill={getattr(config.train, 'aux_distill', 0.0)}, "
                      f"scale_fn={'on' if gnn_teacher.scale_fn else 'none'}, "
                      f"extra_fn={'on' if gnn_teacher.extra_fn else 'none'})", flush=True)

        agent = DistributedAgent(input_length, self.gpu_index,
                                 self.device, config, rank)

        init_eps_threshold = config.train.eps_threshold
        if config.ckpt.checkpoint is not None and config.ckpt.use_checkpoint_eps:
            init_eps_threshold = agent.eps_threshold
        if self.is_test:
            init_eps_threshold = 0.0  # eps is zero for test (greedy), regardless of checkpoint

        environment = MultiMolecules(
            config=config, device=self.device,
            init_mols=[Chem.MolFromSmiles(s) for s in self.init_mols],
        )
        max_iteration = config.train.iteration
        max_steps_per_episode = config.train.max_steps_per_episode
        max_episodes = max_iteration // max_steps_per_episode
        min_batch_size = config.optim.min_batch_size
        observation = config.env.observation

        batch_losses = []
        if config.reward.type is RewardType.BDE_IP:
            reward_list = {'reward': [], 'BDE': [], 'IP': [], 'RRAB': []}
        elif config.reward.type is RewardType.BDE_IP2:
            # SIZE_D replaces RRAB: the multiplicative size desirability actually applied
            reward_list = {'reward': [], 'BDE': [], 'IP': [], 'SIZE_D': []}
        elif config.reward.type is RewardType.QED:
            reward_list = {'reward': [], 'QED': [], 'SA_score': []}
        elif config.reward.type is RewardType.PLOGP:
            reward_list = {'reward': [], 'plogp': [], 'sim': []}
        else:
            raise ValueError(f"Unknown reward: {config.reward.type!r}")

        episode_time_list = []
        bde_cache_hit_rate_list = []
        memory_list = []
        top_path = PriorityQueue()
        all_path = []
        last_path = []
        current_process = psutil.Process(os.getpid())

        episodes = 0
        eps_threshold = init_eps_threshold
        it = 0

        while it < max_iteration:
            if rank == 0:
                episode_start_time = time.time()
            environment.initialize()
            if gnn_teacher is not None and environment.states:
                # bde_ip's rrab is relative to the EPISODE's start molecule; refresh it so
                # r_hat's correction matches what the reward will actually compute. Cheap,
                # and done unconditionally so a future correction that reads init_size
                # cannot silently see a stale value.
                _m0 = environment.states[0]
                gnn_teacher.init_size = _m0.GetNumAtoms() + _m0.GetNumBonds()
            for st in range(max_steps_per_episode):
                steps_left = max_steps_per_episode - st - 1
                done = steps_left == 0
                valid_actions_batch, fingerprints_batch = environment.calc_valid_actions()
                rewards = environment.find_reward()
                actions = []
                for valid_actions, fingerprints, reward in zip(valid_actions_batch, fingerprints_batch, rewards['reward']):
                    if observation is ObservationType.RDKIT:
                        saved_observations = np.vstack([utils.get_observations(fp, st) for fp in fingerprints])
                        observations = torch.tensor(saved_observations, device=agent.device).float()
                    elif observation is ObservationType.LIST:
                        saved_observations = (st, fingerprints)
                        observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                        observations = torch.tensor(observations, device=agent.device).float()
                    elif observation is ObservationType.NUMPY:
                        saved_observations = np.vstack([np.append(ob, st) for ob in fingerprints])
                        observations = torch.tensor(saved_observations, device=agent.device).float()
                    elif config.env.observation is ObservationType.GNN:
                        # valid_actions are RDKit Mols; the frozen teacher turns them into
                        # [embedding, step, r_hat] with r_hat calibrated to the true reward
                        saved_observations = gnn_teacher.observe(valid_actions, st)
                        observations = torch.tensor(saved_observations, device=agent.device).float()

                    aid, is_greedy = agent.get_action(observations, eps_threshold)
                    actions.append(valid_actions[aid])
                    if st != 0:
                        data = (reward, float(done), saved_observations)
                        agent.replay_buffer.add(data)
                environment.step(actions, rewards)
                it += 1
                if it >= max_iteration:
                    break

            for k, v in rewards.items():
                reward_list[k].append(v)
            f_rewards = rewards['reward']
            memory_list.append(current_process.memory_info().rss)

            rec = config.record
            if rec.record_top_path or rec.record_all_path or (rec.record_last_path + episodes >= max_episodes):
                path, rewards = environment.get_path()
                if rec.record_top_path:
                    try:
                        for i in range(len(self.init_mols)):
                            if top_path.qsize() < rec.record_top_path or rewards['reward'][i][-1] > top_path.queue[0][0][0]:
                                sample = {'path': path[i]}
                                for k, v in rewards.items():
                                    sample[k] = v[i]
                                priority = (rewards['reward'][i][-1], -it, -i)
                                top_path.put((priority, sample))
                                if top_path.qsize() > rec.record_top_path:
                                    top_path.get()
                    except Exception as e:
                        print(top_path.queue)
                        print(e)
                        raise
                if rec.record_all_path:
                    for mi in path:
                        for mj in mi:
                            all_path.append(Chem.MolToSmiles(mj))
                if rec.record_last_path + episodes >= max_episodes:
                    last_path.append((path, rewards))

            if self.is_test or should_save(episodes, rec.save_path_freq) or (episodes + 1 >= max_episodes):
                if rec.record_top_path or rec.record_last_path:
                    self.recorder.record_paths(top=list(top_path.queue), last=last_path,
                                               all_smiles=all_path if rec.record_all_path else None)
                    self.recorder.flush()
            if self.is_test:
                self.recorder.flush()
                self._finalize()
                return

            if (should_save(episodes, rec.save_model_freq) or it >= max_iteration) and rank == 0:
                self.recorder.save_checkpoint(
                    agent.dqn.module.state_dict(), agent.target_dqn.state_dict(),
                    eps_threshold, episodes)

            # RED LINE: once-per-update_episodes DQN sync — do not move.
            if (episodes % config.train.update_episodes == 0) and (agent.replay_buffer.__len__() >= min_batch_size) and (not self.is_test):
                loss = agent.training_step()
                batch_losses.append(loss)

            if rank == 0:
                episode_time = time.time() - episode_start_time
                remaining_time = episode_time * (max_iteration - it) / max_steps_per_episode
                mean_loss = np.array(batch_losses).mean() if batch_losses else None
                print(f"episodes: {episodes}, episode time: {float(episode_time):.3f}, "
                      f"remaining time: {remaining_time:.3f}, reward: {f_rewards[0]:.3f}, loss: {mean_loss}", flush=True)
                episode_time_list.append(episode_time)

            bde_cache_hit_rate_list.append(environment.bde_cache.hit_rate(episode=True))

            if should_save(episodes, rec.save_reward_freq) or (it >= max_iteration):
                self.recorder.record_metrics({
                    'batch_losses': batch_losses,
                    'episode_time': episode_time_list,
                    'rewards': reward_list,
                    'memory': memory_list,
                    'bde_cache_hit_rate': bde_cache_hit_rate_list,
                    'total_bde_cache_hit_rate': environment.bde_cache.hit_rate(),
                })
                self.recorder.flush()

            eps_threshold *= config.train.eps_decay
            episodes += 1

        self.recorder.flush()
        self._finalize()

    def _finalize(self):
        """Barrier so every rank has flushed, then rank 0 merges shards -> single .pickle.gz."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            Recorder.merge(os.path.join(".", "Experiments"),
                           self.config.experiment.experiment,
                           self.config.experiment.trial, self.world_size)
