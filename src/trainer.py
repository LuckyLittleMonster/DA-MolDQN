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
from omegaconf import OmegaConf
from rdkit import Chem

from src import utils
from src import config_defaults as hyp
from src.agent import DistributedAgent, MultiMolecules
from src.persistence import Recorder


def should_save(episode, freq):
    if freq <= 0:
        return False
    return episode % freq == freq - 1


def resolve_device(cfg, rank):
    gpu_list = [int(g) for g in cfg.gpu_list]
    if torch.cuda.is_available() and len(gpu_list) > 0:
        gpu_id = gpu_list[rank % len(gpu_list)]
        return torch.device(f"cuda:{gpu_id}"), gpu_id
    return torch.device("cpu"), 0


class Trainer:
    def __init__(self, cfg, rank: int, world_size: int, init_mols, mode: str = "train"):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.init_mols = init_mols
        self.mode = mode  # train | finetune | test
        self.is_test = mode == "test"
        self.device, self.gpu_index = resolve_device(cfg, rank)
        base = os.path.join(".", "Experiments")
        self.recorder = Recorder(base, cfg.experiment, cfg.trial, rank, world_size)

    def run(self):
        cfg = self.cfg
        rank = self.rank
        if rank == 0:
            self.recorder.save_config(OmegaConf.to_yaml(cfg))

        if cfg.torch_num_threads > 0:
            torch.set_num_threads(cfg.torch_num_threads)

        agent = DistributedAgent(hyp.fingerprint_length + 1, self.gpu_index, self.device, cfg, rank)

        init_eps_threshold = cfg.eps_threshold
        if cfg.checkpoint is not None and cfg.use_checkpoint_eps:
            init_eps_threshold = agent.eps_threshold
        if self.is_test:
            init_eps_threshold = 0.0  # eps is zero for test (greedy), regardless of checkpoint

        environment = MultiMolecules(
            args=cfg, device=self.device,
            init_mols=[Chem.MolFromSmiles(s) for s in self.init_mols],
        )
        max_iteration = cfg.iteration
        max_steps_per_episode = cfg.max_steps_per_episode
        max_episodes = max_iteration // max_steps_per_episode
        min_batch_size = cfg.min_batch_size

        batch_losses = []
        if cfg.reward.lower() == "bde_ip":
            reward_list = {'reward': [], 'BDE': [], 'IP': [], 'RRAB': []}
        elif cfg.reward.lower() == "qed":
            reward_list = {'reward': [], 'QED': [], 'SA_score': []}
        elif cfg.reward.lower() == "plogp":
            reward_list = {'reward': [], 'plogp': [], 'sim': []}
        else:
            raise ValueError(f"Unknown reward: {cfg.reward!r}")

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
            for st in range(max_steps_per_episode):
                steps_left = max_steps_per_episode - st - 1
                done = steps_left == 0
                valid_actions_batch, fingerprints_batch = environment.calc_valid_actions()
                rewards = environment.find_reward()
                actions = []
                for valid_actions, fingerprints, reward in zip(valid_actions_batch, fingerprints_batch, rewards['reward']):
                    if cfg.observation_type == 'rdkit':
                        saved_observations = np.vstack([utils.get_observations(fp, st) for fp in fingerprints])
                        observations = torch.tensor(saved_observations, device=agent.device).float()
                    elif cfg.observation_type == 'list':
                        saved_observations = (st, fingerprints)
                        observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                        observations = torch.tensor(observations, device=agent.device).float()
                    elif cfg.observation_type == 'numpy':
                        saved_observations = np.vstack([np.append(ob, st) for ob in fingerprints])
                        observations = torch.tensor(saved_observations, device=agent.device).float()
                    elif cfg.observation_type == 'vector':
                        saved_observations = [utils.get_atom_vectors(mol, st) for mol in valid_actions]
                        saved_observations = utils.mol_to_observation(saved_observations)
                        observations = [torch.tensor(ob, device=agent.device) for ob in saved_observations]

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

            if cfg.record_top_path or cfg.record_all_path or (cfg.record_last_path + episodes >= max_episodes):
                path, rewards = environment.get_path()
                if cfg.record_top_path:
                    try:
                        for i in range(len(self.init_mols)):
                            if top_path.qsize() < cfg.record_top_path or rewards['reward'][i][-1] > top_path.queue[0][0][0]:
                                sample = {'path': path[i]}
                                for k, v in rewards.items():
                                    sample[k] = v[i]
                                priority = (rewards['reward'][i][-1], -it, -i)
                                top_path.put((priority, sample))
                                if top_path.qsize() > cfg.record_top_path:
                                    top_path.get()
                    except Exception as e:
                        print(top_path.queue)
                        print(e)
                        raise
                if cfg.record_all_path:
                    for mi in path:
                        for mj in mi:
                            all_path.append(Chem.MolToSmiles(mj))
                if cfg.record_last_path + episodes >= max_episodes:
                    last_path.append((path, rewards))

            if self.is_test or should_save(episodes, cfg.save_path_freq) or (episodes + 1 >= max_episodes):
                if cfg.record_top_path or cfg.record_last_path:
                    self.recorder.record_paths(top=list(top_path.queue), last=last_path,
                                               all_smiles=all_path if cfg.record_all_path else None)
                    self.recorder.flush()
            if self.is_test:
                self.recorder.flush()
                self._finalize()
                return

            if (should_save(episodes, cfg.save_model_freq) or it >= max_iteration) and rank == 0:
                self.recorder.save_checkpoint(
                    agent.dqn.module.state_dict(), agent.target_dqn.state_dict(),
                    eps_threshold, episodes)

            # RED LINE: once-per-update_episodes DQN sync — do not move.
            if (episodes % cfg.update_episodes == 0) and (agent.replay_buffer.__len__() >= min_batch_size) and (not self.is_test):
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

            if should_save(episodes, cfg.save_reward_freq) or (it >= max_iteration):
                self.recorder.record_metrics({
                    'batch_losses': batch_losses,
                    'episode_time': episode_time_list,
                    'rewards': reward_list,
                    'memory': memory_list,
                    'bde_cache_hit_rate': bde_cache_hit_rate_list,
                    'total_bde_cache_hit_rate': environment.bde_cache.hit_rate(),
                })
                self.recorder.flush()

            eps_threshold *= cfg.eps_decay
            episodes += 1

        self.recorder.flush()
        self._finalize()

    def _finalize(self):
        """Barrier so every rank has flushed, then rank 0 merges shards -> single .pickle.gz."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            Recorder.merge(os.path.join(".", "Experiments"), self.cfg.experiment, self.cfg.trial, self.world_size)
