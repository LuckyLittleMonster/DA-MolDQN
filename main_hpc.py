import os
import numpy as np
import pickle
import psutil
import time
import torch
import torch.distributed as dist
import utils
import platform
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record
import datetime
from rdkit import Chem
from queue import PriorityQueue

import hydra
from omegaconf import DictConfig, OmegaConf


def load_init_mols(cfg):
    with open(cfg.init_mol_path, 'rb') as file:
        mols = file.read().splitlines()
        mols = [str(mol, 'utf-8') for mol in mols]
        return mols[cfg.init_mol_start:]

def should_save(episode, freq):
    if freq <= 0:
        return False
    return episode % freq == freq - 1


def train(cfg, rank, init_mols):

    # Suppress RDKit UFF warnings (S_5+6 etc. during ETKDG) — must be in child process
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    from agent import DistributedAgent, MultiMolecules
    from dqn import MolDQN

    # Init cxx environment and fingerprint gen in worker process
    from environment import init_cxx_environment
    init_cxx_environment(cfg.mol)
    utils.init_fingerprint_gen(cfg.mol.fingerprint_length, cfg.mol.fingerprint_radius)

    GPU_list = [int(g) for g in cfg.gpu_list]
    gpu_id = rank % len(GPU_list)
    device = torch.device("cuda:{}".format(GPU_list[gpu_id]))
    min_batch_size = cfg.min_batch_size
    max_batch_size = cfg.max_batch_size

    agent = DistributedAgent(cfg.mol.fingerprint_length + 1, GPU_list[gpu_id], device,
        cfg, rank)

    init_eps_threshold = cfg.eps_threshold
    if cfg.checkpoint is not None:
        if cfg.use_checkpoint_eps:
            init_eps_threshold = agent.eps_threshold
        if cfg.test:
            init_eps_threshold = 0.0 # eps is zero for test

    environment = MultiMolecules(
        args = cfg,
        device = device,
        init_mols = [Chem.MolFromSmiles(s) for s in  init_mols]
        )

    max_iteration = cfg.iteration
    max_steps_per_episode = cfg.max_steps_per_episode
    max_episodes = max_iteration // max_steps_per_episode

    batch_losses = []

    reward_type = cfg.reward.type.lower()
    if reward_type == "bde_ip":
        reward_list = {'reward': [], 'BDE': [], 'IP': [], 'RRAB': [], 'IP_Probs': []}
    elif reward_type == "qed":
        reward_list = {'reward': [], 'QED': [], 'SA_score': []}
    elif reward_type == "plogp":
        reward_list = {'reward': [], 'plogp': [], 'sim': []}
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    episode_time_list = []
    bde_cache_hit_rate_list = []
    ip_cache_hit_rate_list = []
    memory_list = []

    top_path = PriorityQueue()
    all_path = [] # smiles
    last_path = []


    current_process = psutil.Process(os.getpid())

    best_reward = 0.0
    episodes = 0
    eps_threshold = init_eps_threshold

    it = 0

    while it < max_iteration:
        if rank == 0:
            episode_start_time = time.time()
        environment.initialize()
        prefetched = None
        _trash = []
        for st in range(max_steps_per_episode):
            steps_left = max_steps_per_episode - st - 1
            done = steps_left == 0

            # Use prefetched CVA + rewards, or compute fresh (step 0 / fallback)
            if prefetched is not None:
                _trash = [valid_actions_batch, fingerprints_batch, rewards]
                valid_actions_batch, fingerprints_batch = prefetched['cva']
                rewards = prefetched['rewards']
                prefetched = None
            else:
                _trash = []
                valid_actions_batch, fingerprints_batch = environment.calc_valid_actions()
                rewards = environment.find_reward()

            # Phase 1: Build observations for all molecules
            all_obs_list = []
            splits = []
            saved_obs_list = []
            for valid_actions, fingerprints in zip(valid_actions_batch, fingerprints_batch):
                if isinstance(fingerprints, np.ndarray):
                    n_act = fingerprints.shape[0]
                    step_col = np.full((n_act, 1), st, dtype=np.uint8)
                    obs = np.hstack([fingerprints, step_col])
                    saved_obs_list.append(obs)
                else:
                    saved_obs_list.append((st, fingerprints))
                    obs = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                all_obs_list.append(obs)
                splits.append(obs.shape[0])

            # Phase 2: Single GPU transfer + batch forward
            all_obs_np = np.vstack(all_obs_list)
            all_obs_t = torch.tensor(all_obs_np, device=agent.device).float()
            with torch.no_grad():
                all_q = agent.dqn.forward(all_obs_t).cpu()
            q_chunks = torch.split(all_q[:, 0], splits)

            # Phase 3: Epsilon-greedy action selection + replay buffer
            actions = []
            for q_mol, valid_actions, saved_observations, reward in zip(
                    q_chunks, valid_actions_batch, saved_obs_list, rewards['reward']):
                if np.random.uniform() < eps_threshold:
                    aid = np.random.randint(0, len(valid_actions))
                else:
                    aid = torch.argmax(q_mol).item()
                actions.append(valid_actions[aid])
                if st != 0:
                    data = (reward, float(done), saved_observations)
                    agent.replay_buffer.add(data)
            environment.step(actions, rewards)
            it += 1

            # Prefetch next step: overlap BDE(GPU)||ETKDG(CPU)||cleanup,
            # then AIMNet(GPU)||CVA(CPU). Only if more steps remain.
            if not done and it < max_iteration:
                trash_ref = _trash
                next_rewards, next_cva = environment.find_reward_overlap(
                    prefetch_fn=environment.calc_valid_actions,
                    cleanup_fn=lambda: trash_ref.clear() if trash_ref else None)
                prefetched = {'cva': next_cva, 'rewards': next_rewards}
                _trash = []
            else:
                _trash = []

        if rank == 0:
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
                    for i in range(len(init_mols)):
                        if top_path.qsize() < cfg.record_top_path or rewards['reward'][i][-1] > top_path.queue[0][0][0]:
                            sample = {'path': path[i]}
                            for k, v in rewards.items():
                                sample[k] = v[i]

                            # if mols have the same reward, the old mol is prefered.
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

        if cfg.test or should_save(episodes, cfg.save_path_freq) or (episodes + 1 >= max_episodes ):
            if cfg.record_top_path or cfg.record_last_path:
                with open(f'./Experiments/{cfg.experiment}_{cfg.trial}_{rank}_path.pickle', 'wb') as handle:
                    pickle.dump({'top': top_path.queue, 'last': last_path}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if cfg.record_all_path:
                with open(f'./Experiments/{cfg.experiment}_{cfg.trial}_{rank}_all_path.txt', 'w') as f:
                    for s in all_path:
                        f.write(f'{s}\n')
        if cfg.test:
            return

        if (should_save(episodes, cfg.save_model_freq) or it >= max_iteration) and rank == 0:
            torch.save({
            'episode': episodes,
            'eps_threshold': eps_threshold,
            'model_state_dict': agent.dqn.module.state_dict()
            }, f'./Experiments/models/{cfg.experiment}_{cfg.trial}_best_model_dqn.pth')
            torch.save({
            'episode': episodes,
            'eps_threshold': eps_threshold,
            'model_state_dict': agent.target_dqn.state_dict()
            }, f'./Experiments/models/{cfg.experiment}_{cfg.trial}_best_model_target_dqn.pth')

        if (episodes % cfg.update_episodes == 0) and (agent.replay_buffer.__len__() >= min_batch_size) and ( not cfg.test):
            loss = agent.training_step()
            batch_losses.append(loss)

        if rank == 0:
            episode_time = time.time() - episode_start_time
            remaining_time = episode_time * (max_iteration - it) / max_steps_per_episode
            mean_loss = None
            if len(batch_losses) > 0:
                mean_loss = np.array(batch_losses).mean()
            print(f"episodes: {episodes}, episode time: {float(episode_time):.3f}, "
                f"remaining time: {remaining_time:.3f}, reward: {f_rewards[0]:.3f}, loss: {mean_loss}")
            episode_time_list.append(episode_time)

        if environment.bde_cache is not None:
            bde_cache_hit_rate_list.append(environment.bde_cache.hit_rate(episode = True))
        if environment.ip_cache is not None:
            ip_cache_hit_rate_list.append(environment.ip_cache.hit_rate(episode = True))

        if should_save(episodes, cfg.save_reward_freq) or (it >= max_iteration):

            saved_data = {
            'batch_losses': batch_losses,
            'episode_time': episode_time_list,
            'rewards': reward_list,
            'memory': memory_list,
            'bde_cache_hit_rate': bde_cache_hit_rate_list,
            'ip_cache_hit_rate': ip_cache_hit_rate_list}

            if environment.bde_cache is not None:
                saved_data['total_bde_cache_hit_rate'] = environment.bde_cache.hit_rate()
            if environment.ip_cache is not None:
                saved_data['total_ip_cache_hit_rate'] = environment.ip_cache.hit_rate()

            with open(f'./Experiments/{cfg.experiment}_{cfg.trial}_{rank}_episodes.pickle', 'wb') as handle:
                pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        eps_threshold *= cfg.eps_decay
        episodes += 1


@record
def main(cfg, rank = None):

    if cfg.torch_num_threads > 0 :
        torch.set_num_threads(cfg.torch_num_threads)

    if cfg.starter == 'torchrun':
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        if rank == 0:
            with open(f'./log/trial_{cfg.trial}_torch_error_file.txt', 'w') as f:
                f.write(os.environ['TORCHELASTIC_ERROR_FILE'])
    elif cfg.starter == 'slurm':
        world_size = int(os.environ['SLURM_NPROCS'])
        rank = int(os.environ['SLURM_PROCID'])
    elif cfg.starter is None:
        world_size = 1
        rank = 0
    else:
        # 'fork', 'spawn', 'forkserver'
        world_size = cfg.mp_world_size
        if rank is None:
            print("The rank of mp start method should be set manually.")
            return;

    print(f"rank {rank}/{world_size} started on {platform.node()}")

    # nccl requires env:// init with MASTER_ADDR/PORT; gloo can use file://
    if cfg.backend == 'nccl':
        if 'MASTER_ADDR' not in os.environ:
            if cfg.starter == 'slurm':
                import subprocess
                nodelist = os.environ['SLURM_JOB_NODELIST']
                master = subprocess.check_output(
                    ['scontrol', 'show', 'hostnames', nodelist]
                ).decode().strip().split('\n')[0]
                os.environ['MASTER_ADDR'] = master
            else:
                os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(cfg.mp_master_port)
        init_method = 'env://'
    else:
        init_method = f"{cfg.init_method}_{cfg.experiment}_{cfg.trial}"

    GPU_list = [int(g) for g in cfg.gpu_list]
    gpu_id = rank % len(GPU_list)
    device = torch.device("cuda:{}".format(GPU_list[gpu_id]))

    pg_kwargs = dict(backend=cfg.backend, init_method=init_method,
                     world_size=world_size, rank=rank,
                     timeout=datetime.timedelta(seconds=600))
    if cfg.backend == 'nccl':
        pg_kwargs['device_id'] = device
    dist.init_process_group(**pg_kwargs)
    torch.distributed.barrier()
    if rank == 0:
        print("All ranks is initialized", flush=True)

    init_mols = []
    if cfg.init_mol:
        init_mols = list(cfg.init_mol)
    elif cfg.init_mol_path:
        init_mols = load_init_mols(cfg)
        if len(init_mols) < world_size:
            if rank == 0:
                print("The number of initial molecules should be greater than the world size.")
            return

    mols_per_rank = cfg.num_init_mol // world_size
    bid = rank * mols_per_rank
    if bid > len(init_mols):
        bid = len(init_mols)
    eid = (rank + 1) * mols_per_rank
    if eid > len(init_mols):
        eid = len(init_mols)
    init_mols = init_mols[bid:eid]
    print(f"rank: {rank} init_mols: {init_mols}", flush=True)
    torch.distributed.barrier()

    if rank == 0:
        with open(f'./Experiments/{cfg.experiment}_{cfg.trial}_args.pickle', 'wb') as handle:
            pickle.dump({'args': OmegaConf.to_container(cfg, resolve=True)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        start_time = time.time()
    train(cfg, rank, init_mols)

    if rank == 0:
        comp_time = time.time() - start_time
        with open(f'./Experiments/{cfg.experiment}_{cfg.trial}_args.pickle', 'wb') as handle:
            pickle.dump({'args': OmegaConf.to_container(cfg, resolve=True), 'computation_time': comp_time}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'The computation time of {cfg.experiment} {cfg.trial} is ', comp_time)

    dist.destroy_process_group()

class Worker(mp.Process):
    def __init__(self, cfg, rank):
        super(Worker, self).__init__()
        self.cfg = cfg
        self.rank = rank

    def run(self):
        main(self.cfg, self.rank)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def hydra_main(cfg: DictConfig):
    if cfg.starter in ['fork', 'spawn', 'forkserver']:
        mp.set_start_method(cfg.starter, force=True)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = cfg.mp_master_port
        workers = [Worker(cfg, rank) for rank in range(cfg.mp_world_size)]
        [w.start() for w in workers]
        [w.join() for w in workers]
    else:
        main(cfg)

if __name__ == '__main__':
    hydra_main()
