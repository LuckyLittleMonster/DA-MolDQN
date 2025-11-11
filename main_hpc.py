import argparse
import os
import copy
import heapq
import hyp
import math
import multiprocessing
import numpy as np
import pdb
import pickle
import random
import psutil
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as opt
import utils
import platform
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record
import datetime
from rdkit import Chem
from rdkit.Chem import QED
from queue import PriorityQueue

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default="da", help="experiment name")
parser.add_argument('--note', type=str, default=None, help="experiment notes")
parser.add_argument('--trial', type=int, default=1, help="experiment trials")
parser.add_argument('--iteration', type=int, default=200000, help="number of iterations")
parser.add_argument('--reward', default="BDE_IP", type=str.lower, choices=['bde_ip', 'qed', 'plogp'],
    help="the reward function. Only BDE_IP, QED and plogp rewards are accepted now.")

parser.add_argument('--reward_weight', type=float, default=[0.8, 0.2, 0.5], nargs="+", help="reward weights.\n"
    "For BDE_IP rewards, it is the reward weights of BDE, IP and RRAS. \n"
    "For QED rewards, it is the reward weights of QED and SA_scores")
parser.add_argument('--penalty_weight', type=float, default=0.0, help="reward weight of the penalty function")
parser.add_argument('--eps_threshold', type=float, default=1.0, help="initial epsilson value (epsilon greedy algorithm)")
parser.add_argument('--eps_decay', type=float, default=0.999, help="epsilson decay rate (epsilon greedy algorithm)")
parser.add_argument('--update_episodes', type=int, default=1, help="train the model every n episodes")
parser.add_argument('--optimizer', type=str, default='Adam', help="set up the optimizer")
parser.add_argument('--lr', type=float, default=1e-4, help="set up the learning rate")
parser.add_argument('--max_steps_per_episode', type=int, default=40, help="maximum steps per episode")
parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of the reward")
parser.add_argument('--min_batch_size', type=int, default=128, help="min batch size for training")
parser.add_argument('--max_batch_size', type=int, default=128, help="max batch size for training")
parser.add_argument('--checkpoint', type=str, default=None, help="the checkpoint which stores the dqn, target_dqn, and eps_threshold")
parser.add_argument('--use_checkpoint_eps', action='store_true', help="if true, it will use eps_threshold stored in checkpint.")

parser.add_argument("--gpu_list", nargs="+", default=[0])

parser.add_argument("--init_mol", nargs="+", default=None)
parser.add_argument('--init_mol_path', type=str, default=None, help="path to the initial molecules directary")
parser.add_argument('--num_init_mol', type=int, default = None, help="Number of initial molecules")
parser.add_argument('--init_mol_start', type=int, default = 0, help="Start id of initial molecules")

parser.add_argument('--starter', default=None, type=str.lower, choices=['slurm', 'torchrun', 'fork', 'spawn', 'forksever'],
    help="the start method")
parser.add_argument('--mp_master_port', type=str, default='6500')
parser.add_argument('--mp_world_size', type=int, default=1)

parser.add_argument('--init_method', type=str, default='file:///shared/data1/Users/l1062811/git/RL4-working/tem/sharedfile', help="init_method of torch.dist")
parser.add_argument('--backend', default="gloo", type=str.lower, choices=['gloo', 'nccl'],
    help="backend")

parser.add_argument('--torch_num_threads', type=int, default=1, help="Number of pytorch threads")

parser.add_argument('--seed_offset', type=int, default=-1, help="if seed_offset >= 0, the seeds in ith worker will be (i + seed_offset). If seed_offset < 0, all seeds are random")
parser.add_argument('--use_cxx_multi_threading', type = int, default = 1, help="Use c++ to have multi workers")
parser.add_argument('--observation_type', default="list", type=str.lower, choices=['rdkit', 'list', 'numpy', 'vector'],
    help="rdkit: do not generate inc fp, use rdkit's fingerprints\n"
        "list: Use c++ to generate incremental fingerprints as list. The method saves a lot of memory, but may be slower. (default)\n"
        "numpy: Use c++ to generate incremental fp as numpy\n"
        "vector: Use 27 dimensional vector and adj matrix. The first 26 dims are atom description. (See Molecule Attention Transformer). The last dim is remaining steps.\n")
parser.add_argument('--cache', nargs="*", type=str.lower, default=['bde', 'ip'], choices=['bde', 'ip'], help="if true, it will use cache for bde or ip predictors.")
parser.add_argument('-ec','--etkdg_max_attempts_cache', type=int, default=7, help="Number of ETKDG attempts for cached IP predictors")
parser.add_argument('-eu','--etkdg_max_attempts_uncache', type=int, default=7, help="Number of ETKDG attempts for uncached IP predictors")

parser.add_argument('--record_top_path', type=int, default=10, help="it will record the top n paths of the generated molecules.")
parser.add_argument('--record_last_path', type=int, default=5, help="it will record the last n paths of the generated molecules.")
parser.add_argument('--record_all_path', action='store_true', help="all generated molecules and paths. (for debug).")

parser.add_argument('--maintain_OH', type=str, default=None, help=
    "default: None or 'None': no limitation\n"
    "same: The number of OH bonds are always same to the initial molecules.\n"
    "exist: All molecules must have one or more OH bonds.\n"
    "n: all mols should have the n of OH bonds\n")

parser.add_argument('--test', action='store_true', help="if true, it will only generate the best molecules without training")
parser.add_argument('--debug', action='store_true', help="debug mode")
parser.add_argument('--save_reward_freq', type=int, default=100, help="it saves the rewards and other useful info every 100 episodes by default.\n"
    "If freq <= 0, it will never save during training or testing")
parser.add_argument('--save_model_freq', type=int, default=1000, help="it saves the model every 1000 episodes by default.\n"
    "If freq <= 0, it will never save during training or testing")
parser.add_argument('--save_path_freq', type=int, default=1000, help="it saves the model every 1000 episodes by default.\n"
    "If freq <= 0, it will never save during training or testing")


args = parser.parse_args()
GPU_list = [int(g) for g in args.gpu_list]

def load_init_mols():
    with open(args.init_mol_path, 'rb') as file:
        mols = file.read().splitlines()
        mols = [str(mol, 'utf-8') for mol in mols]
        return mols[args.init_mol_start:]

def should_save(episode, freq):
    if freq <= 0:
        return False
    return episode % freq == freq - 1
 

def train(args, rank, init_mols):

    # TF may use all memory 
    # https://wiki.ncsa.illinois.edu/display/ISL20/Managing+GPU+memory+when+using+Tensorflow+and+Pytorch
    # import tensorflow as tf
    # gpus = tf.config.list_physical_devices('GPU')
    # [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    # from agent import Agent, QEDRewardMolecule, BDEIPRewardMolecule
    from agent import DistributedAgent, MultiMolecules
    from dqn import MolDQN

    # the cenv should be imported at last.
    # to do : fix the dependency issue in cenv
    # import src.cenv as cenv

    gpu_id = rank % len(GPU_list)
    device = torch.device("cuda:{}".format(GPU_list[gpu_id]))
    min_batch_size = args.min_batch_size
    max_batch_size = args.max_batch_size

    # Please note, as DDP broadcasts model states from rank 0 process to 
    # all other processes in the DDP constructor, you do not need to worry 
    # about different DDP processes starting from different initial model 
    # parameter values.

    # So only rank 0 needs to load the previous models. 
    # Link: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    # Huanyi

    agent = DistributedAgent(hyp.fingerprint_length + 1, GPU_list[gpu_id], device,
        args, rank)

    init_eps_threshold = args.eps_threshold
    if args.checkpoint is not None:
        if args.use_checkpoint_eps:
            init_eps_threshold = agent.eps_threshold
        if args.test:    
            init_eps_threshold = 0.0 # eps is zero for test

    environment = MultiMolecules(
        args = args,
        device = device,
        init_mols = [Chem.MolFromSmiles(s) for s in  init_mols]
        )
    # with open('./Experiments/trial_{}_rank_{}_initial_reward.pickle'.format(args.trial, rank), 'wb') as handle:
    #     pickle.dump(environment.init_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_iteration = args.iteration
    max_steps_per_episode = args.max_steps_per_episode
    max_episodes = max_iteration // max_steps_per_episode

    batch_losses = []

    if args.reward.lower() == "BDE_IP".lower():
        reward_list = {'reward': [], 'BDE': [], 'IP': [], 'RRAB': [], 'IP_Probs': []}
    elif args.reward.lower() == "QED".lower():
        reward_list = {'reward': [], 'QED': [], 'SA_score': []}
    elif args.reward.lower() == "plogp".lower():
        reward_list = {'reward': [], 'plogp': [], 'sim': []}
    
    generated_mols = []
    episode_time_list = []
    bde_cache_hit_rate_list = []
    ip_cache_hit_rate_list = []
    memory_list = []

    # debug_path = []
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
        for st in range(max_steps_per_episode):
            steps_left = max_steps_per_episode - st - 1
            done = steps_left == 0
            # vas, fps = environment.calc_valid_actions() 
            valid_actions_batch, fingerprints_batch = environment.calc_valid_actions()
            rewards = environment.find_reward()
            actions = []
            for valid_actions, fingerprints, reward in zip(valid_actions_batch, fingerprints_batch, rewards['reward']):
                # descriptions are for chemical mols, they are stored in replay buffer with minimal computation and storage cost . 
                # observation are for nn
                # saved_observations 

                if args.observation_type == 'rdkit':
                    saved_observations = np.vstack([utils.get_observations(fp, st) for fp in fingerprints])
                    observations = torch.tensor(saved_observations, device = agent.device).float()
                elif args.observation_type == 'list':
                    saved_observations = (st, fingerprints)
                    observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                    observations = torch.tensor(observations, device = agent.device).float()
                elif args.observation_type == 'numpy':
                    saved_observations = np.vstack([np.append(ob, st) for ob in fingerprints])
                    observations = torch.tensor(saved_observations, device = agent.device).float()
                elif args.observation_type == 'vector':
                    saved_observations = [utils.get_atom_vectors(mol, st) for mol in valid_actions]
                    saved_observations = utils.mol_to_observation(saved_observations)
                    observations = [torch.tensor(ob, device = agent.device) for ob in saved_observations]

                # to do: get_action() should works for multi-mols
                aid, is_greedy = agent.get_action(observations, eps_threshold)
                actions.append(valid_actions[aid])
                # action_fingerprints.append(observations[aid])
                """
                    si -> ai -> si+1 -> {ai+1,0}, {ai+1, 1} ... {ai+1, m-1}
                    In MolDQN, "ai" == "si+1" == "ai+1"
                    valid_actions[-1] and observations[-1] are the current state and their fps,
                    because we allowed no modification.
                """
                if st != 0:
                    data = (reward, float(done), saved_observations)
                    agent.replay_buffer.add(data)
            environment.step(actions, rewards)
            it += 1
            if it >= max_iteration:
                break

        # reward_list.append(rewards)
        for k, v in rewards.items():
            reward_list[k].append(v)
        f_rewards = rewards['reward']

        memory_list.append(current_process.memory_info().rss)

        if args.record_top_path or args.record_all_path or (args.record_last_path + episodes >= max_episodes):
            path, rewards = environment.get_path()
            if args.record_top_path:
                try:
                    for i in range(len(init_mols)):
                        if top_path.qsize() < args.record_top_path or rewards['reward'][i][-1] > top_path.queue[0][0][0]:
                            sample = {'path': path[i]}
                            for k, v in rewards.items():
                                sample[k] = v[i]

                            # if mols have the same reward, the old mol is prefered.
                            priority = (rewards['reward'][i][-1], -it, -i)
                            top_path.put((priority, sample))
                            if top_path.qsize() > args.record_top_path:
                                top_path.get()
                except Exception as e:
                    print(top_path.queue)
                    print(e)
                    raise
                else:
                    pass
                finally:
                    pass
                
            if args.record_all_path:
                for mi in path:
                    for mj in mi:
                        all_path.append(Chem.MolToSmiles(mj))
            if args.record_last_path + episodes >= max_episodes:
                last_path.append((path, rewards))

        if args.test or should_save(episodes, args.save_path_freq) or (episodes + 1 >= max_episodes ):
            if args.record_top_path or args.record_last_path:
                with open(f'./Experiments/{args.experiment}_{args.trial}_{rank}_path.pickle', 'wb') as handle:
                    pickle.dump({'top': top_path.queue, 'last': last_path}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if args.record_all_path:
                with open(f'./Experiments/{args.experiment}_{args.trial}_{rank}_all_path.txt', 'w') as f:
                    for s in all_path:
                        f.write(f'{s}\n')
        if args.test:
            return

        if (should_save(episodes, args.save_model_freq) or it >= max_iteration) and rank == 0:
            # I save both dqn and target_dqn, because they may be used to 
            # recover the trainning once the train script crashed.
            torch.save({
            'episode': episodes,
            'eps_threshold': eps_threshold,
            'model_state_dict': agent.dqn.module.state_dict()
            }, f'./Experiments/models/{args.experiment}_{args.trial}_best_model_dqn.pth')
            torch.save({
            'episode': episodes,
            'eps_threshold': eps_threshold,
            'model_state_dict': agent.target_dqn.state_dict()
            }, f'./Experiments/models/{args.experiment}_{args.trial}_best_model_target_dqn.pth') 
            best_reward = rewards

        if (episodes % args.update_episodes == 0) and (agent.replay_buffer.__len__() >= min_batch_size) and ( not args.test):
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

        bde_cache_hit_rate_list.append(environment.bde_cache.hit_rate(episode = True))
        ip_cache_hit_rate_list.append(environment.ip_cache.hit_rate(episode = True))

        if should_save(episodes, args.save_reward_freq) or (it >= max_iteration):

            saved_data = {
            'batch_losses': batch_losses,
            'episode_time': episode_time_list,
            'rewards': reward_list,
            'memory': memory_list,
            'bde_cache_hit_rate': bde_cache_hit_rate_list,
            'ip_cache_hit_rate': ip_cache_hit_rate_list,
            'total_bde_cache_hit_rate': environment.bde_cache.hit_rate(),
            'total_ip_cache_hit_rate': environment.ip_cache.hit_rate()}

            with open(f'./Experiments/{args.experiment}_{args.trial}_{rank}_episodes.pickle', 'wb') as handle:
                pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        eps_threshold *= args.eps_decay
        episodes += 1


@record
def main(args, rank = None):

    if args.torch_num_threads > 0 :
        torch.set_num_threads(args.torch_num_threads)

    if args.starter == 'torchrun':
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        if rank == 0:
            with open(f'./log/trial_{args.trial}_torch_error_file.txt', 'w') as f:
                f.write(os.environ['TORCHELASTIC_ERROR_FILE'])
    elif args.starter == 'slurm':
        world_size = int(os.environ['SLURM_NPROCS'])
        rank = int(os.environ['SLURM_PROCID'])
    elif args.starter is None:
        world_size = 1
        rank = 0
    else:
        # 'fork', 'spawn', 'forksever'
        world_size = args.mp_world_size
        if rank is None:
            print("The rank of mp start method should be set manually.")
            return;

    print(f"rank {rank}/{world_size} started on {platform.node()}")

    dist.init_process_group(backend=args.backend,
                            init_method=f"{args.init_method}_{args.experiment}_{args.trial}",
                            world_size=world_size,
                            rank=rank,
                            timeout=datetime.timedelta(seconds=600))
    torch.distributed.barrier()
    if rank == 0:
        print("All ranks is initialized", flush=True)

    init_mols = []
    if args.init_mol:
        init_mols = args.init_mol
    elif args.init_mol_path:
        init_mols = load_init_mols()
        if len(init_mols) < world_size:
            if rank == 0:
                print("The number of initial molecules should be greater than the world size.")
            return

    mols_per_rank = args.num_init_mol // world_size
    bid = rank * mols_per_rank
    if bid > len(init_mols):
        bid = len(init_mols)
    eid = (rank + 1) * mols_per_rank
    if eid > len(init_mols):
        eid = len(init_mols)
    init_mols = init_mols[bid:eid]
    print(f"rank: {rank} init_mols: {init_mols}", flush=True)
    # with open('./Experiments/trial_{}_rank_{}_init_mols.pickle'.format(args.trial, rank), 'wb') as handle:
    #     pickle.dump(init_mols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.distributed.barrier()

    if rank == 0:
        with open(f'./Experiments/{args.experiment}_{args.trial}_args.pickle', 'wb') as handle:
            pickle.dump({'args': args}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        start_time = time.time()
    train(args, rank, init_mols)

    if rank == 0:
        comp_time = time.time() - start_time
        with open(f'./Experiments/{args.experiment}_{args.trial}_args.pickle', 'wb') as handle:
            pickle.dump({'args': args, 'computation_time': comp_time}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'The computation time of {args.experiment} {args.trial} is ', comp_time)

class Worker(mp.Process):
    def __init__(self, arg, rank):
        super(Worker, self).__init__()
        self.arg = arg
        self.rank = rank

    def run(self):
        main(args, self.rank)
        

if __name__ == '__main__':
    
    if args.starter in ['fork', 'spawn', 'forksever']:
        mp.set_start_method(args.starter, force=True)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.mp_master_port
        workers = [Worker(args, rank) for rank in range(args.mp_world_size)]
        [w.start() for w in workers]
        [w.join() for w in workers]
    else:
        main(args)
