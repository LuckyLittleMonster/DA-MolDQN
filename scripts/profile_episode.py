"""Profile per-phase timing of a single 1-rank episode (256 mols, 20 steps)."""
import time, sys, os
os.chdir('/shared/data1/Users/l1062811/git/DA-MolDQN')
sys.path.insert(0, '.')

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import utils
from agent import DistributedAgent
from environment import init_cxx_environment
from omegaconf import OmegaConf

# Minimal config
cfg = OmegaConf.create({
    'mol': {'fingerprint_length': 2048, 'fingerprint_radius': 3,
            'atom_types': ['C', 'O', 'N'], 'allowed_ring_sizes': [3, 5, 6],
            'max_steps_per_episode': 20, 'min_atom_valence': 2,
            'allow_removal': True, 'allow_no_modification': True,
            'allow_bonds_between_rings': False},
    'max_batch_size': 32768, 'min_batch_size': 128,
    'max_steps_per_episode': 20, 'num_init_mol': 256,
    'dqn': {'replay_buffer_size': 256000, 'gamma': 0.95, 'polyak': 0.995,
            'learning_rate': 1e-4, 'optimizer': 'Adam'},
    'checkpoint': None, 'discount_factor': 1.0,
    'reward': {'type': 'qed', 'reward_weight': [1.0]},
    'maintain_OH': None, 'cache': [],
})

# Init
init_cxx_environment(cfg.mol)
utils.init_fingerprint_gen(2048, 3)

with open("Data/zinc_10000.txt") as f:
    smiles = [l.strip() for l in f][:256]

from agent import MultiMolecules
device = torch.device("cuda:0")

# Fake DDP init
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group('gloo', rank=0, world_size=1)

agent = DistributedAgent(2049, 0, device, cfg, 0)

env = MultiMolecules(args=cfg, device=device,
                     init_mols=[Chem.MolFromSmiles(s) for s in smiles])

# Fill replay buffer with some data first (need enough for training)
print("Filling replay buffer with 2 warmup episodes...")
for warmup in range(2):
    env.initialize()
    for st in range(20):
        va, fp = env.calc_valid_actions()
        rewards = env.find_reward()
        actions = []
        for v, f, r in zip(va, fp, rewards['reward']):
            aid = np.random.randint(0, len(v))
            actions.append(v[aid])
            if st != 0:
                if isinstance(f, np.ndarray):
                    n = f.shape[0]
                    obs = np.hstack([f, np.full((n, 1), st, dtype=np.uint8)])
                else:
                    obs = np.vstack([utils.get_observations_from_list(fp_i, st) for fp_i in f])
                agent.replay_buffer.add((r, float(st == 19), obs))
        env.step(actions, rewards)
print(f"Replay buffer size: {len(agent.replay_buffer)}")

# Profile one episode
print("\n=== Profiling 1 episode (256 mols, 20 steps) ===")
env.initialize()

t_cva_total = 0
t_qed_total = 0
t_obs_total = 0
t_dqn_fwd_total = 0
t_action_total = 0
t_step_total = 0

t_ep_start = time.time()

for st in range(20):
    # C++ valid actions
    t0 = time.time()
    va, fp = env.calc_valid_actions()
    t_cva_total += time.time() - t0

    # QED reward
    t0 = time.time()
    rewards = env.find_reward()
    t_qed_total += time.time() - t0

    # Build observations (Phase 1)
    t0 = time.time()
    all_obs_list = []
    splits = []
    saved_obs_list = []
    for valid_actions, fingerprints in zip(va, fp):
        if isinstance(fingerprints, np.ndarray):
            n_act = fingerprints.shape[0]
            step_col = np.full((n_act, 1), st, dtype=np.uint8)
            obs = np.hstack([fingerprints, step_col])
            saved_obs_list.append(obs)
        else:
            saved_obs_list.append((st, fingerprints))
            obs = np.vstack([utils.get_observations_from_list(fp_i, st) for fp_i in fingerprints])
        all_obs_list.append(obs)
        splits.append(obs.shape[0])
    t_obs_total += time.time() - t0

    # DQN forward for action selection (Phase 2)
    t0 = time.time()
    all_obs_np = np.vstack(all_obs_list)
    all_obs_t = torch.tensor(all_obs_np, device=device).float()
    with torch.no_grad():
        all_q = agent.dqn.forward(all_obs_t).cpu()
    q_chunks = torch.split(all_q[:, 0], splits)
    t_dqn_fwd_total += time.time() - t0

    # Action selection + replay buffer (Phase 3)
    t0 = time.time()
    actions = []
    for q_mol, valid_actions, saved_observations, reward in zip(
            q_chunks, va, saved_obs_list, rewards['reward']):
        aid = np.random.randint(0, len(valid_actions))  # all random for profiling
        actions.append(valid_actions[aid])
        if st != 0:
            agent.replay_buffer.add((reward, float(st == 19), saved_observations))
    t_action_total += time.time() - t0

    # Step environment
    t0 = time.time()
    env.step(actions, rewards)
    t_step_total += time.time() - t0

# Training step
t0 = time.time()
loss = agent.training_step()
t_train = time.time() - t0

t_episode = time.time() - t_ep_start

print(f"\nTotal episode time: {t_episode:.3f}s")
print(f"  C++ valid actions:   {t_cva_total:6.3f}s ({t_cva_total/t_episode*100:5.1f}%)")
print(f"  QED/SA reward:       {t_qed_total:6.3f}s ({t_qed_total/t_episode*100:5.1f}%)")
print(f"  Build observations:  {t_obs_total:6.3f}s ({t_obs_total/t_episode*100:5.1f}%)")
print(f"  DQN forward (act):   {t_dqn_fwd_total:6.3f}s ({t_dqn_fwd_total/t_episode*100:5.1f}%)")
print(f"  Action sel + buffer: {t_action_total:6.3f}s ({t_action_total/t_episode*100:5.1f}%)")
print(f"  Env step:            {t_step_total:6.3f}s ({t_step_total/t_episode*100:5.1f}%)")
print(f"  Training (batch={min(len(agent.replay_buffer), 32768)}): {t_train:6.3f}s ({t_train/t_episode*100:5.1f}%)")
env_total = t_cva_total + t_qed_total + t_obs_total + t_dqn_fwd_total + t_action_total + t_step_total
print(f"\nEnv total: {env_total:.3f}s, Training: {t_train:.3f}s")
print(f"Avg obs per mol: {sum(splits)/len(splits):.0f}")

dist.destroy_process_group()
