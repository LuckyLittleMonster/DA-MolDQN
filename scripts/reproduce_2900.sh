#!/bin/bash
#SBATCH --job-name=repr_2900
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/repr_2900_%j.out
#SBATCH --error=log/repr_2900_%j.err

###############################################################################
# Reproduce trial_2900: 64-rank DDP (32 per node × 2 nodes)
#
# PURPOSE:
#   Verify that 64-rank DDP with independent target_dqn per rank
#   reproduces the trial_2900 mean reward (~1.65).
#   This tests the hypothesis that the implicit target_dqn ensemble
#   (from unsynced per-rank initialization) drives the performance gap
#   vs single-process training (mean reward ~1.32).
#
# KEY PARAMETERS (matching trial_2900):
#   RL:
#     iteration        = 2500  (250 episodes × 10 steps)
#     max_steps        = 10
#     eps_threshold    = 1.0   → eps_decay = 0.968 → 0.968^250 ≈ 2e-4
#     discount_factor  = 1.0
#     update_episodes  = 1     (train every episode)
#   DQN:
#     gamma            = 0.95
#     polyak           = 0.995  → target_dqn initial contrib at ep250: 0.995^250 = 28.6%
#     learning_rate    = 1e-4
#     optimizer        = Adam
#     replay_buffer    = 4000 per rank (× 4 mols = 16,000 capacity per rank)
#     max_batch_size   = 512 per rank → 64 ranks × 512 = 32,768 total gradient samples
#   Reward:
#     reward           = bde_ip
#     reward_weight    = [0.8, 0.2, 0.5]  (w_BDE, w_IP, w_RRAB)
#     bde_factor       = 0.9
#     ip_factor        = 0.8
#     BDE scaler       = MinMax [59.80, 96.59]
#     IP  scaler       = MinMax [110.83, 178.16]
#     BDE model        = bde_db2_model3.npz (NOT ALFABET — this differs from original 2900)
#     IP  model        = AIMNet-NSE cv4
#     reward_of_invalid_mol = -1000
#   Molecules:
#     init_mol_path    = anti_400.txt (first 256)
#     num_init_mol     = 256 total → 4 per rank
#   ETKDG:
#     turbo mode       (max_iterations=0, timeout=1, max_attempts=1)
#   Infrastructure:
#     64 ranks         = 32 per node × 2 nodes (each node has 1 GH200 GPU)
#     backend          = gloo  (multi-rank per GPU, DDP gradient averaging)
#     Each rank: independent target_dqn (random init, never broadcast)
#     DDP wraps only dqn → gradients averaged across 64 ranks
#     target_dqn updated via polyak from local dqn copy
#   Cache:
#     BDE cache        = per-rank LRU (128 × 4 = 512 entries)
#     IP cache         = off (not in cache list)
#
# DIFFERENCES FROM ORIGINAL trial_2900:
#   - BDE model: bde_db2_model3 (PyTorch) vs ALFABET (TensorFlow)
#     → On anti_400 init mols, mean diff = 0.53 kcal/mol (negligible)
#   - Code: refactored main_hpc.py (same reward formula, same DQN architecture)
#   - ETKDG: turbo mode vs normal (original used default ETKDG settings)
#   - Container: native conda env vs singularity
#
# EXPECTED RESULT:
#   If ensemble target_dqn is the key factor: mean reward ≈ 1.5-1.7
#   If not: mean reward ≈ 1.3 (similar to single-process ablation_rw_2)
#
###############################################################################

set -euo pipefail

echo "============================================"
echo "Reproduce trial_2900: 64-rank DDP"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

srun --kill-on-bad-exit=1 ${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=2900 \
    iteration=2500 \
    num_init_mol=256 \
    max_batch_size=512 \
    starter=slurm \
    backend=gloo \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
