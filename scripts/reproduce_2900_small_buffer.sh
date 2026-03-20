#!/bin/bash
#SBATCH --job-name=repr_2900b
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/repr_2900b_%j.out
#SBATCH --error=log/repr_2900b_%j.err

###############################################################################
# Reproduce trial_2900: 64-rank DDP with SMALL REPLAY BUFFER
#
# HYPOTHESIS:
#   Original trial_29000 used replay_buffer_size=4000 (fixed, per rank).
#   New code multiplies by n_mols: 4000 * 4 = 16,000 per rank.
#   With 36 data points per episode, 4000 fills at ep111.
#   16,000 NEVER fills (only 9000 total data in 250 episodes).
#
#   Small buffer = recent data only → on-policy-ish → faster learning
#   Large buffer = keeps all stale exploration data → diluted signal
#
# CHANGE FROM reproduce_2900.sh:
#   dqn.replay_buffer_size: 4000 → 1000 (code multiplies by 4 → effective 4000)
#   ETKDG: turbo (same as reproduce_2900.sh, faster)
#
# EXPECTED:
#   If buffer size matters: mean reward closer to trial_2900 (~1.65)
#   If not: similar to previous runs (~1.29)
#
###############################################################################

set -euo pipefail

echo "============================================"
echo "Reproduce trial_2900: 64-rank DDP + small buffer"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

srun --kill-on-bad-exit=1 ${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=2902 \
    iteration=2500 \
    num_init_mol=256 \
    max_batch_size=512 \
    starter=slurm \
    backend=gloo \
    dqn.replay_buffer_size=1000 \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
