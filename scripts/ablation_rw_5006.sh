#!/bin/bash
#SBATCH --job-name=rw_5006
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/rw_5006_%j.out
#SBATCH --error=Experiments/logs/rw_5006_%j.err

###############################################################################
# Trial 5006: replicate Aug 11 broadcasting bug (cross_batch_loss)
#   Key change: dqn.cross_batch_loss=true reproduces the [N,N] outer product
#   in Bellman td_error that Aug 11 had due to shape mismatch.
#   Keeps max_batch_size=512 to match Aug 11 trial_29000.
#   Also keeps manual gradient allreduce from trial 5005.
###############################################################################

set -euo pipefail

echo "============================================"
echo "Trial 5006: ablation_rw, 64 rank, cross_batch_loss=true, bs=512"
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
    trial=5006 \
    starter=slurm \
    backend=gloo \
    dqn.cross_batch_loss=true
