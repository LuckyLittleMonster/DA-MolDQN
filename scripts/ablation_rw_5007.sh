#!/bin/bash
#SBATCH --job-name=rw_5007
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/rw_5007_%j.out
#SBATCH --error=Experiments/logs/rw_5007_%j.err

###############################################################################
# Trial 5007: fix ReplayBuffer setseed (random.seed(0) on first sample)
#   Key change from 5005: utils.py ReplayBuffer now resets random.seed(0)
#   on first sample() call, matching May 2023 original behavior.
#   This was the sole cause of training divergence (verified deterministic).
###############################################################################

set -euo pipefail

echo "============================================"
echo "Trial 5007: ablation_rw, 64 rank, setseed fix"
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
    trial=5007 \
    starter=slurm \
    backend=gloo
