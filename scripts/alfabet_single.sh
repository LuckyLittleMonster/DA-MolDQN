#!/bin/bash
#SBATCH --job-name=alf_1r
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/alf_1r_%j.out
#SBATCH --error=log/alf_1r_%j.err

###############################################################################
# ALFABET single-process — control experiment
#
# Same ALFABET model, but single-process (no DDP).
# If ALFABET is the key factor, this should also reach ~1.65.
# If DDP is also needed, this will stay at ~1.32.
###############################################################################

set -euo pipefail

echo "============================================"
echo "ALFABET single-process"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

srun --kill-on-bad-exit=1 ${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=2911 \
    iteration=2500 \
    num_init_mol=256 \
    starter=slurm \
    backend=gloo \
    reward.bde_weights=bde_predictor/weights/alfabet.npz \
    reward.bde_preprocessor=bde_predictor/weights/alfabet_preprocessor.json \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
