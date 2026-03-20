#!/bin/bash
#SBATCH --job-name=bl_v3
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/bl_v3_%j.out
#SBATCH --error=log/bl_v3_%j.err

###############################################################################
# Baseline bde_db2 single-process v3 — with no-mod ordering fix
###############################################################################

set -euo pipefail

echo "============================================"
echo "Baseline bde_db2 single-process v3 (no-mod fix)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

srun --kill-on-bad-exit=1 ${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=2915 \
    iteration=2500 \
    num_init_mol=256 \
    starter=slurm \
    backend=gloo \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
