#!/bin/bash
#SBATCH --job-name=alf_64r
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/alf_64r_%j.out
#SBATCH --error=log/alf_64r_%j.err

###############################################################################
# ALFABET 64-rank DDP — reproduce trial_29000
#
# This is the definitive test: same DDP setup as trial_29000, now with the
# correct ALFABET BDE model (extracted from x86 machine's pooch cache).
#
# EXPECTED: mean reward ~1.65 at ep224 (matching trial_29000)
###############################################################################

set -euo pipefail

echo "============================================"
echo "ALFABET 64-rank DDP"
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
    trial=2910 \
    iteration=2500 \
    num_init_mol=256 \
    max_batch_size=512 \
    starter=slurm \
    backend=gloo \
    reward.bde_weights=bde_predictor/weights/alfabet.npz \
    reward.bde_preprocessor=bde_predictor/weights/alfabet_preprocessor.json \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
