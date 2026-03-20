#!/bin/bash
#SBATCH --job-name=qed_64r
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=log/qed_64r_%j.out
#SBATCH --error=log/qed_64r_%j.err

###############################################################################
# QED 64-rank — reproduce ZincQED baseline with no-mod ordering fix
# Original params: iteration=5000, max_steps=20, zinc_10000, rw=[1.0]
###############################################################################

set -euo pipefail

echo "============================================"
echo "QED 64-rank reproduction (no-mod fix)"
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
    reward=qed \
    'reward.reward_weight=[1.0]' \
    trial=3006 \
    iteration=5000 \
    max_steps_per_episode=20 \
    eps_threshold=1.0 \
    eps_decay=0.968 \
    discount_factor=1.0 \
    init_mol_path=Data/zinc_10000.txt \
    num_init_mol=256 \
    starter=slurm \
    backend=gloo \
    max_batch_size=512 \
    min_batch_size=128 \
    maintain_OH=null \
    'cache=[]' \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
