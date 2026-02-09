#!/bin/bash
#SBATCH --job-name=hybrid_t2m
#SBATCH --account=dpp
#SBATCH --partition=maple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=Experiments/logs/hybrid_template_%j.out
#SBATCH --error=Experiments/logs/hybrid_template_%j.err

# Hybrid 2-Model + Template RL training
# 1000 episodes, 64 mols, max_steps=5, top_k=128

source ~/.bashrc_maple 2>/dev/null
cd /shared/data1/Users/l1062811/git/DA-MolDQN

mkdir -p Experiments/logs Experiments/models

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream python -u main_sync.py \
    --experiment hybrid_template \
    --trial 1 \
    --reactant_method template_2model \
    --reaction_only \
    --hybrid_total_top_k 128 \
    --hybrid_model_top_k 20 \
    --max_steps_per_episode 5 \
    --iteration 5000 \
    --num_molecules 64 \
    --init_mol CCO 'c1ccccc1O' 'CC(=O)O' 'c1ccc(N)cc1' \
    --batch_size 64 \
    --replay_buffer_size 50000 \
    --lr 1e-4 \
    --eps_start 1.0 \
    --eps_decay 0.997 \
    --eps_min 0.01 \
    --save_freq 50 \
    --log_freq 10
