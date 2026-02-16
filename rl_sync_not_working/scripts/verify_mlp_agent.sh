#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=32G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -J mlp_verify
#SBATCH -o Experiments/logs/mlp_verify_%j.out
#SBATCH -e Experiments/logs/mlp_verify_%j.err

cd /shared/data1/Users/l1062811/git/DA-MolDQN

eval "$(conda shell.bash hook)"
conda activate rl4

echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# MLP baseline with classic MolDQN actions
python main_gnn_verify.py \
    --experiment mlp_verify \
    --trial 0 \
    --episodes 8000 \
    --max_steps_per_episode 10 \
    --agent_model mlp \
    --eps_start 1.0 \
    --eps_decay 0.999 \
    --eps_min 0.01 \
    --lr 1e-4 \
    --batch_size 128 \
    --replay_buffer_size 500000 \
    --discount_factor 0.9 \
    --gamma 0.95 \
    --update_interval 20 \
    --log_freq 50 \
    --save_freq 500 \
    --gpu 0

echo "End: $(date)"
