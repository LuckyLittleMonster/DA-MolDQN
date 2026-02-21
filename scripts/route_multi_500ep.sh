#!/bin/bash
#SBATCH --job-name=route_multi
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=dpp
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/route_multi_500ep_%j.out
#SBATCH --error=Experiments/logs/route_multi_500ep_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "Route-DQN + Multi-objective (1iep) - 500 episodes"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=route \
        reward=multi \
        reward.target=1iep \
        exp_name=route_multi \
        trial=1 \
        episodes=500 \
        max_steps=5 \
        num_molecules=64 \
        batch_size=64 \
        grad_steps=4 \
        eps_start=1.0 \
        eps_decay=0.997 \
        eps_min=0.05 \
        log_freq=5 \
        save_freq=50

echo ""
echo "Done at $(date)"
