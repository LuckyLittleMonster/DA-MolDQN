#!/bin/bash
#SBATCH --job-name=reasyn_dock
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/reasyn_dock_500ep_%j.out
#SBATCH --error=Experiments/logs/reasyn_dock_500ep_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "ReaSyn + Docking (1iep) - 500 episodes"
echo "Parallel ReaSyn (16 workers) + sync batch dock"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=reasyn \
        reward=dock \
        reward.target=1iep \
        exp_name=reasyn_dock \
        trial=4 \
        episodes=500 \
        max_steps=5 \
        num_molecules=64 \
        method.num_workers=16 \
        method.use_cache=true \
        method.explore_prob=0.05 \
        batch_size=64 \
        grad_steps=4 \
        eps_start=1.0 \
        eps_decay=0.997 \
        eps_min=0.05 \
        log_freq=1 \
        save_freq=50

echo ""
echo "Done at $(date)"
