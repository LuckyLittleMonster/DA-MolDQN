#!/bin/bash
#SBATCH --job-name=rs_moo_px2
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=24:00:00
#SBATCH --output=Experiments/logs/reasyn_multi_proxy_500ep_v2_%j.out
#SBATCH --error=Experiments/logs/reasyn_multi_proxy_500ep_v2_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "ReaSyn + Multi-objective (proxy, product) v2"
echo "  64 mol, 500 episodes, sEH proxy, 16 workers"
echo "  Reward: dock_norm x QED x SA_norm"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=reasyn \
        reward=multi \
        reward.scoring_method=proxy \
        reward.target=seh \
        exp_name=reasyn_multi_proxy \
        trial=2 \
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
