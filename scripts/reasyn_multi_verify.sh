#!/bin/bash
#SBATCH --job-name=reasyn_mv
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=2:00:00
#SBATCH --output=Experiments/logs/reasyn_multi_verify_%j.out
#SBATCH --error=Experiments/logs/reasyn_multi_verify_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "ReaSyn + Multi verify synthesis_map persistence"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=reasyn \
        reward=multi \
        reward.target=1iep \
        reward.primary=qed \
        reward.primary_weight=0.6 \
        reward.dock_weight=0.4 \
        reward.sa_weight=0.2 \
        reward.sa_threshold=5.0 \
        reward.lipinski=true \
        reward.max_logp=5.0 \
        exp_name=reasyn_multi \
        trial=3 \
        episodes=20 \
        max_steps=5 \
        num_molecules=64 \
        method.num_workers=16 \
        method.use_cache=true \
        method.explore_prob=0.05 \
        batch_size=64 \
        grad_steps=4 \
        eps_start=0.22 \
        eps_decay=0.971 \
        eps_min=0.05 \
        log_freq=1 \
        save_freq=5 \
        load_checkpoint=Experiments/models/reasyn_multi_1_checkpoint.pth

echo ""
echo "Done at $(date)"
