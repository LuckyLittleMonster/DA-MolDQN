#!/bin/bash
#SBATCH --job-name=reasyn_dock
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=Experiments/logs/reasyn_dock_%j.out
#SBATCH --error=Experiments/logs/reasyn_dock_%j.err

# ReaSyn DQN docking: 16 mols × 500 episodes, 4 workers
# eps_decay=0.98 → eps≈0.05 at ep150, exploitation for remaining 350 eps
# Estimated: ~100s/ep × 500 = ~14h

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn

PYTHONUNBUFFERED=1 conda run -n reasyn --live-stream \
    python ../../scripts/train_reasyn_dqn.py \
    --num_mols 16 \
    --episodes 500 \
    --num_workers 4 \
    --reward_mode dock \
    --target 1iep \
    --dock_weight 0.7 \
    --sa_weight 0.3 \
    --experiment reasyn_dock \
    --trial 2 \
    --mol_json Experiments/aizynth_zinc128_results.json \
    --max_steps 5 \
    --top_k 20 \
    --batch_size 64 \
    --grad_steps 20 \
    --lr 1e-4 \
    --gamma 0.9 \
    --eps_start 1.0 \
    --eps_decay 0.98 \
    --eps_min 0.05 \
    --polyak 0.995 \
    --save_freq 50 \
    --seed 42
