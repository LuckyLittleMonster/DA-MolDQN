#!/bin/bash
#SBATCH --job-name=reasyn_dqn
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=Experiments/logs/reasyn_dqn_%j.out
#SBATCH --error=Experiments/logs/reasyn_dqn_%j.err

# ReaSyn DQN: 64 mols × 20 episodes, 16 workers
# Expected: ~7 min (20 ep × 20s/ep)

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn

PYTHONUNBUFFERED=1 conda run -n reasyn --live-stream \
    python ../../scripts/train_reasyn_dqn.py \
    --num_mols 64 \
    --episodes 20 \
    --num_workers 16 \
    --mol_json Experiments/aizynth_zinc128_results.json \
    --experiment reasyn_qed \
    --trial 1 \
    --max_steps 5 \
    --top_k 20 \
    --batch_size 64 \
    --grad_steps 10 \
    --lr 1e-4 \
    --gamma 0.9 \
    --eps_start 1.0 \
    --eps_decay 0.85 \
    --eps_min 0.05 \
    --polyak 0.995 \
    --save_freq 5 \
    --seed 42
