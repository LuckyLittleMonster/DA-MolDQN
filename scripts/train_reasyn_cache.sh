#!/bin/bash
#SBATCH --job-name=reasyn_cache
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/reasyn_cache_%j.out
#SBATCH --error=Experiments/logs/reasyn_cache_%j.err

# ReaSyn DQN with incremental action cache
# 64 paroutes mols × 500 episodes, 16 workers
# Expected: ep0 ~115s (cold), ep50+ ~20-30s (cached), total ~3-5h

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn

PYTHONUNBUFFERED=1 conda run -n reasyn --live-stream \
    python ../../scripts/train_reasyn_dqn.py \
    --num_mols 64 \
    --episodes 500 \
    --num_workers 16 \
    --mol_json Data/paroutes/n1_targets_128.json \
    --experiment reasyn_paroutes_cache \
    --trial 1 \
    --max_steps 5 \
    --top_k 20 \
    --batch_size 64 \
    --grad_steps 10 \
    --lr 1e-4 \
    --gamma 0.9 \
    --eps_start 1.0 \
    --eps_decay 0.995 \
    --eps_min 0.05 \
    --polyak 0.995 \
    --save_freq 20 \
    --seed 42 \
    --use_cache \
    --explore_prob 0.1 \
    --min_cached_actions 5 \
    --max_neighbors 200 \
    --cache_path Experiments/reasyn_paroutes_cache_1_cache.pickle
