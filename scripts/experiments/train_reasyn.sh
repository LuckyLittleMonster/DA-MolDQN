#!/bin/bash
# Phase 1: ReaSyn-DQN training (500 episodes, 64 train mols)
# Usage: sbatch train_reasyn.sh <TARGET> <REWARD> [TRIAL]
# Examples:
#   sbatch train_reasyn.sh seh dock_rxnflow 1  # T7: RxnFlow-matching (0.5*QED+0.5*Vina)
#   sbatch train_reasyn.sh seh multi 1          # T8: MOO product
#   sbatch train_reasyn.sh drd2 dock 1          # T9: RGFN-matching (pure dock)
#   sbatch train_reasyn.sh drd2 multi 1         # T10: MOO product
#   sbatch train_reasyn.sh gsk3b dock 1         # T11: HN-GFN-matching (pure dock)
#   sbatch train_reasyn.sh gsk3b multi 1        # T12: MOO product

#SBATCH --job-name=rs_train
#SBATCH --partition=maple_night
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=24:00:00
#SBATCH --output=Experiments/logs/%x_%j.out
#SBATCH --error=Experiments/logs/%x_%j.err

set -euo pipefail

TARGET=${1:?Usage: sbatch train_reasyn.sh <TARGET> <REWARD> [TRIAL]}
REWARD=${2:?Usage: sbatch train_reasyn.sh <TARGET> <REWARD> [TRIAL]}
TRIAL=${3:-1}

EXP_NAME="reasyn_${TARGET}_${REWARD}_train"

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "ReaSyn-DQN Training (Phase 1)"
echo "  target=${TARGET}, reward=${REWARD}, trial=${TRIAL}"
echo "  64 train mols (offset=0), 500 episodes, max_steps=5"
echo "  eps: 1.0 -> decay 0.98 (no floor)"
echo "  workers=16, cache=true, explore_prob=0.05"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=reasyn \
        reward=${REWARD} \
        reward.scoring_method=proxy \
        reward.target=${TARGET} \
        exp_name=${EXP_NAME} \
        trial=${TRIAL} \
        episodes=500 \
        max_steps=5 \
        num_molecules=64 \
        init_mol_offset=0 \
        mol_json=Data/paroutes/n1_targets_128.json \
        batch_size=64 \
        grad_steps=4 \
        eps_start=1.0 \
        eps_decay=0.98 \
        log_freq=1 \
        save_freq=50 \
        method.num_workers=16 \
        method.use_cache=true \
        method.explore_prob=0.05

echo ""
echo "Done at $(date)"
