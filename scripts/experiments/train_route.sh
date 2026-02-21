#!/bin/bash
# Phase 1: Route-DQN training (500 episodes, 64 train mols)
# Usage: sbatch train_route.sh <TARGET> <REWARD> [TRIAL]
# Examples:
#   sbatch train_route.sh seh dock_rxnflow 1  # T1: RxnFlow-matching (0.5*QED+0.5*Vina)
#   sbatch train_route.sh seh multi 1          # T2: MOO product
#   sbatch train_route.sh drd2 dock 1          # T3: RGFN-matching (pure dock)
#   sbatch train_route.sh drd2 multi 1         # T4: MOO product
#   sbatch train_route.sh gsk3b dock 1         # T5: HN-GFN-matching (pure dock)
#   sbatch train_route.sh gsk3b multi 1        # T6: MOO product
#   sbatch train_route.sh - qed 1             # QED only (no target)

#SBATCH --job-name=rt_train
#SBATCH --partition=maple_night
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=dpp
#SBATCH --time=8:00:00
#SBATCH --output=Experiments/logs/%x_%j.out
#SBATCH --error=Experiments/logs/%x_%j.err

set -euo pipefail

TARGET=${1:?Usage: sbatch train_route.sh <TARGET> <REWARD> [TRIAL]}
REWARD=${2:?Usage: sbatch train_route.sh <TARGET> <REWARD> [TRIAL]}
TRIAL=${3:-1}

# Build experiment name and reward args (QED has no target)
if [ "${REWARD}" = "qed" ]; then
    EXP_NAME="route_qed_train"
    REWARD_ARGS="reward=qed"
else
    EXP_NAME="route_${TARGET}_${REWARD}_train"
    REWARD_ARGS="reward=${REWARD} reward.scoring_method=proxy reward.target=${TARGET}"
fi

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "Route-DQN Training (Phase 1)"
echo "  target=${TARGET}, reward=${REWARD}, trial=${TRIAL}"
echo "  64 train mols (offset=0), 500 episodes, max_steps=3"
echo "  eps: 1.0 -> decay 0.98 (no floor)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=route \
        ${REWARD_ARGS} \
        exp_name=${EXP_NAME} \
        trial=${TRIAL} \
        episodes=500 \
        max_steps=3 \
        num_molecules=64 \
        init_mol_offset=0 \
        mol_json=Data/paroutes/n1_targets_128.json \
        batch_size=64 \
        grad_steps=4 \
        eps_start=1.0 \
        eps_decay=0.98 \
        log_freq=1 \
        save_freq=50 \
        method.decompose_method=paroutes \
        method.cascade_workers=8

echo ""
echo "Done at $(date)"
