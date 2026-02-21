#!/bin/bash
# Phase 1b: Random baseline (100 episodes, 64 train mols, eps=1.0 always)
# Usage: sbatch random_baseline.sh <METHOD> <TARGET> <REWARD> [TRIAL]
# Examples:
#   sbatch random_baseline.sh route seh dock 1
#   sbatch random_baseline.sh reasyn drd2 dock 1

#SBATCH --job-name=random
#SBATCH --partition=maple_night
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=8:00:00
#SBATCH --output=Experiments/logs/%x_%j.out
#SBATCH --error=Experiments/logs/%x_%j.err

set -euo pipefail

METHOD=${1:?Usage: sbatch random_baseline.sh <METHOD> <TARGET> <REWARD> [TRIAL]}
TARGET=${2:?Usage: sbatch random_baseline.sh <METHOD> <TARGET> <REWARD> [TRIAL]}
REWARD=${3:?Usage: sbatch random_baseline.sh <METHOD> <TARGET> <REWARD> [TRIAL]}
TRIAL=${4:-1}

EXP_NAME="${METHOD}_${TARGET}_${REWARD}_random"

# Method-specific settings
if [ "$METHOD" = "route" ]; then
    MAX_STEPS=3
    METHOD_EXTRA="method.decompose_method=paroutes method.cascade_workers=8"
    MEM="64G"
else
    MAX_STEPS=5
    METHOD_EXTRA="method.num_workers=16 method.use_cache=true method.explore_prob=0.05"
    MEM="128G"
fi

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "Random Baseline (Phase 1b)"
echo "  method=${METHOD}, target=${TARGET}, reward=${REWARD}, trial=${TRIAL}"
echo "  64 train mols, 100 episodes, eps=1.0 always (no learning)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=${METHOD} \
        reward=${REWARD} \
        reward.scoring_method=proxy \
        reward.target=${TARGET} \
        exp_name=${EXP_NAME} \
        trial=${TRIAL} \
        episodes=100 \
        max_steps=${MAX_STEPS} \
        num_molecules=64 \
        init_mol_offset=0 \
        mol_json=Data/paroutes/n1_targets_128.json \
        batch_size=64 \
        grad_steps=4 \
        eps_start=1.0 \
        eps_decay=1.0 \
        log_freq=1 \
        save_freq=999 \
        ${METHOD_EXTRA}

echo ""
echo "Done at $(date)"
