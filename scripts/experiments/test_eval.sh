#!/bin/bash
# Phase 2: Test evaluation (100 episodes, 64 test mols, load checkpoint)
# Usage: sbatch test_eval.sh <METHOD> <TARGET> <REWARD> <CHECKPOINT> [TRIAL]
# Examples:
#   sbatch test_eval.sh route seh dock Experiments/models/route_seh_dock_train_1_checkpoint.pth

#SBATCH --job-name=test_eval
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

METHOD=${1:?Usage: sbatch test_eval.sh <METHOD> <TARGET> <REWARD> <CHECKPOINT> [TRIAL]}
TARGET=${2:?}
REWARD=${3:?}
CHECKPOINT=${4:?}
TRIAL=${5:-1}

EXP_NAME="${METHOD}_${TARGET}_${REWARD}_test"

# Method-specific settings
if [ "$METHOD" = "route" ]; then
    MAX_STEPS=3
    METHOD_EXTRA="method.decompose_method=paroutes method.cascade_workers=8"
else
    MAX_STEPS=5
    METHOD_EXTRA="method.num_workers=16 method.use_cache=true method.explore_prob=0.05"
fi

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "Test Evaluation (Phase 2)"
echo "  method=${METHOD}, target=${TARGET}, reward=${REWARD}, trial=${TRIAL}"
echo "  64 test mols (offset=64), 100 episodes"
echo "  eps: 0.8 -> decay 0.955 (no floor)"
echo "  checkpoint: ${CHECKPOINT}"
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
        init_mol_offset=64 \
        mol_json=Data/paroutes/n1_targets_128.json \
        batch_size=64 \
        grad_steps=4 \
        eps_start=0.8 \
        eps_decay=0.955 \
        log_freq=1 \
        save_freq=999 \
        load_checkpoint=${CHECKPOINT} \
        ${METHOD_EXTRA}

echo ""
echo "Done at $(date)"
