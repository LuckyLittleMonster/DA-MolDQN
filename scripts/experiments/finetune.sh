#!/bin/bash
# Phase 3: Per-molecule fine-tuning (20 episodes per molecule)
# Spawns N concurrent processes, each optimizing one molecule independently.
# Usage: sbatch finetune.sh <METHOD> <TARGET> <REWARD> <CHECKPOINT> <START_IDX> <END_IDX> [TRIAL]
# Examples:
#   # Fine-tune train bottom-25% (indices identified from training results)
#   sbatch finetune.sh route seh dock ckpt.pth 0 63 1
#   # Fine-tune specific test mols
#   sbatch finetune.sh reasyn drd2 dock ckpt.pth 64 127 1

#SBATCH --job-name=finetune
#SBATCH --partition=maple_night
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=dpp
#SBATCH --time=4:00:00
#SBATCH --output=Experiments/logs/%x_%j.out
#SBATCH --error=Experiments/logs/%x_%j.err

set -euo pipefail

METHOD=${1:?Usage: sbatch finetune.sh <METHOD> <TARGET> <REWARD> <CHECKPOINT> <START_IDX> <END_IDX> [TRIAL]}
TARGET=${2:?}
REWARD=${3:?}
CHECKPOINT=${4:?}
START_IDX=${5:?}
END_IDX=${6:?}
TRIAL=${7:-1}

EXP_NAME="${METHOD}_${TARGET}_${REWARD}_finetune"

# Method-specific settings
if [ "$METHOD" = "route" ]; then
    MAX_STEPS=3
    METHOD_EXTRA="method.decompose_method=paroutes method.cascade_workers=8"
    MAX_CONCURRENT=64  # Route-DQN: ~50MB/process
else
    MAX_STEPS=5
    METHOD_EXTRA="method.num_workers=16 method.use_cache=true method.explore_prob=0.05"
    MAX_CONCURRENT=16  # ReaSyn: ~1.2GB/process (fp16)
fi

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "======================================"
echo "Per-Molecule Fine-tuning (Phase 3)"
echo "  method=${METHOD}, target=${TARGET}, reward=${REWARD}, trial=${TRIAL}"
echo "  molecules: offset ${START_IDX} to ${END_IDX}"
echo "  20 episodes/mol, eps: 0.5 -> decay 0.82"
echo "  max_concurrent=${MAX_CONCURRENT}"
echo "  checkpoint: ${CHECKPOINT}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo "======================================"

RUNNING=0
for i in $(seq ${START_IDX} ${END_IDX}); do
    echo "[$(date +%H:%M:%S)] Starting mol ${i}..."
    PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
        python main.py \
            method=${METHOD} \
            reward=${REWARD} \
            reward.scoring_method=proxy \
            reward.target=${TARGET} \
            exp_name=${EXP_NAME}_mol${i} \
            trial=${TRIAL} \
            episodes=20 \
            max_steps=${MAX_STEPS} \
            num_molecules=1 \
            init_mol_offset=${i} \
            mol_json=Data/paroutes/n1_targets_128.json \
            load_checkpoint=${CHECKPOINT} \
            batch_size=64 \
            grad_steps=4 \
            eps_start=0.5 \
            eps_decay=0.82 \
            save_freq=999 \
            log_freq=1 \
            ${METHOD_EXTRA} &

    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= MAX_CONCURRENT )); then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done
wait

echo ""
echo "All per-mol fine-tuning complete at $(date)"
