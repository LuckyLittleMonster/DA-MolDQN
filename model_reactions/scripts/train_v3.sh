#!/bin/bash
# Train Hypergraph Link Predictor v3 (Phase 1: rich features + reaction center + RCNS + multi-task)

set -e

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Hypergraph Link Predictor v3 Training"
echo "=========================================="

python -m model_reactions.link_prediction.hypergraph_link_predictor_v3 \
    --train \
    --data_dir Data/uspto \
    --n_epochs 40 \
    --batch_size 128 \
    --lr 3e-4 \
    --model_size medium \
    --device auto \
    --save_dir model_reactions/checkpoints

echo ""
echo "Training complete."
