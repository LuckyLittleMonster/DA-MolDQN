#!/bin/bash
# Phase 3: Reaction-Aware Contrastive Pre-training
# Then fine-tune v3 with pretrained encoder

set -e

cd "$(dirname "$0")/../.."

SAVE_DIR="model_reactions/checkpoints"

echo "=========================================="
echo "Phase 3a: Reaction-Aware Pre-training"
echo "=========================================="

python -m model_reactions.link_prediction.pretrain \
    --data_dir Data/uspto \
    --n_epochs 30 \
    --batch_size 256 \
    --lr 1e-3 \
    --model_size medium \
    --device auto \
    --save_dir $SAVE_DIR

echo ""
echo "=========================================="
echo "Phase 3b: Fine-tune v3 with pretrained encoder"
echo "=========================================="

python -m model_reactions.link_prediction.hypergraph_link_predictor_v3 \
    --train \
    --data_dir Data/uspto \
    --n_epochs 40 \
    --batch_size 128 \
    --lr 3e-4 \
    --model_size medium \
    --device auto \
    --save_dir $SAVE_DIR \
    --pretrained_encoder ${SAVE_DIR}/pretrained_encoder.pt

echo ""
echo "All phases complete."
