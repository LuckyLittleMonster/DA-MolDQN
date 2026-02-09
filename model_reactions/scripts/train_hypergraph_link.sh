#!/bin/bash
# Train Hypergraph Link Predictor on GH200
# Usage: bash model_reactions/scripts/train_hypergraph_link.sh

set -e

echo "============================================================"
echo "Training Hypergraph Link Predictor (Model 1a)"
echo "============================================================"

# Full training on all data
python -m model_reactions.link_prediction.hypergraph_link_predictor \
    --train \
    --data_dir Data/uspto \
    --n_epochs 30 \
    --batch_size 128 \
    --lr 3e-4 \
    --model_size medium \
    --device auto \
    --num_workers 0 \
    --save_dir model_reactions/checkpoints

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
