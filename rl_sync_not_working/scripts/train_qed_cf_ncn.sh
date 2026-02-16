#!/bin/bash
# Train Hypergraph Link Predictor V3 with CF-NCN (Collaborative Filtering NCN)
# CF-NCN uses Morgan FP kNN to find bridge molecules: similar to A that react with B

set -e

cd /shared/data1/Users/l1062811/git/DA-MolDQN

SAVE_DIR="model_reactions/checkpoints/cf_ncn"
mkdir -p "$SAVE_DIR"

python -m model_reactions.link_prediction.hypergraph_link_predictor_v3 --train \
    --data_dir Data/uspto \
    --use_cf_ncn \
    --cf_ncn_k 200 \
    --cf_ncn_update_freq 5 \
    --n_epochs 40 \
    --batch_size 128 \
    --lr 3e-4 \
    --model_size medium \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/train.log"
