#!/bin/bash
# CF-NCN V2: pretrained encoder + CF-NCN + NCN + asymmetric + query interaction + count + early stop
set -e
cd /shared/data1/Users/l1062811/git/DA-MolDQN

SAVE_DIR="model_reactions/checkpoints/cf_ncn_v2"
mkdir -p "$SAVE_DIR"

python -m model_reactions.link_prediction.hypergraph_link_predictor_v3 --train \
    --data_dir Data/uspto \
    --pretrained_encoder model_reactions/checkpoints/pretrained_encoder.pt \
    --use_cf_ncn \
    --cf_ncn_k 200 \
    --cf_ncn_update_freq 5 \
    --use_ncn \
    --ncn_update_freq 5 \
    --n_epochs 60 \
    --batch_size 128 \
    --lr 3e-4 \
    --model_size medium \
    --early_stop_patience 10 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/train.log"
