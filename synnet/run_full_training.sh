#!/bin/bash
# Script to launch full MolDQN training with SynNet environment

# Configuration
ITERATIONS=5000
MAX_STEPS=15
INIT_MOL="C"
NUM_INIT_MOL=1
CKPT_DIR="synnet/checkpoints"
RXN_TEMPLATES="Data/synnet/preprocessed/rxn_collection.json.gz"
BUILDING_BLOCKS="Data/synnet/preprocessed/zinc_filtered.csv.gz"
EMBEDDINGS="Data/synnet/preprocessed/building_blocks_emb.npy"

echo "Launching Full MolDQN Training with SynNet..."
echo "Iterations: $ITERATIONS"
echo "Max Steps: $MAX_STEPS"

python main_hpc.py \
    --env_type synnet \
    --max_steps_per_episode $MAX_STEPS \
    --iteration $ITERATIONS \
    --init_mol "$INIT_MOL" \
    --num_init_mol $NUM_INIT_MOL \
    --init_method tcp://127.0.0.1:23456 \
    --synnet_ckpt_dir $CKPT_DIR \
    --synnet_rxn_templates $RXN_TEMPLATES \
    --synnet_building_blocks $BUILDING_BLOCKS \
    --synnet_embeddings $EMBEDDINGS

echo "Training launched."
