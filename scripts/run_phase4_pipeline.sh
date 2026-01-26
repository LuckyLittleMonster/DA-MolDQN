#!/bin/bash
set -e # Exit on error

# Ensure directories exist
mkdir -p Data/synnet/preprocessed
mkdir -p synnet/checkpoints

echo "=== Step 2: Compute Embeddings ==="
python -m synnet.data_generation.scripts.02_compute_embeddings \
    --building-blocks-file Data/synnet/preprocessed/zinc_filtered.csv.gz \
    --output-file Data/synnet/preprocessed/building_blocks_emb.npy \
    --featurization-fct fp_256 \
    --ncpu 8 --verbose

echo "=== Step 3: Generate Synthetic Trees (1000 samples) ==="
python -m synnet.data_generation.scripts.03_generate_syntrees \
    --building-blocks-file Data/synnet/preprocessed/zinc_filtered.csv.gz \
    --rxn-templates-file Data/synnet/assets/reaction-templates/hb.txt \
    --output-file Data/synnet/preprocessed/synthetic-trees.json.gz \
    --number-syntrees 1000 \
    --ncpu 8 --verbose

echo "=== Step 4: Filter Syntrees ==="
python -m synnet.data_generation.scripts.04_filter_syntrees \
    --input-file Data/synnet/preprocessed/synthetic-trees.json.gz \
    --output-file Data/synnet/preprocessed/synthetic-trees-filtered.json.gz \
    --verbose

echo "=== Step 5: Split Syntrees ==="
python -m synnet.data_generation.scripts.05_split_syntrees \
    --input-file Data/synnet/preprocessed/synthetic-trees-filtered.json.gz \
    --output-dir Data/synnet/preprocessed \
    --verbose

echo "=== Step 6: Featurize Syntrees ==="
python -m synnet.data_generation.scripts.06_featurize_syntrees \
    --input-dir Data/synnet/preprocessed \
    --output-dir Data/synnet/preprocessed/featurized \
    --ncpu 8 --verbose

echo "=== Step 7: Split Data into Xy ==="
python -m synnet.data_generation.scripts.07_split_data_for_networks \
    --input-dir Data/synnet/preprocessed/featurized

# Correct data directory based on 07 output
DATA_DIR="Data/synnet/preprocessed/featurized/Xy"

echo "=== Step 8: Train Models (Dry Run 10 epochs) ==="
# Train all 4 networks
python synnet/train_models.py --data-dir $DATA_DIR --step all --epochs 10 --save-dir synnet/checkpoints

echo "=== Phase 4 Pipeline Complete! ==="
