#!/bin/bash
#SBATCH --job-name=v3_ddp
#SBATCH --output=Experiments/logs/v3_ddp_%j.out
#SBATCH --error=Experiments/logs/v3_ddp_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Line-buffered output for monitoring
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "V3 Link Predictor DDP Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo ""

# Setup environment
source ~/.bashrc_maple
conda activate rl4

cd /shared/data1/Users/l1062811/git/DA-MolDQN

# Create log directory
mkdir -p Experiments/logs

# DDP environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Training parameters
DATA_DIR="Data/uspto_full"
SAVE_DIR="${1:-model_reactions/checkpoints/ddp_full}"
MODEL_SIZE="${2:-large}"
BATCH_SIZE="${3:-256}"
LR="${4:-3e-4}"
N_EPOCHS="${5:-60}"
EARLY_STOP="${6:-10}"
WORKERS=8

echo "Config:"
echo "  Data: $DATA_DIR"
echo "  Model: $MODEL_SIZE"
echo "  Save: $SAVE_DIR"
echo "  Batch size: ${BATCH_SIZE}/GPU (effective: $((BATCH_SIZE * SLURM_NTASKS)))"
echo "  LR: $LR (no scaling)"
echo "  Epochs: $N_EPOCHS (early stop: $EARLY_STOP)"
echo ""

# Run with srun (each node gets 1 task = 1 GPU)
srun -u python scripts/train_v3_ddp.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --n_epochs $N_EPOCHS \
    --early_stop_patience $EARLY_STOP \
    --num_workers $WORKERS \
    --use_cf_ncn \
    --cf_ncn_k 200 \
    --knn_cache Data/precomputed/knn_full_k200.pkl \
    --use_ncn \
    --pretrained_encoder model_reactions/checkpoints/pretrained_encoder.pt

echo ""
echo "End time: $(date)"
echo "=========================================="
