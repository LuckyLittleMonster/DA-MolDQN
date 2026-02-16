#!/bin/bash
#SBATCH --job-name=v3_full_A
#SBATCH --output=Experiments/logs/v3_full_A_%j.out
#SBATCH --error=Experiments/logs/v3_full_A_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Job A: lr=1e-4, warmup=10%, cosine decay (default 5% → we use longer epochs to compensate)
# Effective LR: 1e-4 (no bs scaling since train_v3 doesn't do auto-scaling)

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "V3 FULL LR Tuning - Job A"
echo "  lr=1e-4, warmup~10%, cosine decay"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

source ~/.bashrc_maple
conda activate rl4

cd /shared/data1/Users/l1062811/git/DA-MolDQN

mkdir -p Experiments/logs
mkdir -p model_reactions/checkpoints/full_A

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

python -m model_reactions.link_prediction.hypergraph_link_predictor_v3 --train \
    --data_dir Data/uspto_full \
    --save_dir model_reactions/checkpoints/full_A \
    --model_size medium \
    --batch_size 512 \
    --lr 1e-4 \
    --n_epochs 60 \
    --early_stop_patience 10 \
    --num_workers 8 \
    --use_cf_ncn \
    --cf_ncn_k 200 \
    --knn_cache Data/precomputed/knn_full_k200.pkl \
    --use_ncn \
    --pretrained_encoder model_reactions/checkpoints/pretrained_encoder.pt

echo ""
echo "End time: $(date)"
echo "=========================================="
