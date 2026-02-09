#!/bin/bash
#SBATCH --job-name=aio_full
#SBATCH --output=Experiments/logs/aio_full_%j.out
#SBATCH --error=Experiments/logs/aio_full_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# AIO Directed Hypergraph Model - FULL dataset training (655K mols, 1.04M reactions)
# Uses shared graph cache to avoid OOM
# RCNS hard negatives + in-batch negatives

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "AIO Directed Hypergraph - USPTO-FULL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

source ~/.bashrc_maple
conda activate rl4

cd /shared/data1/Users/l1062811/git/DA-MolDQN

mkdir -p Experiments/logs
mkdir -p hypergraph/checkpoints

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Train with bs=128, lr=1e-4 (auto-scaled to 2e-4 by base_bs=64 rule)
# hard_neg OFF for FULL dataset (CPU kNN build for 655K mols is infeasible)
# In-batch negatives still provide B-1 free negatives per sample
python hypergraph/train_neighbor_predictor.py \
    --directed \
    --data-dir Data/uspto_full \
    --batch-size 128 \
    --lr 1e-4 \
    --epochs 60 \
    --num-workers 8 \
    --patience 15 \
    --set-aggr attention \
    --no-hard-neg

echo ""
echo "End time: $(date)"
echo "=========================================="
