#!/bin/bash
#SBATCH --job-name=v3_full_C
#SBATCH --output=Experiments/logs/v3_full_C_%j.out
#SBATCH --error=Experiments/logs/v3_full_C_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Job C: lr=1e-4, ReduceLROnPlateau (patience=3, factor=0.5)
# Uses wrapper script that patches the scheduler

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "V3 FULL LR Tuning - Job C"
echo "  lr=1e-4, ReduceLROnPlateau (patience=3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

source ~/.bashrc_maple
conda activate rl4

cd /shared/data1/Users/l1062811/git/DA-MolDQN

mkdir -p Experiments/logs
mkdir -p model_reactions/checkpoints/full_C

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

python scripts/train_v3_full_C.py

echo ""
echo "End time: $(date)"
echo "=========================================="
