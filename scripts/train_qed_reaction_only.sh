#!/bin/bash
#SBATCH --job-name=qed_reaction
#SBATCH --output=Experiments/logs/qed_reaction_%j.out
#SBATCH --error=Experiments/logs/qed_reaction_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=dpp
#SBATCH --partition=maple

# Activate environment
source ~/.bashrc_maple
conda activate rl4

# Create log directory
mkdir -p Experiments/logs
mkdir -p Experiments/models

# Change to project directory
cd /home/l1062811/data/git/DA-MolDQN

echo "=============================================="
echo "QED Optimization - Pure Reaction Mode"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Training parameters
EXPERIMENT="qed_reaction"
TRIAL=0
ITERATION=50000
MAX_STEPS=10
BATCH_SIZE=128
LR=0.0001
EPS_START=1.0
EPS_END=0.01
EPS_DECAY=0.995

# Run training with reaction_only mode
python main_sync.py \
    --experiment ${EXPERIMENT} \
    --trial ${TRIAL} \
    --iteration ${ITERATION} \
    --max_steps_per_episode ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --eps_start ${EPS_START} \
    --eps_end ${EPS_END} \
    --eps_decay ${EPS_DECAY} \
    --save_freq 100 \
    --log_freq 10 \
    --reaction_only \
    --init_mol "CCO"

echo ""
echo "Training completed at: $(date)"
echo "=============================================="

# Run evaluation after training
echo ""
echo "Running evaluation..."
python main_sync.py \
    --experiment ${EXPERIMENT} \
    --trial ${TRIAL} \
    --eval \
    --reaction_only \
    --init_mol "CCO" \
    --max_steps_per_episode 10

echo ""
echo "Evaluation completed at: $(date)"
echo "=============================================="
