#!/bin/bash
#SBATCH --job-name=qed_hyper
#SBATCH --output=Experiments/logs/qed_hyper_%j.out
#SBATCH --error=Experiments/logs/qed_hyper_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

echo "=========================================="
echo "QED + Hypergraph Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Setup environment
source ~/.bashrc_maple
conda activate rl4

cd /home/l1062811/data/git/DA-MolDQN

# Create directories
mkdir -p Experiments/logs
mkdir -p Experiments/models

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Training parameters
EXPERIMENT="qed_hypergraph"
TRIAL=$SLURM_JOB_ID
ITERATION=50000
MAX_STEPS=20
BATCH_SIZE=128

# Initial molecules - drug-like starting points
INIT_MOLS="CCO c1ccccc1O CC(=O)O c1ccc(N)cc1 c1ccc(O)cc1O CC(C)O CCN CCCO"

echo "Running training..."
echo "  Experiment: $EXPERIMENT"
echo "  Trial: $TRIAL"
echo "  Iterations: $ITERATION"
echo "  Max steps/episode: $MAX_STEPS"
echo ""

python main_sync.py \
    --experiment $EXPERIMENT \
    --trial $TRIAL \
    --iteration $ITERATION \
    --max_steps_per_episode $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --qed_weight 0.8 \
    --sa_weight 0.2 \
    --eps_start 1.0 \
    --eps_decay 0.998 \
    --eps_min 0.01 \
    --lr 1e-4 \
    --use_hypergraph \
    --hypergraph_top_k 5 \
    --init_mol $INIT_MOLS \
    --save_freq 500 \
    --log_freq 50 \
    --gpu 0

echo ""
echo "End time: $(date)"
echo "=========================================="
