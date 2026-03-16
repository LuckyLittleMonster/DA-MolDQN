#!/bin/bash
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=uqed
#SBATCH --output=Experiments/logs/unimol_qed_%j.out
#SBATCH --error=Experiments/logs/unimol_qed_%j.err
#SBATCH --array=0-1

# Only run UniMol experiments (MolDQN already completed)
# Array 0: unimol seed=42, 1: unimol seed=123

source ~/.bashrc_maple 2>/dev/null
conda activate rl4
cd /shared/data1/Users/l1062811/git/SynDQN

export LD_LIBRARY_PATH=/home/l1062811/data/envs/rl4/lib/gcc/aarch64-conda-linux-gnu/14.3.0:$LD_LIBRARY_PATH

ITERS=100000
STEPS=40

SEEDS=(42 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== UniMolDQN seed=${SEED} (pre-computed embeddings in replay buffer) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Clear stale results from previous (slow) runs
rm -rf Experiments/unimol_dqn/unimol_qed_s${SEED}

PYTHONUNBUFFERED=1 python run_unimol.py \
    --model unimol --device cuda \
    --iterations $ITERS --max_steps $STEPS \
    --seed $SEED --output_dir Experiments/unimol_dqn

echo ""
echo "=== Done ==="
echo "Date: $(date)"
