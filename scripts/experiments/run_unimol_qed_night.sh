#!/bin/bash
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=uqed_n3
#SBATCH --output=Experiments/logs/unimol_qed_%j.out
#SBATCH --error=Experiments/logs/unimol_qed_%j.err

source ~/.bashrc_maple 2>/dev/null
conda activate rl4
cd /shared/data1/Users/l1062811/git/SynDQN

export LD_LIBRARY_PATH=/home/l1062811/data/envs/rl4/lib/gcc/aarch64-conda-linux-gnu/14.3.0:$LD_LIBRARY_PATH

echo "=== UniMolDQN seed=123 (night QOS) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

PYTHONUNBUFFERED=1 python run_unimol.py \
    --model unimol --device cuda \
    --iterations 100000 --max_steps 40 \
    --seed 123 --output_dir Experiments/unimol_dqn

echo ""
echo "=== Done ==="
echo "Date: $(date)"
