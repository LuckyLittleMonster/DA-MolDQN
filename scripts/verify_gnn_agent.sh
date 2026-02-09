#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -J gnn_verify
#SBATCH -o Experiments/logs/gnn_verify_%j.out
#SBATCH -e Experiments/logs/gnn_verify_%j.err

cd /shared/data1/Users/l1062811/git/DA-MolDQN

eval "$(conda shell.bash hook)"
conda activate rl4

echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── GNN agent with classic MolDQN actions ──
# 8000 episodes × 10 steps = 80000 iterations
# eps_decay=0.999 → reaches 0.01 at ~ep 4600
python main_gnn_verify.py \
    --experiment gnn_verify \
    --trial 0 \
    --episodes 8000 \
    --max_steps_per_episode 10 \
    --agent_model gnn \
    --eps_start 1.0 \
    --eps_decay 0.999 \
    --eps_min 0.01 \
    --lr 1e-4 \
    --batch_size 128 \
    --replay_buffer_size 500000 \
    --discount_factor 0.9 \
    --gamma 0.95 \
    --update_interval 20 \
    --log_freq 50 \
    --save_freq 500 \
    --gpu 0

echo ""
echo "─────────────────────────────────────"
echo ""

# ── MLP baseline for comparison ──
python main_gnn_verify.py \
    --experiment mlp_verify \
    --trial 0 \
    --episodes 8000 \
    --max_steps_per_episode 10 \
    --agent_model mlp \
    --eps_start 1.0 \
    --eps_decay 0.999 \
    --eps_min 0.01 \
    --lr 1e-4 \
    --batch_size 128 \
    --replay_buffer_size 500000 \
    --discount_factor 0.9 \
    --gamma 0.95 \
    --update_interval 20 \
    --log_freq 50 \
    --save_freq 500 \
    --gpu 0

echo "End: $(date)"
