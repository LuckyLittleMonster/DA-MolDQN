#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -J qed_mlp
#SBATCH -o Experiments/logs/qed_mlp_%j.out
#SBATCH -e Experiments/logs/qed_mlp_%j.err

cd /shared/data1/Users/l1062811/git/DA-MolDQN

eval "$(conda shell.bash hook)"
conda activate rl4

# eps_decay = 0.999^8 (slower decay for large N)
EPS_DECAY=$(python3 -c "print(f'{0.999**8:.10f}')")
echo "eps_decay = $EPS_DECAY"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2000 episodes × 5 steps = 10000 iterations
# MLP agent + hypergraph reactions, 64 diverse zinc molecules
python main_sync.py \
    --experiment qed_mlp \
    --trial 0 \
    --iteration 10000 \
    --max_steps_per_episode 5 \
    --init_mol_path ./Data/zinc_10000.txt \
    --num_molecules 64 \
    --reaction_only \
    --reactant_method hypergraph \
    --product_num_beams 1 \
    --hypergraph_top_k 5 \
    --agent_model mlp \
    --eps_start 1.0 \
    --eps_decay "$EPS_DECAY" \
    --eps_min 0.01 \
    --lr 1e-4 \
    --batch_size 64 \
    --replay_buffer_size 50000 \
    --discount_factor 0.9 \
    --qed_weight 0.8 \
    --sa_weight 0.2 \
    --log_freq 10 \
    --save_freq 100 \
    --gpu 0

echo "End: $(date)"
