#!/bin/bash
#SBATCH --job-name=test_alfabet
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=Experiments/logs/test_alfabet_%j.out
#SBATCH --error=Experiments/logs/test_alfabet_%j.err

set -euo pipefail

echo "============================================"
echo "Test: PyTorch ALFABET, 64 rank, 256 mols"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

# Create rendezvous directory
RDZV_DIR="/shared/data1/Users/l1062811/git/DA-MolDQN/Experiments/rdzv"
mkdir -p ${RDZV_DIR}
# Clean stale rendezvous file
rm -f ${RDZV_DIR}/sharedfile_test_alfabet_6001

srun --kill-on-bad-exit=1 ${PYTHON} -u main_hpc.py \
    --trial 6001 \
    --experiment test_alfabet \
    --iteration 2500 \
    --max_steps_per_episode 10 \
    --init_mol_path ./Data/anti_400.txt \
    --num_init_mol 256 \
    --discount_factor 1.0 \
    --reward BDE_IP \
    --reward_weight 0.8 0.2 0.5 \
    --eps_decay 0.968 \
    --max_batch_size 512 \
    --cache bde \
    --backend gloo \
    --starter slurm \
    --maintain_OH exist \
    --observation_type list \
    --save_reward_freq 100 \
    --save_model_freq 1000 \
    --save_path_freq 1000 \
    --init_method "file://${RDZV_DIR}/sharedfile"
