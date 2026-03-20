#!/bin/bash
#SBATCH --job-name=ablation_rw
#SBATCH --array=0-1,3-5
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/ablation_%a_%A.out
#SBATCH --error=log/ablation_%a_%A.err

# Ablation: reward weight sensitivity (w_BDE, w_IP), w_RRAB fixed at 0.5
# All other params are locked in conf/experiment/ablation_rw.yaml
#
# Array ID | w_BDE | w_IP
# ---------|-------|------
#    0     |  1.0  |  0.0
#    1     |  0.9  |  0.1
#    2     |  0.8  |  0.2   (baseline, running on login node)
#    3     |  0.7  |  0.3
#    4     |  0.6  |  0.4
#    5     |  0.5  |  0.5

set -euo pipefail

W_BDE=(1.0 0.9 0.8 0.7 0.6 0.5)
W_IP=(0.0 0.1 0.2 0.3 0.4 0.5)

IDX=${SLURM_ARRAY_TASK_ID}
WBDE=${W_BDE[$IDX]}
WIP=${W_IP[$IDX]}

echo "============================================"
echo "Ablation: w_BDE=${WBDE}, w_IP=${WIP}"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${IDX}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=${IDX} \
    "reward.reward_weight=[${WBDE},${WIP},0.5]" \
    mp_master_port=$((6600 + IDX))

echo "Ablation trial=${IDX} completed at $(date)"
