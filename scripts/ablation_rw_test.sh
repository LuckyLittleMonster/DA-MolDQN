#!/bin/bash
#SBATCH --job-name=abl_rw
#SBATCH --array=0-4
#SBATCH --partition=maple_night
#SBATCH --account=dpp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=Experiments/logs/abl_rw_%a_%A.out
#SBATCH --error=Experiments/logs/abl_rw_%a_%A.err

###############################################################################
# Ablation: reward weight sensitivity (w_BDE, w_IP), w_RRAB fixed at 0.5
# Reviewer R1#10, R2#3: justify the 0.8/0.2 split
#
# Array ID | w_BDE | w_IP | Trial
# ---------|-------|------|------
#    0     |  1.0  |  0.0 | 7000
#    1     |  0.9  |  0.1 | 7001
#    2     |  0.7  |  0.3 | 7003   (skip 0.8/0.2 = trial 6001 baseline)
#    3     |  0.6  |  0.4 | 7004
#    4     |  0.5  |  0.5 | 7005
###############################################################################

set -euo pipefail

W_BDE=(1.0 0.9 0.7 0.6 0.5)
W_IP=(0.0 0.1 0.3 0.4 0.5)
TRIALS=(7000 7001 7003 7004 7005)

IDX=${SLURM_ARRAY_TASK_ID}
WBDE=${W_BDE[$IDX]}
WIP=${W_IP[$IDX]}
TRIAL=${TRIALS[$IDX]}

echo "============================================"
echo "Ablation: w_BDE=${WBDE}, w_IP=${WIP}, trial=${TRIAL}"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${IDX}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

# Create rendezvous directory
RDZV_DIR="/shared/data1/Users/l1062811/git/DA-MolDQN/Experiments/rdzv"
mkdir -p ${RDZV_DIR}
rm -f ${RDZV_DIR}/sharedfile_test_alfabet_${TRIAL}

srun --kill-on-bad-exit=1 ${PYTHON} -u main_hpc.py \
    --trial ${TRIAL} \
    --experiment test_alfabet \
    --iteration 2500 \
    --max_steps_per_episode 10 \
    --init_mol_path ./Data/anti_400.txt \
    --num_init_mol 256 \
    --discount_factor 1.0 \
    --reward BDE_IP \
    --reward_weight ${WBDE} ${WIP} 0.5 \
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

echo "Trial ${TRIAL} completed at $(date)"
