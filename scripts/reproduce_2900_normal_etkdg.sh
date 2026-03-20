#!/bin/bash
#SBATCH --job-name=repr_2900n
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --qos=part_maple_night
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/repr_2900n_%j.out
#SBATCH --error=log/repr_2900n_%j.err

###############################################################################
# Reproduce trial_2900: 64-rank DDP with NORMAL ETKDG
#
# CHANGE FROM reproduce_2900.sh:
#   ETKDG mode: turbo → normal
#     turbo:  max_iterations=0, timeout=1, max_attempts=1
#     normal: max_iterations=500, timeout=0, max_attempts=7
#
# RATIONALE:
#   The original trial_2900 used old code with maxAttempts=7 (EmbedMolecule),
#   equivalent to normal ETKDG mode. The turbo run (job 712099) gave
#   mean reward 1.2896 at ep224, worse than single-process (1.3206).
#   This run tests whether ETKDG quality affects reward.
#
# EXPECTED:
#   If ETKDG matters: mean reward closer to trial_2900 (~1.65)
#   If not: similar to turbo run (~1.29)
#
###############################################################################

set -euo pipefail

echo "============================================"
echo "Reproduce trial_2900: 64-rank DDP + normal ETKDG"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "============================================"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

srun --kill-on-bad-exit=1 ${PYTHON} main_hpc.py \
    +experiment=ablation_rw \
    trial=2901 \
    iteration=2500 \
    num_init_mol=256 \
    max_batch_size=512 \
    starter=slurm \
    backend=gloo \
    etkdg=normal \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250

echo "Completed at $(date)"
