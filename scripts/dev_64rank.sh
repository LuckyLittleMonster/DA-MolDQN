#!/bin/bash
#SBATCH --job-name=dev_run
#SBATCH --account=app
#SBATCH --partition=maple
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=Experiments/slurm_%x_%j.log

# Parametrized dev 2.0 run. Override via sbatch --export:
#   REWARD (bde_ip|qed|plogp), EXP, TRIAL, MAINTAIN_OH (exist|null)
REWARD=${REWARD:-bde_ip}
EXP=${EXP:-anti_dev}
TRIAL=${TRIAL:-1}
MAINTAIN_OH=${MAINTAIN_OH:-exist}

cd /shared/data1/Users/l1062811/git/DA-MolDQN

# DDP rendezvous: TCP/env:// (no shared-file dependency, no stale files).
# MASTER_ADDR = first allocated node; MASTER_PORT = job-derived free port.
# These are exported to every srun task; launcher uses init_method=None -> env://.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
echo "rendezvous env://  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

srun conda run -n rl4 --no-capture-output python train.py \
  reward="${REWARD}" \
  launcher=slurm \
  backend=gloo \
  gpu_list='[0]' \
  init_mol_path=Data/anti_400.txt \
  num_init_mol=256 \
  max_steps_per_episode=10 \
  iteration=2500 \
  eps_decay=0.968 \
  maintain_OH="${MAINTAIN_OH}" \
  experiment="${EXP}" \
  trial="${TRIAL}"

# Move the SLURM log into the unified run folder.
mv "Experiments/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" "Experiments/${EXP}_${TRIAL}/" 2>/dev/null || true
