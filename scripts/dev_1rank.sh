#!/bin/bash
#SBATCH --job-name=dev_run1
#SBATCH --account=app
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=Experiments/slurm_%x_%j.log

# Single-rank dev run (QED or BDE_IP). Override via sbatch --export:
#   REWARD (bde_ip|qed|plogp), EXP, TRIAL, MAINTAIN_OH (exist|null), NMOL, DATA
REWARD=${REWARD:-bde_ip}
EXP=${EXP:-dev_1rank}
TRIAL=${TRIAL:-1}
MAINTAIN_OH=${MAINTAIN_OH:-exist}
NMOL=${NMOL:-256}
DATA=${DATA:-Data/anti_400.txt}

cd /shared/data1/Users/l1062811/git/DA-MolDQN

# DDP rendezvous (env://) — world_size=1 still goes through init_process_group.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
echo "rendezvous env://  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

srun conda run -n rl4 --no-capture-output python train.py \
  reward="${REWARD}" \
  launcher=slurm \
  backend=gloo \
  gpu_list='[0]' \
  init_mol_path="${DATA}" \
  num_init_mol="${NMOL}" \
  max_steps_per_episode=10 \
  iteration=2500 \
  eps_decay=0.968 \
  etkdg_threads=2 \
  maintain_OH="${MAINTAIN_OH}" \
  experiment="${EXP}" \
  trial="${TRIAL}"

# Move the SLURM log into the unified run folder.
mv "Experiments/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" "Experiments/${EXP}_${TRIAL}/" 2>/dev/null || true
