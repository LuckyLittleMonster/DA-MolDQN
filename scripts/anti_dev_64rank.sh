#!/bin/bash
#SBATCH --job-name=anti_dev
#SBATCH --account=maple
#SBATCH --partition=maple
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=anti_dev_%j.log

# dev 2.0 BDE_IP validation: 256 mols (64 ranks x 4), 2500 steps, eps_decay 0.968, 2x GH200.
cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p tem
RDV="file:///shared/data1/Users/l1062811/git/DA-MolDQN/tem/sharedfile_${SLURM_JOB_ID}"

srun conda run -n rl4 --no-capture-output python train.py \
  reward=bde_ip \
  launcher=slurm \
  backend=gloo \
  gpu_list='[0]' \
  init_mol_path=Data/anti_400.txt \
  num_init_mol=256 \
  max_steps_per_episode=10 \
  iteration=2500 \
  eps_decay=0.968 \
  maintain_OH=exist \
  init_method="${RDV}" \
  experiment=anti_dev \
  trial=1
