#!/bin/bash
#SBATCH --job-name=ip_error
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --account=dpp
#SBATCH --output=scripts/ip_error_%j.log

cd /shared/data1/Users/l1062811/git/DA-MolDQN
conda run -n rl4 --live-stream python scripts/compute_ip_error.py
