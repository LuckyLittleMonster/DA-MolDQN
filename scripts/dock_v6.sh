#!/bin/bash
#SBATCH --job-name=dock_v6
#SBATCH --partition=maple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=app
#SBATCH --output=/tmp/dock_v6_%j.log

source ~/.bashrc_maple 2>/dev/null
conda activate rl4

cd /shared/data1/Users/l1062811/git/DA-MolDQN

PYTHONUNBUFFERED=1 python main_sync.py \
    --env_type docking \
    --receptor_pdbqt docking/targets/1iep/receptor.pdbqt \
    --dock_config docking/targets/1iep/config.json \
    --dock_weight 0.7 \
    --sa_weight_dock 0.3 \
    --dock_search_mode fast \
    --reactant_method template \
    --iteration 300 \
    --max_steps_per_episode 3 \
    --num_molecules 64 \
    --init_mol_path template/data/zinc_first64.smi \
    --experiment dock_v6 \
    --trial 1
