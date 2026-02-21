#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=dpp
#SBATCH --time=0:15:00
#SBATCH --output=Experiments/logs/rescore_route_%j.out
#SBATCH --error=Experiments/logs/rescore_route_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "Re-score route_multi_1 products (SA + Dock) and regenerate txt"
echo "Node: $(hostname) | Date: $(date)"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python scripts/rescore_route_txt.py

echo ""
echo "Done at $(date)"
