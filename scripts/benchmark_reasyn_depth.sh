#!/bin/bash
#SBATCH --job-name=bench_depth
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=dpp
#SBATCH --time=0:30:00
#SBATCH --output=Experiments/logs/benchmark_reasyn_depth_%j.out
#SBATCH --error=Experiments/logs/benchmark_reasyn_depth_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "ReaSyn Depth Benchmark"
echo "Node: $(hostname) | Date: $(date)"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python scripts/benchmark_reasyn_depth.py

echo "Done at $(date)"
