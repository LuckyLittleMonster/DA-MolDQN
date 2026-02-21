#!/bin/bash
#SBATCH --job-name=rea_smoke
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=dpp
#SBATCH --time=0:30:00
#SBATCH --output=Experiments/logs/reasyn_smoke_deep_%j.out
#SBATCH --error=Experiments/logs/reasyn_smoke_deep_%j.err

set -euo pipefail

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

source ~/.bashrc_maple 2>/dev/null || true

echo "Smoke test: ReaSyn + QED + Deep synthesis post-training"
echo "Node: $(hostname) | Date: $(date)"

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream \
    python main.py \
        method=reasyn \
        reward=qed \
        exp_name=reasyn_smoke_deep \
        trial=0 \
        episodes=3 \
        max_steps=2 \
        num_molecules=4 \
        method.num_workers=0 \
        method.use_cache=false \
        log_freq=1

echo ""
echo "Done at $(date)"
