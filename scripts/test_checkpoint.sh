#!/bin/bash
# Test a trained checkpoint on anti_400 first 256 mols (1 episode, greedy)
# Usage: bash scripts/test_checkpoint.sh <checkpoint_name> [trial_id]
# Example:
#   bash scripts/test_checkpoint.sh trial_573 test_573
#   bash scripts/test_checkpoint.sh trial_29000 test_29000

set -euo pipefail

CHECKPOINT=${1:?Usage: $0 <checkpoint_name> [trial_id]}
TRIAL=${2:-test_${CHECKPOINT}}

PYTHON="/home/l1062811/data/envs/rl4/bin/python"

echo "Testing checkpoint: ${CHECKPOINT}"
echo "Trial ID: ${TRIAL}"
echo "Mols: anti_400[0:256]"

cd /shared/data1/Users/l1062811/git/DA-MolDQN

${PYTHON} -u main_hpc.py \
    +experiment=ablation_rw \
    trial=${TRIAL} \
    test=true \
    checkpoint=${CHECKPOINT} \
    starter=null \
    backend=gloo \
    num_init_mol=256
