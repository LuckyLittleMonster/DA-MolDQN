#!/bin/bash
# Submit all Phase 1b random baseline jobs.
# 12 jobs: 6 route + 6 reasyn, each 100 episodes with eps=1.0 always.

set -euo pipefail
cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

SCRIPTS=scripts/experiments

echo "=== Phase 1b: Random Baselines ==="

echo "Route random baselines (R1-R6)..."
R1=$(sbatch --parsable ${SCRIPTS}/random_baseline.sh route seh dock_baseline 1)
R2=$(sbatch --parsable ${SCRIPTS}/random_baseline.sh route seh multi 1)
R3=$(sbatch --parsable ${SCRIPTS}/random_baseline.sh route drd2 dock 1)
echo "  R1: route+sEH+dock_baseline  -> ${R1}"
echo "  R2: route+sEH+multi          -> ${R2}"
echo "  R3: route+DRD2+dock          -> ${R3}"

R4=$(sbatch --parsable --dependency=afterany:${R1}:${R2}:${R3} ${SCRIPTS}/random_baseline.sh route drd2 multi 1)
R5=$(sbatch --parsable --dependency=afterany:${R1}:${R2}:${R3} ${SCRIPTS}/random_baseline.sh route gsk3b dock 1)
R6=$(sbatch --parsable --dependency=afterany:${R1}:${R2}:${R3} ${SCRIPTS}/random_baseline.sh route gsk3b multi 1)
echo "  R4: route+DRD2+multi         -> ${R4}"
echo "  R5: route+GSK3β+dock         -> ${R5}"
echo "  R6: route+GSK3β+multi        -> ${R6}"

echo ""
echo "ReaSyn random baselines (R7-R12)..."
R7=$(sbatch --parsable --dependency=afterany:${R4}:${R5}:${R6} ${SCRIPTS}/random_baseline.sh reasyn seh dock_baseline 1)
R8=$(sbatch --parsable --dependency=afterany:${R4}:${R5}:${R6} ${SCRIPTS}/random_baseline.sh reasyn seh multi 1)
R9=$(sbatch --parsable --dependency=afterany:${R4}:${R5}:${R6} ${SCRIPTS}/random_baseline.sh reasyn drd2 dock 1)
echo "  R7:  reasyn+sEH+dock_baseline -> ${R7}"
echo "  R8:  reasyn+sEH+multi         -> ${R8}"
echo "  R9:  reasyn+DRD2+dock         -> ${R9}"

R10=$(sbatch --parsable --dependency=afterany:${R7}:${R8}:${R9} ${SCRIPTS}/random_baseline.sh reasyn drd2 multi 1)
R11=$(sbatch --parsable --dependency=afterany:${R7}:${R8}:${R9} ${SCRIPTS}/random_baseline.sh reasyn gsk3b dock 1)
R12=$(sbatch --parsable --dependency=afterany:${R7}:${R8}:${R9} ${SCRIPTS}/random_baseline.sh reasyn gsk3b multi 1)
echo "  R10: reasyn+DRD2+multi        -> ${R10}"
echo "  R11: reasyn+GSK3β+dock        -> ${R11}"
echo "  R12: reasyn+GSK3β+multi       -> ${R12}"

echo ""
echo "=== All 12 random baseline jobs submitted ==="
