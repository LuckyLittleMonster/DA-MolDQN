#!/bin/bash
# Submit all Phase 1 training jobs.
# SLURM maple QOS: 3 GPU limit, so submit in batches.
#
# Phase 1: Training (500 episodes, 64 train mols, offset=0)
#   T1:  route  + sEH   + dock_baseline  (matches RxnFlow)
#   T2:  route  + sEH   + multi          (MOO product)
#   T3:  route  + DRD2  + dock           (matches RGFN)
#   T4:  route  + DRD2  + multi          (MOO product)
#   T5:  route  + GSK3β + dock           (matches HN-GFN)
#   T6:  route  + GSK3β + multi          (MOO product)
#   T7:  reasyn + sEH   + dock_baseline  (matches RxnFlow)
#   T8:  reasyn + sEH   + multi          (MOO product)
#   T9:  reasyn + DRD2  + dock           (matches RGFN)
#   T10: reasyn + DRD2  + multi          (MOO product)
#   T11: reasyn + GSK3β + dock           (matches HN-GFN)
#   T12: reasyn + GSK3β + multi          (MOO product)
#
# Priority: Route first (fast ~4h), then ReaSyn (slow ~12h)

set -euo pipefail
cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

SCRIPTS=scripts/experiments

echo "=== Phase 1: Route-DQN Training (T1-T6) ==="
echo "Submitting batch 1 (3 jobs, ~8h)..."
J1=$(sbatch --parsable ${SCRIPTS}/train_route.sh seh dock_baseline 1)
J2=$(sbatch --parsable ${SCRIPTS}/train_route.sh seh multi 1)
J3=$(sbatch --parsable ${SCRIPTS}/train_route.sh drd2 dock 1)
echo "  T1: route+sEH+dock_baseline  -> ${J1}"
echo "  T2: route+sEH+multi          -> ${J2}"
echo "  T3: route+DRD2+dock          -> ${J3}"

echo ""
echo "Submitting batch 2 (after batch 1 completes)..."
J4=$(sbatch --parsable --dependency=afterany:${J1}:${J2}:${J3} ${SCRIPTS}/train_route.sh drd2 multi 1)
J5=$(sbatch --parsable --dependency=afterany:${J1}:${J2}:${J3} ${SCRIPTS}/train_route.sh gsk3b dock 1)
J6=$(sbatch --parsable --dependency=afterany:${J1}:${J2}:${J3} ${SCRIPTS}/train_route.sh gsk3b multi 1)
echo "  T4: route+DRD2+multi         -> ${J4}"
echo "  T5: route+GSK3β+dock         -> ${J5}"
echo "  T6: route+GSK3β+multi        -> ${J6}"

echo ""
echo "=== Phase 1: ReaSyn-DQN Training (T7-T12) ==="
echo "Submitting batch 3 (after route batch 2)..."
J7=$(sbatch --parsable --dependency=afterany:${J4}:${J5}:${J6} ${SCRIPTS}/train_reasyn.sh seh dock_baseline 1)
J8=$(sbatch --parsable --dependency=afterany:${J4}:${J5}:${J6} ${SCRIPTS}/train_reasyn.sh seh multi 1)
J9=$(sbatch --parsable --dependency=afterany:${J4}:${J5}:${J6} ${SCRIPTS}/train_reasyn.sh drd2 dock 1)
echo "  T7:  reasyn+sEH+dock_baseline -> ${J7}"
echo "  T8:  reasyn+sEH+multi         -> ${J8}"
echo "  T9:  reasyn+DRD2+dock         -> ${J9}"

echo ""
echo "Submitting batch 4 (after batch 3)..."
J10=$(sbatch --parsable --dependency=afterany:${J7}:${J8}:${J9} ${SCRIPTS}/train_reasyn.sh drd2 multi 1)
J11=$(sbatch --parsable --dependency=afterany:${J7}:${J8}:${J9} ${SCRIPTS}/train_reasyn.sh gsk3b dock 1)
J12=$(sbatch --parsable --dependency=afterany:${J7}:${J8}:${J9} ${SCRIPTS}/train_reasyn.sh gsk3b multi 1)
echo "  T10: reasyn+DRD2+multi        -> ${J10}"
echo "  T11: reasyn+GSK3β+dock        -> ${J11}"
echo "  T12: reasyn+GSK3β+multi       -> ${J12}"

echo ""
echo "=== All 12 Phase 1 jobs submitted ==="
echo "Monitor: squeue -u \$USER"
echo "Route batch 1: ${J1},${J2},${J3}"
echo "Route batch 2: ${J4},${J5},${J6}"
echo "ReaSyn batch 3: ${J7},${J8},${J9}"
echo "ReaSyn batch 4: ${J10},${J11},${J12}"
