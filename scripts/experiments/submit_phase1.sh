#!/bin/bash
# Submit all Phase 1 training jobs.
# Using maple_night QOS: no GPU limit, all jobs can run in parallel.
#
# Phase 1: Training (500 episodes, 64 train mols, offset=0)
#   T1:  route  + sEH   + dock_rxnflow (matches RxnFlow: 0.5*QED + 0.5*Vina)
#   T2:  route  + sEH   + multi        (MOO product)
#   T3:  route  + DRD2  + dock         (matches RGFN: pure dock)
#   T4:  route  + DRD2  + multi        (MOO product)
#   T5:  route  + GSK3β + dock         (matches HN-GFN: pure dock)
#   T6:  route  + GSK3β + multi        (MOO product)
#   T7:  reasyn + sEH   + dock_rxnflow (matches RxnFlow: 0.5*QED + 0.5*Vina)
#   T8:  reasyn + sEH   + multi        (MOO product)
#   T9:  reasyn + DRD2  + dock         (matches RGFN: pure dock)
#   T10: reasyn + DRD2  + multi        (MOO product)
#   T11: reasyn + GSK3β + dock         (matches HN-GFN: pure dock)
#   T12: reasyn + GSK3β + multi        (MOO product)
#   T13: route  + QED                  (pure QED, no target)
#   T14: reasyn + QED                  (pure QED, no target)

set -euo pipefail
cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs

SCRIPTS=scripts/experiments

echo "=== Phase 1: Route-DQN Training (T1-T6, T13) ==="
J1=$(sbatch --parsable ${SCRIPTS}/train_route.sh seh dock_rxnflow 1)
J2=$(sbatch --parsable ${SCRIPTS}/train_route.sh seh multi 1)
J3=$(sbatch --parsable ${SCRIPTS}/train_route.sh drd2 dock 1)
J4=$(sbatch --parsable ${SCRIPTS}/train_route.sh drd2 multi 1)
J5=$(sbatch --parsable ${SCRIPTS}/train_route.sh gsk3b dock 1)
J6=$(sbatch --parsable ${SCRIPTS}/train_route.sh gsk3b multi 1)
J13=$(sbatch --parsable ${SCRIPTS}/train_route.sh - qed 1)
echo "  T1:  route+sEH+dock_rxnflow   -> ${J1}"
echo "  T2:  route+sEH+multi          -> ${J2}"
echo "  T3:  route+DRD2+dock          -> ${J3}"
echo "  T4:  route+DRD2+multi         -> ${J4}"
echo "  T5:  route+GSK3β+dock         -> ${J5}"
echo "  T6:  route+GSK3β+multi        -> ${J6}"
echo "  T13: route+QED                -> ${J13}"

echo ""
echo "=== Phase 1: ReaSyn-DQN Training (T7-T12, T14) ==="
J7=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh seh dock_rxnflow 1)
J8=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh seh multi 1)
J9=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh drd2 dock 1)
J10=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh drd2 multi 1)
J11=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh gsk3b dock 1)
J12=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh gsk3b multi 1)
J14=$(sbatch --parsable ${SCRIPTS}/train_reasyn.sh - qed 1)
echo "  T7:  reasyn+sEH+dock_rxnflow -> ${J7}"
echo "  T8:  reasyn+sEH+multi        -> ${J8}"
echo "  T9:  reasyn+DRD2+dock        -> ${J9}"
echo "  T10: reasyn+DRD2+multi       -> ${J10}"
echo "  T11: reasyn+GSK3β+dock       -> ${J11}"
echo "  T12: reasyn+GSK3β+multi      -> ${J12}"
echo "  T14: reasyn+QED              -> ${J14}"

echo ""
echo "=== All 14 Phase 1 jobs submitted (parallel, no QOS limit) ==="
echo "Monitor: squeue -u \$USER"
echo "Route:  ${J1},${J2},${J3},${J4},${J5},${J6},${J13}"
echo "ReaSyn: ${J7},${J8},${J9},${J10},${J11},${J12},${J14}"
