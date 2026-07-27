#!/bin/bash
# CLEAN CONTROL for the rrab confound. The BDE_IP final-molecule comparison reversed because
# rrab is an additive bonus for SHRINKING and the better-ranking agent harvests it harder
# (final molecules 9.89 vs 13.92 heavy atoms, n_unique 0.19 vs 0.59). The QED reward has NO
# rrab, so if gnn_distill's ranking advantage is genuine it should survive here on the SAME
# strict metric: the final molecule of the final episode.
# NOTE: production QED is 0.8*QED - 0.2*SA (src/reward/rewards/qed.py:28), so there is a weak
# IMPLICIT size preference through SA (smaller molecules tend to score lower SA, and SA is
# subtracted) -- but it is a pure molecule function, unlike rrab which depends on the start.
# Fewer ranks than the bde_ip matrix so the two can share the node; QED's oracle is free.
cd "$(dirname "$0")/.."
export MASTER_ADDR=127.0.0.1
NR=${NR:-8}; ITER=${ITER:-2500}; STEPS=${STEPS:-10}; NMOL=${NMOL:-32}
COMMON="reward=qed launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
mols.init_mol_path=Data/zinc_10000.txt mols.num_init_mol=${NMOL} \
train.max_steps_per_episode=${STEPS} train.iteration=${ITER} train.eps_decay=0.968 \
env.maintain_OH=null experiment.experiment=qedc"
run() { tag=$1; trial=$2; off=$3; shift 3
  [ -s "Experiments/qedc_${trial}/qedc_${trial}.pickle.gz" ] && { echo "skip $tag"; return; }
  export MASTER_PORT=$(( 25000 + (trial % 4000) ))
  echo "$(date +%H:%M:%S) START $tag trial=$trial"
  srun --ntasks=$NR --cpus-per-task=1 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py $COMMON \
    experiment.trial=$trial dist.seed_offset=$off "$@" > "logs_qedc_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }
for s in 0 1 2; do
  off=$(( 3000 + s * 100 ))
  run "base_s${s}" $((9300+s*2)) $off env.observation=list
  run "gnn_s${s}"  $((9301+s*2)) $off env.observation=gnn \
      env.gnn_ckpt=gnn_models/gnn_qed.pt train.aux_distill=1.0
done
echo QEDCDONE
