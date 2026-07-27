#!/bin/bash
# NOTE: rep_gnn/ckpt/gnn_bdeipprod.pt is not shipped (trained on proprietary BDE/IP
#       labels). Rebuild it with train_combined.py on the `rep` branch before running.
# Multi-seed confirmation of the 32-rank / 128-molecule result. This whole investigation has
# repeatedly found single-seed conclusions to flip sign (BDE 'linear wins', the SAGE smoke
# test), so the +30.8% population gain does not count until it survives several seeds.
# Seeds come from dist.seed_offset (per-rank seed = offset + rank), so the two arms share
# identical RNG streams within a seed and differ ONLY in the Q-network.
cd "$(dirname "$0")/.."
export MASTER_ADDR=127.0.0.1
NR=${NR:-32}; ITER=${ITER:-4000}; STEPS=${STEPS:-20}; NMOL=${NMOL:-128}
COMMON="reward=bde_ip launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
mols.init_mol_path=Data/anti_400.txt mols.num_init_mol=${NMOL} \
train.max_steps_per_episode=${STEPS} train.iteration=${ITER} train.eps_decay=0.968 \
env.etkdg.threads=1 env.maintain_OH=exist experiment.experiment=gnn32s"

run() { tag=$1; trial=$2; off=$3; shift 3
  [ -s "Experiments/gnn32s_${trial}/gnn32s_${trial}.pickle.gz" ] && { echo "skip $tag"; return; }
  export MASTER_PORT=$(( 22000 + (trial % 5000) ))
  echo "$(date +%H:%M:%S) START $tag trial=$trial seed_offset=$off"
  srun --ntasks=$NR --cpus-per-task=2 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py $COMMON \
    experiment.trial=$trial dist.seed_offset=$off "$@" > "logs_gnn32s_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }

for s in 0 1 2; do
  off=$(( 1000 + s * 100 ))
  run "base_s${s}" $(( 9100 + s * 2 ))     $off env.observation=list
  run "gnn_s${s}"  $(( 9101 + s * 2 ))     $off env.observation=gnn \
      env.gnn_ckpt=rep_gnn/ckpt/gnn_bdeipprod.pt train.aux_distill=1.0
done
echo SEEDSDONE
