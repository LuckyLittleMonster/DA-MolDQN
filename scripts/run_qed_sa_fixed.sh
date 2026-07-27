#!/bin/bash
# DIRECT TEST of the r_hat-coverage hypothesis on the metric that matters.
# The QED control showed gnn_distill wins PURE QED (+10.4%, 3/3 seeds) -- the quantity its
# teacher models -- but loses the production reward 0.8*QED - 0.2*SA, because the teacher
# never saw SA: molecules grew 16.3 -> 18.6 heavy atoms, SA worsened 19.4%, and the penalty
# ate the gain. r_hat now includes -0.2*SA (Spearman vs the true reward 0.374 -> 0.981).
# If the hypothesis is right, the SAME arm should now stop growing molecules and win the
# PRODUCTION reward on the final molecule of the final episode.
# Compared against the already-completed defective-r_hat runs (qedc_930*), same seeds/offsets.
cd "$(dirname "$0")/.."
export MASTER_ADDR=127.0.0.1
NR=${NR:-8}; NMOL=${NMOL:-32}
COMMON="reward=qed launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
mols.init_mol_path=Data/zinc_10000.txt mols.num_init_mol=${NMOL} \
train.max_steps_per_episode=10 train.iteration=2500 train.eps_decay=0.968 \
env.maintain_OH=null experiment.experiment=qedsa"
run() { tag=$1; trial=$2; off=$3; shift 3
  [ -s "Experiments/qedsa_${trial}/qedsa_${trial}.pickle.gz" ] && { echo "skip $tag"; return; }
  export MASTER_PORT=$(( 27000 + (trial % 4000) ))
  echo "$(date +%H:%M:%S) START $tag trial=$trial"
  srun --ntasks=$NR --cpus-per-task=1 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py $COMMON \
    experiment.trial=$trial dist.seed_offset=$off "$@" > "logs_qedsa_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }
for s in 0 1 2; do
  off=$(( 3000 + s * 100 ))          # SAME offsets as the qedc runs -> paired comparison
  run "gnn_s${s}" $((9500+s)) $off env.observation=gnn \
      env.gnn_ckpt=gnn_models/gnn_qed.pt train.aux_distill=1.0
done
echo QEDSADONE
