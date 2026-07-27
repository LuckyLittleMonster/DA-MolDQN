#!/bin/bash
# NOTE: rep_gnn/ckpt/gnn_bdeipprod.pt is not shipped (trained on proprietary BDE/IP
#       labels). Rebuild it with train_combined.py on the `rep` branch before running.
# Standard-parameter matrix (matched to scripts/dev_64rank.sh): 250 episodes x 10 steps,
# 32 ranks x 128 molecules, one GH200.
#   reward=bde_ip   additive rrab (current production)
#   reward=bde_ip2  multiplicative size desirability + clamped scalers
#   observation=list current fingerprint Q-net | gnn  frozen-teacher + prior + distillation
# ip_ensemble=false (cv4 only) is the measured throughput setting: 3.04 s per 10-step episode
# vs 5.73 s for the 5-CV mean, i.e. 12.7 vs 23.9 min per 2500 steps. Profiling showed the run
# is GPU-bound (99% util, CPU load only 23-31 of 72 cores), so the AIMNet ensemble -- not
# ETKDG threading -- was the bottleneck; etkdg.threads made no measurable difference
# (5.890 s at 1 thread vs 5.799 s at 2).
#
# The 2x2 tests three things at once:
#   (a) does gnn_distill's population gain survive at the standard budget,
#   (b) does bde_ip2 cut the invalid-molecule rate (predicted: yes -- #10 showed the rise is
#       rrab rewarding shrinkage, which drives 3-ring/bridged systems ETKDG cannot embed),
#   (c) does bde_ip2 cost any population quality.
cd "$(dirname "$0")/.."
export MASTER_ADDR=127.0.0.1
NR=${NR:-32}; ITER=${ITER:-2500}; STEPS=${STEPS:-10}; NMOL=${NMOL:-128}; SEEDS=${SEEDS:-"0 1 2"}
run() { tag=$1; trial=$2; off=$3; rw=$4; shift 4
  [ -s "Experiments/bi2_${trial}/bi2_${trial}.pickle.gz" ] && { echo "skip $tag"; return; }
  export MASTER_PORT=$(( 23000 + (trial % 5000) ))
  echo "$(date +%H:%M:%S) START $tag trial=$trial seed_off=$off reward=$rw"
  srun --ntasks=$NR --cpus-per-task=2 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py \
      reward=$rw launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
      mols.init_mol_path=Data/anti_400.txt mols.num_init_mol=$NMOL \
      train.max_steps_per_episode=$STEPS train.iteration=$ITER train.eps_decay=0.968 \
      env.etkdg.threads=2 env.maintain_OH=exist reward.ip_ensemble=false \
      experiment.experiment=bi2 experiment.trial=$trial dist.seed_offset=$off \
      "$@" > "logs_bi2_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }

GNN="env.observation=gnn env.gnn_ckpt=rep_gnn/ckpt/gnn_bdeipprod.pt train.aux_distill=1.0"
for s in $SEEDS; do
  off=$(( 2000 + s * 100 ))
  run "r1_base_s${s}" $((9200+s*4))   $off bde_ip  env.observation=list
  run "r1_gnn_s${s}"  $((9201+s*4))   $off bde_ip  $GNN
  run "r2_base_s${s}" $((9202+s*4))   $off bde_ip2 env.observation=list
  run "r2_gnn_s${s}"  $((9203+s*4))   $off bde_ip2 $GNN
done
echo BI2DONE
