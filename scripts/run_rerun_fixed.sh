#!/bin/bash
# NOTE: rep_gnn/ckpt/gnn_bdeipprod.pt is not shipped (trained on proprietary BDE/IP
#       labels). Rebuild it with train_combined.py on the `rep` branch before running.
# RERUN with the fixed r_hat (extra_fn wired, verified Spearman 0.363 -> 0.980 against the
# true QED reward and EXACT for BDE_IP's rrab). Everything measured before this fix is a
# defective-r_hat baseline: the teacher only modelled the property, so the better-ranking
# agent drifted on the unmodelled size term -- shrinking on BDE_IP (rrab rewards it) and
# growing on QED (SA penalises it).
#
# Primary metric is the FINAL molecule of the FINAL episode (scripts/analyze_final_mol.py),
# not pop_mean or top-k: those are whole-run maxima dominated by lucky epsilon-greedy steps.
# Waits for the in-flight defective-r_hat runs so the node is not oversubscribed.
cd "$(dirname "$0")/.."
while pgrep -f "run_bdeip2_matrix|run_qed_control" > /dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) previous matrices finished, starting fixed-r_hat reruns"
export MASTER_ADDR=127.0.0.1
NR=32; STEPS=10; ITER=2500; NMOL=128
COMMON="launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
train.max_steps_per_episode=$STEPS train.iteration=$ITER train.eps_decay=0.968 \
env.etkdg.threads=2 reward.ip_ensemble=false experiment.experiment=fix"
GNN="env.observation=gnn train.aux_distill=1.0"
run() { tag=$1; trial=$2; off=$3; shift 3
  [ -s "Experiments/fix_${trial}/fix_${trial}.pickle.gz" ] && { echo "skip $tag"; return; }
  export MASTER_PORT=$(( 26000 + (trial % 4000) ))
  echo "$(date +%H:%M:%S) START $tag trial=$trial"
  srun --ntasks=$NR --cpus-per-task=2 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py $COMMON \
    experiment.trial=$trial dist.seed_offset=$off "$@" > "logs_fix_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }
for s in 0 1 2; do
  off=$(( 4000 + s * 100 ))
  A="reward=bde_ip mols.init_mol_path=Data/anti_400.txt mols.num_init_mol=$NMOL env.maintain_OH=exist"
  B="reward=bde_ip2 mols.init_mol_path=Data/anti_400.txt mols.num_init_mol=$NMOL env.maintain_OH=exist"
  run "ip_base_s$s"  $((9400+s*6)) $off $A env.observation=list
  run "ip_gnn_s$s"   $((9401+s*6)) $off $A $GNN env.gnn_ckpt=rep_gnn/ckpt/gnn_bdeipprod.pt
  run "ip2_base_s$s" $((9402+s*6)) $off $B env.observation=list
  run "ip2_gnn_s$s"  $((9403+s*6)) $off $B $GNN env.gnn_ckpt=rep_gnn/ckpt/gnn_bdeipprod.pt
done
for s in 0 1 2; do
  off=$(( 4000 + s * 100 ))
  Q="reward=qed mols.init_mol_path=Data/zinc_10000.txt mols.num_init_mol=$NMOL env.maintain_OH=null"
  run "qed_base_s$s" $((9404+s*6)) $off $Q env.observation=list
  run "qed_gnn_s$s"  $((9405+s*6)) $off $Q $GNN env.gnn_ckpt=gnn_models/gnn_qed.pt
done
echo FIXDONE
