#!/bin/bash
# NOTE: rep_gnn/ckpt/gnn_bdeipprod.pt is not shipped (trained on proprietary BDE/IP
#       labels). Rebuild it with train_combined.py on the `rep` branch before running.
# 32-rank / 128-molecule comparison on ONE GH200: canonical fingerprint MolDQN vs the frozen
# property-GNN teacher + zero-init prior head + candidate-set distillation (ported from the
# rep_gnn study). Identical reward (exact production BDE_IP), identical init molecules
# (Data/anti_400.txt) and budget -- the ONLY variable is the Q-network observation + head.
cd "$(dirname "$0")/.."
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 21000 + (RANDOM % 5000) ))
NR=${NR:-32}; ITER=${ITER:-4000}; STEPS=${STEPS:-20}; NMOL=${NMOL:-128}
COMMON="reward=bde_ip launcher=slurm dist.backend=gloo mols.gpu_list=[0] \
mols.init_mol_path=Data/anti_400.txt mols.num_init_mol=${NMOL} \
train.max_steps_per_episode=${STEPS} train.iteration=${ITER} train.eps_decay=0.968 \
env.etkdg.threads=1 env.maintain_OH=exist experiment.experiment=gnn32"

run() { tag=$1; trial=$2; shift 2
  echo "$(date +%H:%M:%S) START $tag (trial=$trial)"
  srun --ntasks=$NR --cpus-per-task=2 --gres=gpu:1 --overlap \
    conda run -n rl4 --no-capture-output python train.py $COMMON \
    experiment.trial=$trial "$@" > "logs_gnn32_${tag}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE  $tag rc=$?"; }

run baseline ${TRIAL:-9001} env.observation=list
run gnn       $(( ${TRIAL:-9001} + 1 )) env.observation=gnn \
  env.gnn_ckpt=rep_gnn/ckpt/gnn_bdeipprod.pt train.aux_distill=1.0
echo GNN32DONE
