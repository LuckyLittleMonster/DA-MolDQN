#!/bin/bash
#SBATCH --job-name=rl_matrix
#SBATCH --output=Experiments/logs/rl_matrix_%A_%a.out
#SBATCH --error=Experiments/logs/rl_matrix_%A_%a.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-6%2

# ============================================================
# RL Experiment Matrix: 6 systematic comparisons
# ============================================================
# Submit: sbatch scripts/rl_experiment_matrix.sh
# Monitor: squeue -u $USER
# Analyze: python scripts/analyze_experiment_matrix.py
#
# Core comparisons (Exp 1-3):
#   E1: BASELINE_FIXED — hypergraph 2-step + filter (reference)
#   E2: AIO_BASIC — AIO replaces V3 (key comparison)
#   E3: AIO_REWARD_SHAPED — AIO + reward shaping (expected best)
#
# Ablations (Exp 4-6):
#   E4: BASELINE_REWARD_SHAPED — reward shaping on 2-step
#   E5: AIO_NO_FILTER — filter ablation
#   E6: AIO_TOPK5 — top_k sensitivity (k=5 vs k=20 in E2)
#
# Key comparisons:
#   E1 vs E2: hypergraph 2-step vs AIO (same filter, same top_k)
#   E2 vs E3: effect of reward shaping (AIO)
#   E1 vs E4: effect of reward shaping (2-step)
#   E2 vs E5: value of product filter (AIO)
#   E2 vs E6: top_k=20 vs top_k=5 (AIO)
#   E3 vs E4: best AIO vs best 2-step
# ============================================================

export PYTHONUNBUFFERED=1

source ~/.bashrc_maple 2>/dev/null
conda activate rl4

echo "=========================================="
echo "RL Experiment Matrix - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs Experiments/models

# Common parameters
ITERATION=50000
MAX_STEPS=10
BATCH_SIZE=128
LR=1e-4
SAVE_FREQ=2000
LOG_FREQ=50
INIT_MOLS="CCO c1ccccc1O CC(=O)O c1ccc(N)cc1 c1ccc(O)cc1O CC(C)O CCN CCCO"

case ${SLURM_ARRAY_TASK_ID} in
  1)
    EXP_NAME="E1_baseline_fixed"
    METHOD="hypergraph"
    TOP_K=20
    EXTRA_ARGS=""
    ;;
  2)
    EXP_NAME="E2_aio_basic"
    METHOD="aio"
    TOP_K=20
    EXTRA_ARGS=""
    ;;
  3)
    EXP_NAME="E3_aio_reward_shaped"
    METHOD="aio"
    TOP_K=20
    EXTRA_ARGS="--reward_tanimoto_bonus 0.1 --reward_mw_penalty 0.001"
    ;;
  4)
    EXP_NAME="E4_baseline_reward_shaped"
    METHOD="hypergraph"
    TOP_K=20
    EXTRA_ARGS="--reward_tanimoto_bonus 0.1"
    ;;
  5)
    EXP_NAME="E5_aio_no_filter"
    METHOD="aio"
    TOP_K=20
    EXTRA_ARGS="--no_product_filter"
    ;;
  6)
    EXP_NAME="E6_aio_topk5"
    METHOD="aio"
    TOP_K=5
    EXTRA_ARGS=""
    ;;
  *)
    echo "Unknown array task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "============================================================"
echo "Experiment: ${EXP_NAME}"
echo "  Method: ${METHOD}, top_k: ${TOP_K}, max_steps: ${MAX_STEPS}"
echo "  Iterations: ${ITERATION}, batch: ${BATCH_SIZE}"
echo "  Init mols: ${INIT_MOLS}"
echo "  Extra args: ${EXTRA_ARGS}"
echo "============================================================"

python main_sync.py \
    --experiment ${EXP_NAME} \
    --trial ${SLURM_ARRAY_JOB_ID} \
    --iteration ${ITERATION} \
    --max_steps_per_episode ${MAX_STEPS} \
    --reactant_method ${METHOD} \
    --hypergraph_top_k ${TOP_K} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --reaction_only \
    --save_freq ${SAVE_FREQ} \
    --log_freq ${LOG_FREQ} \
    --gpu 0 \
    --init_mol ${INIT_MOLS} \
    ${EXTRA_ARGS}

echo ""
echo "Training complete at $(date)"
echo "=========================================="
