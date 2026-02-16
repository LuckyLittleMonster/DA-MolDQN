#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --output=Experiments/logs/eval_full_%A_%a.out
#SBATCH --error=Experiments/logs/eval_full_%A_%a.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=1-4

# ============================================================
# FULL Dataset Retrieval Benchmark
# ============================================================
# Submit: sbatch scripts/eval_full_retrieval.sh
# Monitor: squeue -u $USER
#
# Methods (array tasks):
#   1: fingerprint  — Tanimoto FP baseline (CPU-heavy, no model)
#   2: v3           — V3 CF-NCN link predictor (embedding retrieval)
#   3: aio          — AIO DirectedHypergraphNet Stage 1 (co-reactant prediction)
#   4: aio_v3_rerank — AIO initial retrieval + V3 NCN/CF-NCN re-ranking
#
# Dataset: Data/uspto_full (655K mols, 103K test reactions)
# Queries: 5000 (sampled from ~103K multi-reactant test reactions)
# kNN cache: Data/precomputed/knn_full_k200.pkl (1.98GB)
#
# Prerequisites:
#   - V3 FULL checkpoint: model_reactions/checkpoints/full_A/hypergraph_link_v3_best.pt
#   - AIO FULL checkpoint: hypergraph/checkpoints/directed_predictor_best.pt
#   - kNN cache: Data/precomputed/knn_full_k200.pkl
#   - Reaction DB cache: Data/uspto_full/reaction_db_cache.pkl
#
# Output:
#   Experiments/logs/eval_full_{JOBID}_{1-4}.out (per-method logs)
#   docs/full_retrieval_{method}.md (per-method results)
#   docs/full_retrieval_{method}.pkl (raw results pickle)
# ============================================================

export PYTHONUNBUFFERED=1

source ~/.bashrc_maple 2>/dev/null
conda activate rl4

echo "=========================================="
echo "FULL Retrieval Benchmark - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /shared/data1/Users/l1062811/git/DA-MolDQN
mkdir -p Experiments/logs docs

# ============================================================
# Configuration
# ============================================================
DATA_DIR="Data/uspto_full"
MAX_QUERIES=5000

# Checkpoint paths — update these when FULL training completes
# V3: pick the best LR tuning variant (A=lr1e-4/cosine, B=lr5e-5/cosine, C=lr1e-4/plateau)
V3_CKPT="model_reactions/checkpoints/full_A/hypergraph_link_v3_best.pt"
AIO_CKPT="hypergraph/checkpoints/directed_predictor_best.pt"

# Hybrid re-ranking parameters
RERANK_TOPN=100
FUSION_ALPHA=0.0  # 0.0 = pure V3 re-rank (best in 50K benchmark)

# ============================================================
# Pre-flight checks
# ============================================================
echo "Pre-flight checks:"
echo "  Data dir: ${DATA_DIR}"

if [ ! -f "${DATA_DIR}/test.csv" ]; then
    echo "  ERROR: ${DATA_DIR}/test.csv not found"
    exit 1
fi
echo "  test.csv: $(wc -l < ${DATA_DIR}/test.csv) lines"

if [ ! -f "${DATA_DIR}/reaction_db_cache.pkl" ]; then
    echo "  WARNING: reaction_db_cache.pkl not found, will build from CSV (slow)"
fi

if [ ! -f "Data/precomputed/knn_full_k200.pkl" ]; then
    echo "  WARNING: kNN cache not found, CF-NCN features will be empty"
fi
echo ""

# ============================================================
# Method dispatch
# ============================================================
case ${SLURM_ARRAY_TASK_ID} in
  1)
    METHOD="fingerprint"
    echo "Method: Fingerprint (Tanimoto baseline)"
    echo "  No model checkpoint needed"
    echo "============================================================"

    python scripts/eval_reactant_retrieval.py \
        --method fingerprint \
        --data-dir ${DATA_DIR} \
        --max-queries ${MAX_QUERIES} \
        --output docs/full_retrieval_fingerprint.md
    ;;

  2)
    METHOD="v3"
    echo "Method: V3 CF-NCN Link Predictor"
    echo "  Checkpoint: ${V3_CKPT}"

    if [ ! -f "${V3_CKPT}" ]; then
        echo "  ERROR: V3 checkpoint not found at ${V3_CKPT}"
        echo "  V3 FULL training may not be complete yet."
        echo "  Available checkpoints:"
        ls model_reactions/checkpoints/full_*/hypergraph_link_v3_best.pt 2>/dev/null
        exit 1
    fi
    echo "============================================================"

    python scripts/eval_reactant_retrieval.py \
        --method v3 \
        --data-dir ${DATA_DIR} \
        --checkpoint ${V3_CKPT} \
        --max-queries ${MAX_QUERIES} \
        --output docs/full_retrieval_v3.md
    ;;

  3)
    METHOD="aio"
    echo "Method: AIO DirectedHypergraphNet Stage 1"
    echo "  Checkpoint: ${AIO_CKPT}"

    if [ ! -f "${AIO_CKPT}" ]; then
        echo "  ERROR: AIO checkpoint not found at ${AIO_CKPT}"
        echo "  AIO FULL training may not be complete yet."
        exit 1
    fi
    echo "============================================================"

    python scripts/eval_reactant_retrieval.py \
        --method aio \
        --data-dir ${DATA_DIR} \
        --checkpoint ${AIO_CKPT} \
        --max-queries ${MAX_QUERIES} \
        --output docs/full_retrieval_aio.md
    ;;

  4)
    METHOD="aio_v3_rerank"
    echo "Method: AIO + V3 Re-ranking (Hybrid)"
    echo "  AIO checkpoint: ${AIO_CKPT}"
    echo "  V3 checkpoint:  ${V3_CKPT}"
    echo "  Re-rank top-N:  ${RERANK_TOPN}"
    echo "  Fusion alpha:   ${FUSION_ALPHA}"

    if [ ! -f "${AIO_CKPT}" ]; then
        echo "  ERROR: AIO checkpoint not found at ${AIO_CKPT}"
        exit 1
    fi
    if [ ! -f "${V3_CKPT}" ]; then
        echo "  ERROR: V3 checkpoint not found at ${V3_CKPT}"
        exit 1
    fi
    echo "============================================================"

    python scripts/eval_reactant_retrieval.py \
        --method aio_v3_rerank \
        --data-dir ${DATA_DIR} \
        --aio-checkpoint ${AIO_CKPT} \
        --v3-checkpoint ${V3_CKPT} \
        --max-queries ${MAX_QUERIES} \
        --rerank-topn ${RERANK_TOPN} \
        --fusion-alpha ${FUSION_ALPHA} \
        --output docs/full_retrieval_hybrid.md
    ;;

  *)
    echo "Unknown array task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Method ${METHOD} completed with exit code ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="
