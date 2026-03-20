#!/bin/bash
#
# Diversity QED Experiments for JCIM R1-07 Reviewer Response
# 3 SLURM jobs using part_maple_night QOS:
#   Job 1: 1,2,4,8,16,32,64 mols — parallel on 1 node (33 ranks total)
#   Job 2: 128 mols (32 ranks on 1 node)
#   Job 3: 256 mols (64 ranks on 2 nodes, 32/node)
#
# Usage:
#   bash scripts/run_diversity_qed.sh              # submit all 3 jobs
#   bash scripts/run_diversity_qed.sh --dry-run    # print without submitting
#

set -e
cd "$(dirname "$0")/.."

# ========== Shared Parameters ==========
EXPERIMENT="diversity_qed"
TRIAL_BASE=4000
INIT_MOL_PATH="./Data/zinc_10000.txt"
ITERATION=5000               # 250 episodes × 20 steps
MAX_STEPS=20
REWARD="qed"
REWARD_WEIGHT="1.0"
EPS_DECAY=0.968
MAX_BATCH_SIZE=512
SAVE_PATH_FREQ=50
SAVE_REWARD_FREQ=50
RECORD_TOP_PATH=10
RECORD_LAST_PATH=250
PARTITION="maple"
ACCOUNT="app"
QOS="part_maple_night"
INIT_METHOD_DIR="/shared/data1/Users/l1062811/git/RL4-working/tem"
PYTHON="/home/l1062811/data/envs/rl4/bin/python"

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

mkdir -p log "$INIT_METHOD_DIR"

# Helper: build the srun command for one config
make_srun_cmd() {
    local nmol=$1
    local nranks=$2
    local trial=$((TRIAL_BASE + nmol))
    echo "srun --exact -n ${nranks} -u ${PYTHON} main_hpc.py \
    --experiment ${EXPERIMENT} \
    --trial ${trial} \
    --note 'diversity_qed_${nmol}mols_${nranks}ranks' \
    --iteration ${ITERATION} \
    --init_mol_path ${INIT_MOL_PATH} \
    --num_init_mol ${nmol} \
    --gpu_list 0 \
    --starter slurm \
    --backend gloo \
    --max_steps_per_episode ${MAX_STEPS} \
    --reward ${REWARD} \
    --reward_weight ${REWARD_WEIGHT} \
    --eps_decay ${EPS_DECAY} \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --record_all_path \
    --record_top_path ${RECORD_TOP_PATH} \
    --record_last_path ${RECORD_LAST_PATH} \
    --save_path_freq ${SAVE_PATH_FREQ} \
    --save_reward_freq ${SAVE_REWARD_FREQ}"
}

echo "========================================"
echo "Diversity QED Experiments (3 GH200 jobs)"
echo "========================================"

# ============================================================
# Job 1: 1,2,4,8,16,32,64 mols — ALL parallel on 1 node
# Total ranks: 1+1+1+2+4+8+16 = 33. GH200 has 72 CPUs, fine.
# Each config is an independent srun step running in background.
# ============================================================
JOB1_SCRIPT=$(mktemp /tmp/dqed_job1_XXXXXX.sh)
cat > "$JOB1_SCRIPT" << 'HEADER'
#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --qos=part_maple_night
#SBATCH --mem=200G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=33
#SBATCH -o log/diversity_qed_job1_small.out
#SBATCH -e log/diversity_qed_job1_small.err
#SBATCH -J dqed_1to64
#SBATCH --time=01:00:00

echo "=========================================="
echo "Job 1: 1,2,4,8,16,32,64 mols (parallel)"
echo "Node: $(hostname), 33 tasks total"
echo "Start: $(date)"
echo "=========================================="

INIT_METHOD_DIR="/shared/data1/Users/l1062811/git/RL4-working/tem"
HEADER

# Append each config as a background srun step
for cfg in "1:1" "2:1" "4:1" "8:2" "16:4" "32:8" "64:16"; do
    nmol="${cfg%%:*}"
    nranks="${cfg##*:}"
    trial=$((TRIAL_BASE + nmol))

    cat >> "$JOB1_SCRIPT" << EOF

# --- ${nmol} mols, ${nranks} ranks (trial ${trial}) ---
rm -f "\${INIT_METHOD_DIR}/sharedfile_${EXPERIMENT}_${trial}"
$(make_srun_cmd $nmol $nranks) &
PID_${nmol}=\$!
echo "Launched ${nmol} mols (${nranks} ranks, trial ${trial}), PID=\${PID_${nmol}}"
EOF
done

cat >> "$JOB1_SCRIPT" << 'FOOTER'

echo ""
echo "All 7 configs launched, waiting..."
wait
echo "=========================================="
echo "Job 1 complete: $(date)"
echo "=========================================="
FOOTER

echo ""
echo "Job 1: 1,2,4,8,16,32,64 mols (parallel, 33 ranks on 1 node)"
echo "  Time limit: 60min, Memory: 200G"
if $DRY_RUN; then
    echo "  [DRY RUN] Would submit: $JOB1_SCRIPT"
    cat "$JOB1_SCRIPT"
    echo "---"
else
    echo "  Submitting..."
    sbatch "$JOB1_SCRIPT"
fi
rm -f "$JOB1_SCRIPT"

# ============================================================
# Job 2: 128 mols — 32 ranks on 1 node
# ============================================================
TRIAL_128=$((TRIAL_BASE + 128))
JOB2_SCRIPT=$(mktemp /tmp/dqed_job2_XXXXXX.sh)
cat > "$JOB2_SCRIPT" << EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --mem=200G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -o log/diversity_qed_job2_128.out
#SBATCH -e log/diversity_qed_job2_128.err
#SBATCH -J dqed_128
#SBATCH --time=01:00:00

echo "=========================================="
echo "Job 2: 128 mols, 32 ranks"
echo "Node: \$(hostname)"
echo "Start: \$(date)"
echo "=========================================="

rm -f "${INIT_METHOD_DIR}/sharedfile_${EXPERIMENT}_${TRIAL_128}"

$(make_srun_cmd 128 32)

echo "=========================================="
echo "Job 2 complete: \$(date)"
echo "=========================================="
EOF

echo ""
echo "Job 2: 128 mols (32 ranks, 1 node)"
echo "  Time limit: 60min, Memory: 200G"
if $DRY_RUN; then
    echo "  [DRY RUN] Would submit: $JOB2_SCRIPT"
else
    echo "  Submitting..."
    sbatch "$JOB2_SCRIPT"
fi
rm -f "$JOB2_SCRIPT"

# ============================================================
# Job 3: 256 mols — 64 ranks on 2 nodes (32 ranks/node)
# ============================================================
TRIAL_256=$((TRIAL_BASE + 256))
JOB3_SCRIPT=$(mktemp /tmp/dqed_job3_XXXXXX.sh)
cat > "$JOB3_SCRIPT" << EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --mem=200G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=1
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH -o log/diversity_qed_job3_256.out
#SBATCH -e log/diversity_qed_job3_256.err
#SBATCH -J dqed_256
#SBATCH --time=01:00:00

echo "=========================================="
echo "Job 3: 256 mols, 64 ranks (2 nodes)"
echo "Nodes: \$(hostname)"
echo "Start: \$(date)"
echo "=========================================="

rm -f "${INIT_METHOD_DIR}/sharedfile_${EXPERIMENT}_${TRIAL_256}"

$(make_srun_cmd 256 64)

echo "=========================================="
echo "Job 3 complete: \$(date)"
echo "=========================================="
EOF

echo ""
echo "Job 3: 256 mols (64 ranks, 2 nodes × 32 ranks/node)"
echo "  Time limit: 60min, Memory: 200G/node"
if $DRY_RUN; then
    echo "  [DRY RUN] Would submit: $JOB3_SCRIPT"
else
    echo "  Submitting..."
    sbatch "$JOB3_SCRIPT"
fi
rm -f "$JOB3_SCRIPT"

echo ""
echo "========================================"
echo "3 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: Experiments/${EXPERIMENT}_*"
echo "Analyze: python scripts/analyze_diversity_qed.py --include_reference"
echo "========================================"
