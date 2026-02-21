#!/bin/bash
# Benchmark: 64 mols × 10 episodes with CPU/GPU monitoring
# Usage: bash scripts/bench_64mol.sh

set -e
source ~/.bashrc_maple 2>/dev/null

LOG_DIR="Experiments/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_LOG="$LOG_DIR/gpu_bench64_${TIMESTAMP}.csv"
CPU_LOG="$LOG_DIR/cpu_bench64_${TIMESTAMP}.csv"
TRAIN_LOG="$LOG_DIR/bench64_train_${TIMESTAMP}.log"

echo "=== Benchmark: 64 mols × 10 episodes ==="
echo "GPU log: $GPU_LOG"
echo "CPU log: $CPU_LOG"
echo "Train log: $TRAIN_LOG"
echo ""

# --- Start GPU monitor (1s interval) ---
echo "timestamp,gpu_util_pct,mem_used_mb,mem_total_mb,gpu_temp_c" > "$GPU_LOG"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            while IFS=, read -r util mem_used mem_total temp; do
                echo "$(date +%H:%M:%S),$util,$mem_used,$mem_total,$temp"
            done >> "$GPU_LOG"
        sleep 1
    done
) &
GPU_PID=$!

# --- Start CPU monitor (1s interval) ---
echo "timestamp,cpu_pct,mem_pct,n_threads,load_1m" > "$CPU_LOG"
(
    while true; do
        py_cpu=$(ps -eo pcpu,comm 2>/dev/null | grep -E 'python|conda' | awk '{s+=$1} END {printf "%.1f", s}')
        mem_pct=$(free | awk '/Mem:/ {printf "%.1f", $3/$2*100}')
        n_threads=$(ps -eLf 2>/dev/null | grep -c python || echo 0)
        load=$(awk '{print $1}' /proc/loadavg)
        echo "$(date +%H:%M:%S),${py_cpu:-0},${mem_pct},${n_threads},${load}" >> "$CPU_LOG"
        sleep 1
    done
) &
CPU_PID=$!

echo "Monitors started (GPU=$GPU_PID, CPU=$CPU_PID)"
echo ""

# --- Run benchmark ---
echo "Starting Route-DQN benchmark..."
echo "  method=route, decompose=paroutes, episodes=10, max_steps=5, num_mol=64"
echo "  cascade_workers=8, enable_variable_length=true"
echo ""

PYTHONUNBUFFERED=1 conda run -n rl4 --live-stream python main.py \
    method=route \
    episodes=10 \
    max_steps=5 \
    num_molecules=64 \
    log_freq=1 \
    save_freq=5 \
    eps_start=0.5 \
    eps_decay=0.9 \
    exp_name=bench64 \
    trial=0 \
    method.cascade_workers=8 \
    method.enable_variable_length=true \
    method.min_route_len=2 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=$?

# --- Stop monitors ---
kill $GPU_PID $CPU_PID 2>/dev/null
wait $GPU_PID $CPU_PID 2>/dev/null

echo ""
echo "=========================================="
echo "=== Monitoring Summary ==="
echo "=========================================="
echo ""

# GPU summary - split by phase (init vs training)
echo "--- GPU Usage (Overall) ---"
if [ -f "$GPU_LOG" ]; then
    tail -n +2 "$GPU_LOG" | awk -F, '
    {
        n++;
        gpu_sum += $2; mem_sum += $3;
        if ($2+0 > gpu_max) gpu_max = $2+0;
        if ($3+0 > mem_max) mem_max = $3+0;
        # Track non-zero GPU util samples
        if ($2+0 > 5) gpu_active++;
        if ($3+0 > 100) mem_active++;
    }
    END {
        if (n > 0) {
            printf "  Samples: %d (~%ds)\n", n, n
            printf "  GPU util: avg=%.1f%%, max=%.0f%%\n", gpu_sum/n, gpu_max
            printf "  GPU mem:  avg=%.0f MB, max=%.0f MB (of %s MB)\n", mem_sum/n, mem_max, $4
            printf "  GPU active (>5%%): %d samples (%.1f%% of time)\n", gpu_active, gpu_active*100/n
            printf "  GPU mem active (>100MB): %d samples (%.1f%% of time)\n", mem_active, mem_active*100/n
        }
    }'
fi

echo ""
echo "--- CPU Usage (Overall) ---"
if [ -f "$CPU_LOG" ]; then
    tail -n +2 "$CPU_LOG" | awk -F, '
    {
        n++;
        cpu_sum += $2; thread_sum += $4; load_sum += $5;
        if ($2+0 > cpu_max) cpu_max = $2+0;
        if ($4+0 > thread_max) thread_max = $4+0;
        if ($5+0 > load_max) load_max = $5+0;
    }
    END {
        if (n > 0) {
            printf "  Samples: %d (~%ds)\n", n, n
            printf "  Python CPU: avg=%.1f%%, max=%.0f%%\n", cpu_sum/n, cpu_max
            printf "  Python threads: avg=%.0f, max=%.0f\n", thread_sum/n, thread_max
            printf "  System load (1m): avg=%.1f, max=%.1f\n", load_sum/n, load_max
        }
    }'
fi

echo ""
echo "--- Per-Episode Timing (from train log) ---"
grep -E "^Ep |Episode |TIMING|Init took|Training phase|actions:|cascade:" "$TRAIN_LOG" 2>/dev/null | tail -30

echo ""
echo "--- Logs ---"
echo "  GPU:   $GPU_LOG"
echo "  CPU:   $CPU_LOG"
echo "  Train: $TRAIN_LOG"

exit $TRAIN_EXIT
