#!/bin/bash
# Smoke test with CPU/GPU monitoring
# Usage: bash scripts/smoke_monitor.sh

set -e
source ~/.bashrc_maple 2>/dev/null

LOG_DIR="Experiments/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_LOG="$LOG_DIR/gpu_monitor_${TIMESTAMP}.csv"
CPU_LOG="$LOG_DIR/cpu_monitor_${TIMESTAMP}.csv"
TRAIN_LOG="$LOG_DIR/smoke_train_${TIMESTAMP}.log"

echo "=== Smoke Test with Monitoring ==="
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
        # Get CPU% of python processes, total system load
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

# --- Run smoke test ---
echo "Starting Route-DQN smoke test..."
echo "  method=route, decompose=paroutes, episodes=3, max_steps=3, num_mol=8"
echo "  cascade_workers=8, enable_variable_length=true"
echo ""

conda run -n rl4 python main.py \
    method=route \
    episodes=3 \
    max_steps=3 \
    num_molecules=8 \
    log_freq=1 \
    save_freq=999 \
    eps_start=0.5 \
    eps_decay=0.8 \
    exp_name=smoke_varlen \
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
echo "=== Monitoring Summary ==="
echo ""

# GPU summary
echo "--- GPU Usage ---"
if [ -f "$GPU_LOG" ]; then
    tail -n +2 "$GPU_LOG" | awk -F, '
    {
        n++;
        gpu_sum += $2; mem_sum += $3;
        if ($2+0 > gpu_max) gpu_max = $2+0;
        if ($3+0 > mem_max) mem_max = $3+0;
    }
    END {
        if (n > 0) {
            printf "  Samples: %d (%.0fs)\n", n, n
            printf "  GPU util: avg=%.1f%%, max=%.0f%%\n", gpu_sum/n, gpu_max
            printf "  GPU mem:  avg=%.0f MB, max=%.0f MB (of %s MB)\n", mem_sum/n, mem_max, $4
        }
    }'
fi

echo ""
echo "--- CPU Usage ---"
if [ -f "$CPU_LOG" ]; then
    tail -n +2 "$CPU_LOG" | awk -F, '
    {
        n++;
        cpu_sum += $2; thread_sum += $4; load_sum += $5;
        if ($2+0 > cpu_max) cpu_max = $2+0;
        if ($4+0 > thread_max) thread_max = $4+0;
    }
    END {
        if (n > 0) {
            printf "  Samples: %d (%.0fs)\n", n, n
            printf "  Python CPU: avg=%.1f%%, max=%.0f%%\n", cpu_sum/n, cpu_max
            printf "  Python threads: avg=%.0f, max=%.0f\n", thread_sum/n, thread_max
            printf "  System load (1m): avg=%.1f, max=N/A\n", load_sum/n
        }
    }'
fi

echo ""
echo "--- Logs ---"
echo "  GPU:   $GPU_LOG"
echo "  CPU:   $CPU_LOG"
echo "  Train: $TRAIN_LOG"

# Check if history pickle was saved with new fields
echo ""
echo "--- Pickle Validation ---"
conda run -n rl4 python -c "
import pickle, sys
try:
    with open('Experiments/smoke_varlen_0_history.pickle', 'rb') as f:
        h = pickle.load(f)
    print('  Keys:', sorted(h.keys()))
    if 'docks' in h:
        print('  docks: OK (len=%d)' % len(h['docks']))
    else:
        print('  docks: MISSING')
    if h.get('last_episodes'):
        ep = h['last_episodes'][-1]
        rt = ep['routes'][0] if ep['routes'] else {}
        has_traj = 'trajectory' in rt
        has_sa = 'sa' in rt
        has_reward = 'reward' in rt
        print(f'  last_episode trajectory: {\"OK\" if has_traj else \"MISSING\"}'
              f' (len={len(rt.get(\"trajectory\", []))})')
        print(f'  last_episode sa/reward: {\"OK\" if (has_sa and has_reward) else \"MISSING\"}')
        if has_traj and rt['trajectory']:
            rec = rt['trajectory'][0]
            print(f'  step_record keys: {sorted(rec.keys())}')
            print(f'  action_type sample: {rec[\"action_type\"]}')
    if h.get('best_products'):
        bp = h['best_products'][0]
        print(f'  best_products sa: {\"OK\" if \"sa\" in bp else \"MISSING\"}'
              f', dock: {\"OK\" if \"dock\" in bp else \"MISSING\"}'
              f', reward: {\"OK\" if \"reward\" in bp else \"MISSING\"}')
    if h.get('route_top5'):
        first_key = list(h['route_top5'].keys())[0]
        rt5 = h['route_top5'][first_key][0]
        print(f'  route_top5 sa: {\"OK\" if \"sa\" in rt5 else \"MISSING\"}'
              f', trajectory: {\"OK\" if \"trajectory\" in rt5 else \"MISSING\"}')
except FileNotFoundError:
    print('  History pickle not saved (save_freq=999, may not have triggered)')
except Exception as e:
    print(f'  Error: {e}')
" 2>&1 | grep -v '^$'

exit $TRAIN_EXIT
