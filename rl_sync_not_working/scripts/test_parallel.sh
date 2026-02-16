#!/bin/bash
# Run two test_loop instances in parallel
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Launching 2 instances in parallel on $(hostname)..."
bash "$DIR/test_loop.sh" 1 > /tmp/test_loop_1.log 2>&1 &
PID1=$!
bash "$DIR/test_loop.sh" 2 > /tmp/test_loop_2.log 2>&1 &
PID2=$!
echo "PID1=$PID1, PID2=$PID2"

# Show progress for a few seconds
for t in 1 2 3 4 5; do
    sleep 1
    L1=$(tail -1 /tmp/test_loop_1.log 2>/dev/null)
    L2=$(tail -1 /tmp/test_loop_2.log 2>/dev/null)
    echo "  t=${t}s: inst1='$L1' | inst2='$L2'"
done

echo ""
echo "Both running in background. Monitor with:"
echo "  tail -f /tmp/test_loop_1.log"
echo "  tail -f /tmp/test_loop_2.log"
echo "Stop with: kill $PID1 $PID2"
