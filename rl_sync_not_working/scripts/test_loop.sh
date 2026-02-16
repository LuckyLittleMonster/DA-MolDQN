#!/bin/bash
# Simple test: loop 100 times, sleep 1s, echo counter
INSTANCE=${1:-0}
for i in $(seq 1 100); do
    echo "[Instance $INSTANCE] i=$i"
    sleep 1
done
echo "[Instance $INSTANCE] Done!"
