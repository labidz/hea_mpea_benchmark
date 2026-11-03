#!/bin/bash
# Revision-2 queue (sequential to avoid CPU contention on the 2-core workstation)
cd /home/claude/hea
export N_JOBS=2
for job in "HV shift" "HV ridge" "HV xgbuq" "YS shift" "YS ridge" "YS xgbuq"; do
  python3 bench_v2.py $job >> results/v2/log.txt 2>&1
done
echo DONE >> results/v2/log.txt
