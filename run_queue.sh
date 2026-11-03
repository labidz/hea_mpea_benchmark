#!/bin/bash
# usage: run_queue.sh STAGE
cd /home/claude/hea
case "$1" in
  A) while kill -0 5150 2>/dev/null; do sleep 20; done
     python3 external.py > results/external_log.txt 2>&1
     python3 bench.py YS Mean,kNN,Ridge,SVR,GPR,RF,XGB,MLP random,composition > results/YS_log_A.txt 2>&1
     touch results/.doneA ;;
  B) until [ -f results/.doneA ]; do sleep 30; done
     python3 bench.py YS Mean,kNN,Ridge,SVR,GPR,RF,XGB,MLP system,element,temporal > results/YS_log_B.txt 2>&1
     touch results/.doneB ;;
  C) until [ -f results/.doneB ]; do sleep 30; done
     python3 ablation.py HV,YS > results/ablation.txt 2>&1
     touch results/.doneC ;;
esac
