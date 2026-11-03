#!/bin/bash
# after the revision-2 queue: per-record YS GPR/RF/MLP predictions for P1-P3 (previously only fold metrics from Kaggle)
cd /home/claude/hea
while ! grep -q DONE results/v2/log.txt 2>/dev/null; do sleep 60; done
export N_JOBS=2
python3 bench.py YS GPR,RF,MLP random,composition,system >> results/v2/log_ys_rest.txt 2>&1
echo DONE >> results/v2/log_ys_rest.txt
