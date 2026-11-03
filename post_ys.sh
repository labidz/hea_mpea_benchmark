#!/bin/bash
cd /home/claude/hea
while pgrep -f "bench.py YS" > /dev/null; do sleep 15; done
N_JOBS=2 python3 normalized_cp.py YS > results/ncp_YS_log.txt 2>&1
N_JOBS=2 python3 shap_analysis.py YS > results/shap_YS_log.txt 2>&1
N_JOBS=2 python3 ad_variants.py HV,YS > results/ad_variants_log.txt 2>&1
touch results/.post_ys_done
