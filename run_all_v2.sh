#!/bin/bash
# run_all_v2.sh — full rerun with the improved recipe (new phase-4 calibration,
# fusion-budget scaling, capacity-matched REPLAY_ONLY, grown-arch retrain baseline,
# compute-parity baselines).
set -u
cd /home/ubuntu/kfn/codes_to_run
PY=/home/ubuntu/kfnenv/bin/python
LOG=/home/ubuntu/kfn/logs
mkdir -p $LOG

echo "[$(date)] START run_all_v2" >> $LOG/run_all.log

# two-task tables (parallelizable; run sequentially here for determinism)
$PY run_tables1_3_7.py --table 1 --seeds 0 1 2 3 4 > $LOG/table1.log 2>&1
echo "[$(date)] table1 exit=$?" >> $LOG/run_all.log

$PY run_tables1_3_7.py --table 3 --seeds 0 1 2 3 4 > $LOG/table3.log 2>&1
echo "[$(date)] table3 exit=$?" >> $LOG/run_all.log

$PY run_tables1_3_7.py --table 7 --seeds 0 1 2 3 4 > $LOG/table7.log 2>&1
echo "[$(date)] table7 exit=$?" >> $LOG/run_all.log

# 5-task ablation (narrative-correct setting)
$PY run_table4_long.py --seeds 0 1 2 3 4 > $LOG/table4.log 2>&1
echo "[$(date)] table4-long exit=$?" >> $LOG/run_all.log

# efficiency (compute/memory, grown-arch baseline)
$PY run_efficiency.py --seeds 0 1 2 3 4 --skip-10task > $LOG/efficiency_compute.log 2>&1
echo "[$(date)] efficiency-compute exit=$?" >> $LOG/run_all.log

# 10-task CIFAR-100 (core table)
$PY run_table2.py --seeds 0 1 2 3 4 > $LOG/table2.log 2>&1
echo "[$(date)] table2 exit=$?" >> $LOG/run_all.log

# 10-task parameter scaling (Figure 7)
$PY run_efficiency.py --seeds 0 1 2 3 4 > $LOG/efficiency_full.log 2>&1
echo "[$(date)] efficiency-full exit=$?" >> $LOG/run_all.log

echo "[$(date)] ALL DONE v2" >> $LOG/run_all.log
