#!/bin/bash
# run_all.sh — full canonical experiment sequence on the cloud GPU.
# Every step logs to its own file; each is resumable (runners skip existing results).
set -u
cd /home/ubuntu/kfn/codes_to_run
PY=/home/ubuntu/kfnenv/bin/python
LOG=/home/ubuntu/kfn/logs
mkdir -p $LOG

echo "[$(date)] START run_all" >> $LOG/run_all.log

# ---------- 0) dataset download (CIFAR-10/100) ----------
$PY -c "import torchvision; torchvision.datasets.CIFAR10('./data', train=True, download=True); torchvision.datasets.CIFAR10('./data', train=False, download=True); torchvision.datasets.CIFAR100('./data', train=True, download=True); torchvision.datasets.CIFAR100('./data', train=False, download=True); print('CIFARS-DONE')" \
    >> $LOG/data_dl.log 2>&1
echo "[$(date)] datasets ready" >> $LOG/run_all.log

# ---------- 1) smoke test (fast validation) ----------
$PY smoke_test.py > $LOG/smoke.log 2>&1
SMOKE=$?
echo "[$(date)] smoke exit=$SMOKE" >> $LOG/run_all.log
if [ $SMOKE -ne 0 ]; then
    echo "[$(date)] SMOKE FAILED - aborting full runs" >> $LOG/run_all.log
    exit 1
fi

# ---------- 2) two-task tables (independent) ----------
$PY run_tables1_3_7.py --table 1 --seeds 0 1 2 3 4 > $LOG/table1.log 2>&1
echo "[$(date)] table1 exit=$?" >> $LOG/run_all.log

$PY run_tables1_3_7.py --table 3 --seeds 0 1 2 3 4 > $LOG/table3.log 2>&1
echo "[$(date)] table3 exit=$?" >> $LOG/run_all.log

$PY run_tables1_3_7.py --table 7 --seeds 0 1 2 3 4 > $LOG/table7.log 2>&1
echo "[$(date)] table7 exit=$?" >> $LOG/run_all.log

# ---------- 3) ablation ----------
$PY run_table4.py --seeds 0 1 2 3 4 > $LOG/table4.log 2>&1
echo "[$(date)] table4 exit=$?" >> $LOG/run_all.log

# ---------- 4) efficiency (compute/memory table only first) ----------
$PY run_efficiency.py --seeds 0 1 2 3 4 --skip-10task > $LOG/efficiency_compute.log 2>&1
echo "[$(date)] efficiency-compute exit=$?" >> $LOG/run_all.log

# ---------- 5) 10-task CIFAR-100 (Table II) ----------
$PY run_table2.py --seeds 0 1 2 3 4 > $LOG/table2.log 2>&1
echo "[$(date)] table2 exit=$?" >> $LOG/run_all.log

# ---------- 6) 10-task parameter scaling (Figure 7) ----------
$PY run_efficiency.py --seeds 0 1 2 3 4 > $LOG/efficiency_full.log 2>&1
echo "[$(date)] efficiency-full exit=$?" >> $LOG/run_all.log

echo "[$(date)] ALL DONE" >> $LOG/run_all.log
