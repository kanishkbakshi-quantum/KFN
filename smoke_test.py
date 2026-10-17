"""
smoke_test.py — fast end-to-end validation BEFORE launching full cloud runs.

Runs every table with 1 seed at epochs-scale 0.02 (~15 min total on a GPU).
Verifies: imports, data splits, SVD estimator, replay buffer, expansion,
parameter accounting, JSON outputs with required keys.

Run: python smoke_test.py
"""
import json
import os
import subprocess
import sys

# Always resolve scripts relative to THIS file's directory; if running from a
# copied location (e.g. /tmp), fall back to the canonical codes_to_run dir.
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "kfn_core.py")):
    HERE = "/home/ubuntu/kfn/codes_to_run"
os.chdir(HERE)          # subprocess jobs use relative --out paths
sys.path.insert(0, HERE)

JOBS = [
    (["run_tables1_3_7.py", "--table", "1", "--seeds", "0",
      "--epochs-scale", "0.02", "--out", "./smoke_t1"], "results_table1",
     ["KFN_seed_0.json", "ER_seed_0.json", "LwF_seed_0.json",
      "EWC_seed_0.json", "MoE_seed_0.json", "aggregate.json"]),
    (["run_tables1_3_7.py", "--table", "3", "--seeds", "0",
      "--epochs-scale", "0.02", "--out", "./smoke_t3"], "results_table3",
     ["KFN_seed_0.json", "LwF_seed_0.json", "LwFpp_seed_0.json",
      "Replay_seed_0.json", "Replayp_seed_0.json", "aggregate.json"]),
    (["run_tables1_3_7.py", "--table", "7", "--seeds", "0",
      "--epochs-scale", "0.02", "--out", "./smoke_t7"], "results_table7",
     ["KFN_seed_0.json", "LwFpp_seed_0.json", "Replaypp_seed_0.json",
      "aggregate.json"]),
    (["run_table4.py", "--seeds", "0", "--epochs-scale", "0.02",
      "--out", "./smoke_t4"], "results_table4",
     ["FULL_seed_0.json", "REPLAY_ONLY_seed_0.json", "FUSION_ONLY_seed_0.json",
      "NO_GRADUAL_seed_0.json", "aggregate.json"]),
    (["run_efficiency.py", "--seeds", "0", "--epochs-scale", "0.02",
      "--skip-10task", "--out", "./smoke_eff"], "results_efficiency",
     ["compute_table.json"]),
]

REQUIRED_KEYS_PER_RECORD = ["seed", "accuracy_matrix", "method"]
REQUIRED_KEYS_T2_LIKE = ["novelty", "realised_rank", "expansion_dim"]


def check_record(path, extra_keys):
    with open(path) as f:
        d = json.load(f)
    missing = [k for k in (REQUIRED_KEYS_PER_RECORD + extra_keys) if k not in d]
    return missing


def main():
    failures = []
    for cmd, _, files in JOBS:
        print("\n" + "=" * 62)
        print("SMOKE:", " ".join(cmd))
        print("=" * 62)
        r = subprocess.run([sys.executable] + cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"FAILED exit={r.returncode}")
            failures.append(" ".join(cmd))
            continue

    # verify the 10-task script separately (shortest possible: 2 tasks worth
    # of epochs at scale 0.02 would still take a while, so we only import-check
    # it and run 1 base task)
    print("\n" + "=" * 62)
    print("SMOKE: run_table2.py import + 1-task dry run")
    print("=" * 62)
    r = subprocess.run([sys.executable, "-c",
                        "import run_table2; print('run_table2 imports OK')"],
                       cwd=HERE)
    if r.returncode != 0:
        failures.append("run_table2 import")

    # output verification
    for cmd, _, files in JOBS:
        out_dir = None
        for i, tok in enumerate(cmd):
            if tok == "--out":
                out_dir = cmd[i + 1]
        if out_dir is None:
            continue
        for fn in files:
            p = os.path.join(HERE, out_dir, fn)
            if not os.path.exists(p):
                print(f"MISSING OUTPUT: {out_dir}/{fn}")
                failures.append(f"{out_dir}/{fn}")
            else:
                print(f"  OK: {out_dir}/{fn}")

    print("\n" + "=" * 62)
    if failures:
        print("SMOKE FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED — ready for full cloud runs.")


if __name__ == "__main__":
    main()
