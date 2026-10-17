"""
run_table2.py — 10-task Split CIFAR-100 (Table II + novelty trajectory).

Canonical protocol from kfn_core.py (no test leakage, SVD estimator,
tau=0.2, m=20, ~2.8M specialist, 5 seeds).

Output: results_table2/seed_<s>.json, results_table2/aggregate.json
Run:    python run_table2.py --seeds 0 1 2 3 4
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kfn_core import (
    KFN, Specialist, expand_kfn, compute_novelty_svd, ReplayBuffer,
    parameter_breakdown, evaluate, make_test_loader, split_task,
    load_dataset, get_transforms, set_deterministic, get_device, e,
)
import kfn_core as C

CPT = 10          # classes per task
N_TASKS = 10


def train_seed(seed: int, scale: float, out_dir: str, device) -> dict:
    C.set_deterministic(seed)
    train_ds = C.load_dataset("cifar100", train=True)
    test_ds = C.load_dataset("cifar100", train=False)

    buffer = C.ReplayBuffer()
    acc = np.zeros((N_TASKS, N_TASKS))
    records = []
    model = None
    teacher = None

    for t in range(1, N_TASKS + 1):
        tr_set, va_set, ref_set, class_map = C.split_task(
            train_ds, t, CPT, seed, "cifar100")
        train_loader = DataLoader(tr_set, batch_size=C.BATCH_SIZE,
                                  shuffle=True, num_workers=2)
        ref_loader = DataLoader(ref_set, batch_size=C.BATCH_SIZE,
                                shuffle=False, num_workers=2)
        classes = list(range(CPT * (t - 1), CPT * t))

        if t == 1:
            model = KFN(CPT, 1, old_dim=64, cpt=CPT).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=C.e(scale, C.EPOCHS_BASE))
            for _ in range(C.e(scale, C.EPOCHS_BASE)):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    lg, _ = model(x)
                    F.cross_entropy(lg, y).backward()
                    opt.step()
                sched.step()
            nov = {"novelty": 0.0, "realised_rank": 0, "n_ref": len(ref_set)}
            nd = 0
            timing = {}
        else:
            spec = Specialist(64).to(device)
            old_bank = C.build_old_bank(teacher, buffer, device) if t > 1 else None
            rec = C.train_kfn_task(model, teacher, spec, train_loader,
                                   buffer.loader(), ref_loader, device, CPT,
                                   classes, scale, variant="FULL",
                                   old_bank=old_bank)
            nov, nd, timing = rec["novelty"], rec["expansion_dim"], rec["timing_s"]
            del spec

        buffer.add(train_ds, class_map, seed, "cifar100")

        # single test touch (genuine CIL: all classes exposed so far)
        for tt in range(1, t + 1):
            tl = C.make_test_loader(test_ds, tt, CPT, "cifar100")
            acc[t - 1, tt - 1] = C.evaluate(model, tl, device, "cil", CPT, tt,
                                            n_exposed=t * CPT)

        pbd = C.parameter_breakdown(model)
        records.append({
            "task_id": t, "classes": classes,
            "novelty": nov, "expansion_dim": nd,
            "alpha": None if t == 1 else max(0.1, 1.0 - nov["novelty"]),
            "params": pbd, "replay_bytes": buffer.bytes(),
            "acc_row": acc[t - 1, :t].tolist(),
            "avg_acc": float(acc[t - 1, :t].mean()),
            "timing_s": timing,
        })
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Task {t}: nu={nov['novelty']:.4f} rank={nov['realised_rank']} "
              f"d_new={nd} params={pbd['P_total']:,} "
              f"avg={records[-1]['avg_acc']:.2f} "
              f"buffer={records[-1]['replay_bytes']['mib']} MiB")

    m = C.metrics_from_matrix(acc, N_TASKS)
    out = {"seed": seed, "accuracy_matrix": acc.tolist(), **m,
           "tasks": records, "final_params": records[-1]["params"],
           "final_replay_bytes": records[-1]["replay_bytes"]}
    C.save_json(out, os.path.join(out_dir, f"seed_{seed}.json"))
    return out


def aggregate(out_dir: str, seeds):
    recs = [C.load_json(os.path.join(out_dir, f"seed_{s}.json")) for s in seeds]
    recs = [r for r in recs if r]
    if not recs:
        return
    keys = ["average_final_accuracy", "plasticity", "forgetting",
            "std_final_tasks", "min_final_task", "task_last_accuracy"]
    agg = {"n_seeds": len(recs), "seeds": [r["seed"] for r in recs]}
    for k in keys:
        agg[k] = C.mean_std([r[k] for r in recs])
    # novelty trajectory + realised ranks per task
    agg["novelty_by_task"] = {
        str(t + 1): C.mean_std([r["tasks"][t]["novelty"]["novelty"] for r in recs])
        for t in range(N_TASKS)}
    agg["realised_rank_by_task"] = {
        str(t + 1): [r["tasks"][t]["novelty"]["realised_rank"] for r in recs]
        for t in range(N_TASKS)}
    agg["expansion_dim_by_task"] = {
        str(t + 1): [r["tasks"][t]["expansion_dim"] for r in recs]
        for t in range(N_TASKS)}
    agg["params_by_task"] = {
        str(t + 1): C.mean_std([r["tasks"][t]["params"]["P_total"] for r in recs])
        for t in range(N_TASKS)}
    agg["final_params"] = recs[-1]["final_params"]
    agg["final_replay_bytes"] = recs[-1]["final_replay_bytes"]
    C.save_json(agg, os.path.join(out_dir, "aggregate.json"))
    print("\n=== AGGREGATE Table II ===")
    for k in keys:
        print(f"  {k}: {agg[k]['mean']:.2f} +/- {agg[k]['std']:.2f}")
    print(f"  final params: {agg['final_params']['P_total']:,}")


def train_seed_baseline(method, seed, scale, out_dir, device):
    """10-task CIFAR-100 shared-parameter baseline under matched protocol."""
    C.set_deterministic(seed)
    train_ds = C.load_dataset("cifar100", train=True)
    test_ds = C.load_dataset("cifar100", train=False)
    train_sets, class_maps = [], []
    for t in range(1, N_TASKS + 1):
        tr, _, _, cm = C.split_task(train_ds, t, CPT, seed, "cifar100")
        train_sets.append(tr)
        class_maps.append(cm)
    res = C.train_baseline(method, train_sets, class_maps, test_ds, device,
                           seed, CPT, N_TASKS, scale, "cifar100",
                           full_train_ds=train_ds)
    acc = np.array(res["accuracy_matrix"])
    m = C.metrics_from_matrix(acc, N_TASKS)
    rec = {"method": method, "seed": seed, "accuracy_matrix": acc.tolist(),
           **m, "ta_final": res["ta_final"],
           "replay_bytes": res["replay_bytes"]}
    fn = method.replace("+", "p")
    C.save_json(rec, os.path.join(out_dir, f"{fn}_seed_{seed}.json"))
    print(f"  {method} avg={m['average_final_accuracy']:.2f} "
          f"plast={m['plasticity']:.2f} F={m['forgetting']:.2f}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--out", default="./results_table2")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--methods", type=str, nargs="+", default=["KFN"])
    a = ap.parse_args()
    dev = C.get_device()
    os.makedirs(a.out, exist_ok=True)
    for method in a.methods:
        for s in a.seeds:
            if method != "KFN":
                print(f"\n=== Table2 {method} seed {s} on {dev} ===")
                train_seed_baseline(method, s, a.epochs_scale, a.out, dev)
                continue
            print(f"\n=== Table2 seed {s} on {dev} ===")
            train_seed(s, a.epochs_scale, a.out, dev)
    if not a.skip_aggregate:
        aggregate(a.out, a.seeds)
