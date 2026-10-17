"""
run_efficiency.py — Computational/memory table (Table VIII) + 10-task
parameter scaling curve (Figure 7 data).

[IC3-4.8] Per-component parameter breakdown at every task; verifies the
          linear-growth lower bound P_10 >= P_1 + 9*|phi|.
[IC3-4.15] Replay bytes at every task.
[IC3-4.13] Per-phase wall-clock + peak GPU memory (KFN vs full retrain).

Output: results_efficiency/compute_table.json (2-task CIFAR-10)
        results_efficiency/param_scaling.json (10-task CIFAR-100)
Run:
    python run_efficiency.py --seeds 0 1 2 3 4              # both
    python run_efficiency.py --seeds 0 1 2 3 4 --skip-10task
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import kfn_core as C
from kfn_core import KFN, Specialist, ReplayBuffer, evaluate, make_test_loader


def sync_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def measure_compute(train_ds, test_ds, device, seed, scale):
    """Two-task CIFAR-10: full joint retrain vs KFN 4-phase pipeline."""
    cpt = 5
    C.set_deterministic(seed)

    out = {"seed": seed, "device": str(device)}

    # ---------- Full retrain baseline ----------
    tr1, _, _, _ = C.split_task(train_ds, 1, cpt, seed, "cifar10")
    tr2, _, _, _ = C.split_task(train_ds, 2, cpt, seed, "cifar10")
    joint = ConcatDataset([tr1, tr2])
    loader = DataLoader(joint, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=2)

    reset_peak(device)
    t0 = time.time()
    # Grown-architecture full retrain: KFN's final capacity (2 specialists,
    # widened shared pathway) trained from scratch on the union of tasks with
    # KFN's TOTAL epoch budget. The honest naive competitor: same capacity,
    # full data access, zero incremental machinery.
    full = KFN(cpt * 2, 2, old_dim=64 + C.D_HIGH, new_dim=0, cpt=cpt * 2).to(device)
    opt = torch.optim.AdamW(full.parameters(), lr=1e-3, weight_decay=1e-4)
    ep_total = C.e(scale, C.EPOCHS_BASE + C.EPOCHS_SPECIALIST +
                   C.EPOCHS_FUSION + C.EPOCHS_BIAS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_total)
    for _ in range(ep_total):
        full.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lg, _ = full(x)
            F.cross_entropy(lg, y).backward()
            opt.step()
        sched.step()
    out["full_retrain"] = {"time_s": time.time() - t0, "peak_mem_mb": sync_peak(device),
                           "params": sum(p.numel() for p in full.parameters())}

    # ---------- KFN pipeline ----------
    buffer = C.ReplayBuffer()
    t_start = time.time()
    phase_times = {"phase1": 0.0, "phase2": 0.0, "phase3": 0.0, "phase4": 0.0}
    mem = {}

    # Phase 1 (base)
    reset_peak(device)
    t0 = time.time()
    tr1_set, _, ref1, cm1 = C.split_task(train_ds, 1, cpt, seed, "cifar10")
    l1 = DataLoader(tr1_set, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=2)
    model = KFN(cpt, 1, old_dim=64, cpt=cpt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.e(scale, C.EPOCHS_BASE))
    for _ in range(C.e(scale, C.EPOCHS_BASE)):
        model.train()
        for x, y in l1:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lg, _ = model(x)
            F.cross_entropy(lg, y).backward()
            opt.step()
        sched.step()
    phase_times["phase1"] = time.time() - t0
    mem["phase1"] = sync_peak(device)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    buffer.add(train_ds, cm1, seed, "cifar10")

    # Task 2 phases
    tr2_set, _, ref2, _ = C.split_task(train_ds, 2, cpt, seed, "cifar10")
    l2 = DataLoader(tr2_set, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=2)
    rl2 = DataLoader(ref2, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=2)
    classes2 = list(range(cpt, 2 * cpt))

    spec = Specialist(64).to(device)
    rec = C.train_kfn_task(model, teacher, spec, l2, buffer.loader(), rl2,
                           device, cpt, classes2, scale, variant="FULL")
    phase_times["phase2"] = rec["timing_s"]["phase2"]
    phase_times["phase3"] = rec["timing_s"]["phase3"]
    phase_times["phase4"] = rec["timing_s"]["phase4"]
    del spec

    kfn_total = time.time() - t_start
    peak_kfn = max(mem["phase1"], sync_peak(device))
    peak_full = out["full_retrain"]["peak_mem_mb"]

    out["kfn"] = {
        "time_s": kfn_total,
        "peak_mem_mb": peak_kfn,
        "speedup": out["full_retrain"]["time_s"] / kfn_total if kfn_total > 0 else 0.0,
        "mem_reduction_pct": 100.0 * (1.0 - peak_kfn / peak_full) if peak_full > 0 else 0.0,
        "phase_times_s": phase_times,
        "phase3_pct": 100.0 * phase_times["phase3"] / kfn_total if kfn_total else 0,
        "phase2_pct": 100.0 * phase_times["phase2"] / kfn_total if kfn_total else 0,
        "phase4_pct": 100.0 * phase_times["phase4"] / kfn_total if kfn_total else 0,
        "novelty_task2": rec["novelty"],
        "expansion_dim_task2": rec["expansion_dim"],
    }
    return out


def measure_param_scaling(train_ds, test_ds, device, seed, scale):
    """10-task CIFAR-100: records parameter growth at every task (Figure 7)."""
    cpt, n_tasks = 10, 10
    C.set_deterministic(seed)
    acc = np.zeros((n_tasks, n_tasks))
    buffer = C.ReplayBuffer()
    model = None
    teacher = None
    param_history, task_records = [], []

    for t in range(1, n_tasks + 1):
        tr_set, _, ref_set, cm = C.split_task(train_ds, t, cpt, seed, "cifar100")
        classes = list(range(cpt * (t - 1), cpt * t))
        loader = DataLoader(tr_set, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=2)
        ref_loader = DataLoader(ref_set, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=2)

        if t == 1:
            model = KFN(cpt, 1, old_dim=64, cpt=cpt).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.e(scale, C.EPOCHS_BASE))
            for _ in range(C.e(scale, C.EPOCHS_BASE)):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    lg, _ = model(x)
                    F.cross_entropy(lg, y).backward()
                    opt.step()
                sched.step()
            nov = {"novelty": 0.0, "realised_rank": 0, "n_ref": len(ref_set)}
            nd = 0
        else:
            spec = Specialist(64).to(device)
            old_bank = C.build_old_bank(teacher, buffer, device) if t > 1 else None
            rec = C.train_kfn_task(model, teacher, spec, loader,
                                   buffer.loader(), ref_loader, device,
                                   cpt, classes, scale, variant="FULL",
                                   old_bank=old_bank)
            nov, nd = rec["novelty"], rec["expansion_dim"]
            del spec

        buffer.add(train_ds, cm, seed, "cifar100")
        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, cpt, "cifar100")
            acc[t - 1, tt - 1] = evaluate(model, tl, device, "cil", cpt, tt,
                                          n_exposed=t * cpt)

        pbd = C.parameter_breakdown(model)
        param_history.append({"task": t, **pbd, "replay": buffer.bytes()})
        task_records.append({"task": t, "novelty": nov, "expansion_dim": nd,
                             "avg_acc": float(acc[t - 1, :t].mean())})
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Task {t}: params={pbd['P_total']:,} d_new={nd} "
              f"nu={nov['novelty']:.4f}")

    P1 = param_history[0]["P_total"]
    P10 = param_history[-1]["P_total"]
    phi = param_history[0]["P_specialist_each"]
    return {
        "seed": seed, "param_history": param_history,
        "task_records": task_records, "accuracy_matrix": acc.tolist(),
        "final_avg_acc": float(acc[-1].mean()),
        "final_params": P10,
        "linear_check": {"P_1": P1, "P_10": P10, "specialist_each": phi,
                         "expected_linear_P_10": P1 + 9 * phi,
                         "linear_lower_bound_satisfied": P10 >= P1 + 9 * phi},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--out", default="./results_efficiency")
    ap.add_argument("--skip-10task", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = C.get_device()

    # ---- Part A: compute table (2-task CIFAR-10) ----
    compute_path = os.path.join(a.out, "compute_table.json")
    existing = C.load_json(compute_path) or {"raw": []}
    have = {r["seed"] for r in existing["raw"]}
    train10 = C.load_dataset("cifar10", train=True)
    test10 = C.load_dataset("cifar10", train=False)
    for seed in a.seeds:
        if seed in have:
            print(f"compute: seed {seed} exists, skipping")
            continue
        print(f"\n=== compute table seed {seed} ===")
        existing["raw"].append(measure_compute(train10, test10, device, seed, a.epochs_scale))
        C.save_json(existing, compute_path)

    recs = existing["raw"]
    agg = {
        "n_seeds": len(recs),
        "seeds": [r["seed"] for r in recs],
        "full_retrain_time_s": C.mean_std([r["full_retrain"]["time_s"] for r in recs]),
        "full_retrain_mem_mb": C.mean_std([r["full_retrain"]["peak_mem_mb"] for r in recs]),
        "kfn_time_s": C.mean_std([r["kfn"]["time_s"] for r in recs]),
        "kfn_mem_mb": C.mean_std([r["kfn"]["peak_mem_mb"] for r in recs]),
        "speedup": C.mean_std([r["kfn"]["speedup"] for r in recs]),
        "mem_reduction_pct": C.mean_std([r["kfn"]["mem_reduction_pct"] for r in recs]),
        "phase2_pct": C.mean_std([r["kfn"]["phase2_pct"] for r in recs]),
        "phase3_pct": C.mean_std([r["kfn"]["phase3_pct"] for r in recs]),
        "phase4_pct": C.mean_std([r["kfn"]["phase4_pct"] for r in recs]),
    }
    C.save_json({"aggregate": agg, "raw": recs}, compute_path)
    print("\n=== AGGREGATE Table VIII ===")
    print(f"  Full retrain: {agg['full_retrain_time_s']['mean']:.2f}s, "
          f"{agg['full_retrain_mem_mb']['mean']:.1f} MB")
    print(f"  KFN:          {agg['kfn_time_s']['mean']:.2f}s, "
          f"{agg['kfn_mem_mb']['mean']:.1f} MB | "
          f"speedup {agg['speedup']['mean']:.2f}x | "
          f"mem -{agg['mem_reduction_pct']['mean']:.1f}%")
    print(f"  phases: P2 {agg['phase2_pct']['mean']:.1f}% / "
          f"P3 {agg['phase3_pct']['mean']:.1f}% / "
          f"P4 {agg['phase4_pct']['mean']:.1f}%")

    if a.skip_10task:
        return

    # ---- Part B: 10-task parameter scaling (Figure 7) ----
    scale_path = os.path.join(a.out, "param_scaling.json")
    existing = C.load_json(scale_path) or {"raw": []}
    have = {r["seed"] for r in existing["raw"]}
    train100 = C.load_dataset("cifar100", train=True)
    test100 = C.load_dataset("cifar100", train=False)
    for seed in a.seeds:
        if seed in have:
            print(f"scaling: seed {seed} exists, skipping")
            continue
        print(f"\n=== param scaling 10-task seed {seed} ===")
        existing["raw"].append(measure_param_scaling(train100, test100, device, seed, a.epochs_scale))
        C.save_json(existing, scale_path)

    recs = existing["raw"]
    curve = []
    for t in range(10):
        vals = [r["param_history"][t]["P_total"] for r in recs]
        curve.append({"task": t + 1, **C.mean_std(vals)})
    agg2 = {
        "n_seeds": len(recs), "seeds": [r["seed"] for r in recs],
        "param_curve": curve,
        "final_params": C.mean_std([r["final_params"] for r in recs]),
        "specialist_each": recs[0]["param_history"][0]["P_specialist_each"],
        "final_avg_acc": C.mean_std([r["final_avg_acc"] for r in recs]),
        "novelty_by_task": {str(t + 1): C.mean_std(
            [r["task_records"][t]["novelty"]["novelty"] for r in recs])
            for t in range(10)},
        "realised_rank_by_task": {str(t + 1): [r["task_records"][t]["novelty"]["realised_rank"]
                                               for r in recs] for t in range(10)},
        "linear_check_all_seeds": [r["linear_check"] for r in recs],
    }
    C.save_json({"aggregate": agg2, "raw": recs}, scale_path)
    print("\n=== AGGREGATE Figure 7 ===")
    print(f"  P_1={curve[0]['mean']:,.0f} -> P_10={curve[-1]['mean']:,.0f} "
          f"(specialist = {agg2['specialist_each']:,} each)")
    for t, cpoint in enumerate(curve):
        print(f"  task {t+1}: {cpoint['mean']:,.0f} +/- {cpoint['std']:,.0f}")


if __name__ == "__main__":
    main()
