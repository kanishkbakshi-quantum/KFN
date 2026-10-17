"""
run_tables1_3_7.py — Two-task studies in ONE runner (shared infrastructure,
matched seeds/class ordering across methods within each table).

  --table 1 : 2-task CIFAR-10, 5 classes/task
              methods: KFN, ER, LwF, EWC, MoE          -> Table I
  --table 3 : 2-task CIFAR-100, 50 classes/task, TA+CIL
              methods: KFN, LwF, LwF++, Replay, Replay+ -> Table III
  --table 7 : 2-task EuroSAT, 5 classes/task, TA+CIL
              methods: KFN, LwF++, Replay++             -> Table VII

Output: results_table{N}/<method>_seed_<s>.json + aggregate.json
Run:
    python run_tables1_3_7.py --table 1 --seeds 0 1 2 3 4
    python run_tables1_3_7.py --table 3 --seeds 0 1 2 3 4
    python run_tables1_3_7.py --table 7 --seeds 0 1 2 3 4
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import kfn_core as C
from kfn_core import KFN, Specialist, ReplayBuffer, evaluate, make_test_loader

TABLES = {
    1: dict(dataset="cifar10",  cpt=5,  n_tasks=2,
            methods=["KFN", "ER", "LwF", "EWC", "MoE"],
            out="results_table1"),
    3: dict(dataset="cifar100", cpt=50, n_tasks=2,
            methods=["KFN", "LwF", "LwF++", "Replay", "Replay+"],
            out="results_table3"),
    7: dict(dataset="eurosat",  cpt=5,  n_tasks=2,
            methods=["KFN", "LwF++", "Replay++"],
            out="results_table7"),
}


def train_kfn_2task(cfg, train_sets, ref_sets, class_maps, train_ds_ref,
                    test_ds, device, seed, scale):
    cpt, n_tasks, dataset = cfg["cpt"], cfg["n_tasks"], cfg["dataset"]
    acc = np.zeros((n_tasks, n_tasks))
    ta_final = {}
    buffer = C.ReplayBuffer()
    model = None
    teacher = None
    extras = {}

    for t in range(1, n_tasks + 1):
        classes = list(range(cpt * (t - 1), cpt * t))
        loader = DataLoader(train_sets[t - 1], batch_size=C.BATCH_SIZE,
                            shuffle=True, num_workers=2)
        ref_loader = DataLoader(ref_sets[t - 1], batch_size=C.BATCH_SIZE,
                                shuffle=False, num_workers=2)

        if t == 1:
            model = KFN(cpt, 1, old_dim=64, cpt=cpt).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=C.e(scale, C.EPOCHS_BASE))
            for _ in range(C.e(scale, C.EPOCHS_BASE)):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    lg, _ = model(x)
                    F.cross_entropy(lg, y).backward()
                    opt.step()
                sched.step()
            nov = {"novelty": 0.0, "realised_rank": 0, "n_ref": len(ref_sets[0])}
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

        buffer.add(train_ds_ref, class_maps[t - 1], seed, dataset)

        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, cpt, dataset)
            # genuine Class-IL: expose ALL classes seen so far (t*cpt)
            acc[t - 1, tt - 1] = evaluate(model, tl, device, "cil", cpt, tt,
                                          n_exposed=t * cpt)
            if t == n_tasks:
                ta_final[tt] = evaluate(model, tl, device, "ta", cpt, tt)

        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if t == n_tasks:
            extras = {"novelty_task2": nov, "expansion_dim_task2": nd,
                      "replay_bytes": buffer.bytes(
                          hw=64 if dataset == "eurosat" else 32)}
    return acc, ta_final, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, required=True, choices=[1, 3, 7])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", type=str, nargs="+", default=None)
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-aggregate", action="store_true")
    a = ap.parse_args()

    cfg = TABLES[a.table]
    methods = a.methods or cfg["methods"]
    out_dir = a.out or cfg["out"]
    os.makedirs(out_dir, exist_ok=True)

    device = C.get_device()
    dataset, cpt, n_tasks = cfg["dataset"], cfg["cpt"], cfg["n_tasks"]
    train_ds = C.load_dataset(dataset, train=True)
    test_ds = C.load_dataset(dataset, train=False)

    all_records = []
    for method in methods:
        for seed in a.seeds:
            marker = os.path.join(out_dir, f"{method.replace('+', 'p')}_seed_{seed}.json")
            if C.load_json(marker) is not None:
                print(f"skip existing {marker}")
                continue
            print(f"\n=== T{a.table} {method} seed {seed} on {device} ===")
            C.set_deterministic(seed)

            train_sets, ref_sets, class_maps = [], [], []
            for t in range(1, n_tasks + 1):
                tr, _, rf, cm = C.split_task(train_ds, t, cpt, seed, dataset)
                train_sets.append(tr)
                ref_sets.append(rf)
                class_maps.append(cm)

            if method == "KFN":
                acc, ta_final, extras = train_kfn_2task(
                    cfg, train_sets, ref_sets, class_maps, train_ds,
                    test_ds, device, seed, a.epochs_scale)
            else:
                res = C.train_baseline(method, train_sets, class_maps, test_ds,
                                       device, seed, cpt, n_tasks,
                                       a.epochs_scale, dataset,
                                       full_train_ds=train_ds)
                acc = np.array(res["accuracy_matrix"])
                ta_final = res["ta_final"]
                extras = {"replay_bytes": res["replay_bytes"]}

            A_A_init_TA = ta_final.get(1, float(acc[0, 0]))
            A_A_CIL = float(acc[-1, 0])
            A_B_CIL = float(acc[-1, 1]) if n_tasks > 1 else 0.0
            retention = 100.0 * A_A_CIL / A_A_init_TA if A_A_init_TA > 0 else 0.0
            rec = {
                "method": method, "seed": seed,
                "A_A_init_TA": A_A_init_TA,
                "A_B_init_TA": ta_final.get(2, None) if n_tasks > 1 else None,
                "A_A_final_CIL": A_A_CIL,
                "A_B_final_CIL": A_B_CIL,
                "avg_CIL": float(np.mean([A_A_CIL, A_B_CIL])) if n_tasks > 1 else A_A_CIL,
                "cil_retention": retention,
                "forgetting_TA": (A_A_init_TA - float(acc[-1, 0])) if n_tasks == 2
                                  else (float(acc[0, 0]) - float(acc[-1, 0])),
                "accuracy_matrix_CIL": acc.tolist(),
                "ta_final": ta_final,
                **extras,
            }
            all_records.append(rec)
            print(f"  A_init_TA={rec['A_A_init_TA']:.2f} "
                  f"A_CIL={rec['A_A_final_CIL']:.2f} B_CIL={rec['A_B_final_CIL']:.2f} "
                  f"Ret={rec['cil_retention']:.2f}%")
            C.save_json(rec, marker)

    if a.skip_aggregate:
        return

    # aggregate (all prespecified seeds; retention = ratio of means)
    agg = {}
    for method in methods:
        rows = [r for r in all_records if r["method"] == method]
        rows = rows or [C.load_json(os.path.join(
            out_dir, f"{method.replace('+', 'p')}_seed_{s}.json")) for s in a.seeds]
        rows = [r for r in rows if r]
        if not rows:
            continue
        m_A_init = float(np.mean([r["A_A_init_TA"] for r in rows]))
        m_A_cil = float(np.mean([r["A_A_final_CIL"] for r in rows]))
        m_B_init = float(np.mean([r["A_B_init_TA"] for r in rows
                                  if r["A_B_init_TA"] is not None] or [0.0]))
        m_B_cil = float(np.mean([r["A_B_final_CIL"] for r in rows]))
        agg[method] = {
            "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
            "A_A_init_TA": C.mean_std([r["A_A_init_TA"] for r in rows]),
            "A_B_init_TA": C.mean_std([r["A_B_init_TA"] or 0.0 for r in rows]),
            "A_A_CIL": C.mean_std([r["A_A_final_CIL"] for r in rows]),
            "A_B_CIL": C.mean_std([r["A_B_final_CIL"] for r in rows]),
            "Avg_CIL": {"mean": (m_A_cil + m_B_cil) / 2.0, "std": 0.0, "n": len(rows)},
            "CIL_Retention_ratio_of_means":
                100.0 * m_A_cil / m_A_init if m_A_init > 0 else 0.0,
            "CIL_Retention_mean_of_ratios":
                C.mean_std([r["cil_retention"] for r in rows]),
            "forgetting": C.mean_std([r["forgetting_TA"] for r in rows]),
        }
    C.save_json(agg, os.path.join(out_dir, "aggregate.json"))
    print(f"\n=== AGGREGATE Table {a.table} ===")
    for mth, v in agg.items():
        print(f"  {mth:9s} | TA A {v['A_A_init_TA']['mean']:.2f}"
              f"±{v['A_A_init_TA']['std']:.2f} "
              f"B {v['A_B_init_TA']['mean']:.2f}"
              f"±{v['A_B_init_TA']['std']:.2f} | "
              f"CIL A {v['A_A_CIL']['mean']:.2f}±{v['A_A_CIL']['std']:.2f} "
              f"B {v['A_B_CIL']['mean']:.2f}±{v['A_B_CIL']['std']:.2f} | "
              f"Ret(ROM) {v['CIL_Retention_ratio_of_means']:.2f}%")


if __name__ == "__main__":
    main()
