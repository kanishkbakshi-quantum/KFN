"""
run_table4.py — Structural isolation ablation (Table IV).

Variants (manuscript Sec. IV-C, controlled reduced schedule, 2-task CIFAR-10):
  FULL        : channel isolation + phased unfreezing + replay anchoring
  REPLAY_ONLY : no expansion (d_new=0) -> no channel isolation
  FUSION_ONLY : expansion but NO replay anchoring (no KD / feature match / replay)
  NO_GRADUAL  : expansion + replay, but old gate released immediately

Output: results_table4/<variant>_seed_<s>.json + aggregate.json
Run:    python run_table4.py --seeds 0 1 2 3 4
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

VARIANTS = ["FULL", "REPLAY_ONLY", "FUSION_ONLY", "NO_GRADUAL"]
DATASET, CPT, N_TASKS = "cifar10", 5, 2
# Controlled reduced schedule (manuscript: "reduced epoch schedules and
# simplified augmentation"; absolute numbers differ from unconstrained runs)
SCALE = 0.5


def run_variant(variant, train_sets, ref_sets, class_maps, train_ds_ref,
                test_ds, device, seed, scale):
    cpt, n_tasks, dataset = CPT, N_TASKS, DATASET
    acc = np.zeros((n_tasks, n_tasks))
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
                                   cpt, classes, scale, variant=variant,
                                   old_bank=old_bank)
            nov, nd = rec["novelty"], rec["expansion_dim"]
            del spec

        if variant != "FUSION_ONLY":
            buffer.add(train_ds_ref, class_maps[t - 1], seed, dataset)

        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, cpt, dataset)
            acc[t - 1, tt - 1] = evaluate(model, tl, device, "cil", cpt, tt,
                                          n_exposed=t * cpt)

        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if t == n_tasks:
            extras = {"novelty_task2": nov, "expansion_dim_task2": nd,
                      "params_total": C.parameter_breakdown(model)["P_total"]}
    return acc, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", type=str, nargs="+", default=VARIANTS)
    ap.add_argument("--epochs-scale", type=float, default=SCALE)
    ap.add_argument("--out", default="./results_table4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = C.get_device()

    train_ds = C.load_dataset(DATASET, train=True)
    test_ds = C.load_dataset(DATASET, train=False)

    all_records = []
    for variant in a.variants:
        for seed in a.seeds:
            marker = os.path.join(a.out, f"{variant}_seed_{seed}.json")
            if C.load_json(marker) is not None:
                print(f"skip existing {marker}")
                continue
            print(f"\n=== Table4 {variant} seed {seed} on {device} ===")
            C.set_deterministic(seed)
            train_sets, ref_sets, class_maps = [], [], []
            for t in range(1, N_TASKS + 1):
                tr, _, rf, cm = C.split_task(train_ds, t, CPT, seed, DATASET)
                train_sets.append(tr)
                ref_sets.append(rf)
                class_maps.append(cm)
            acc, extras = run_variant(variant, train_sets, ref_sets, class_maps,
                                      train_ds, test_ds, device, seed, a.epochs_scale)
            A_init = float(acc[0, 0])
            A_final = float(acc[-1, 0])
            B_final = float(acc[-1, 1])
            retention = 100.0 * A_final / A_init if A_init > 0 else 0.0
            rec = {
                "variant": variant, "seed": seed,
                "A_A_init": A_init, "A_A_final": A_final, "A_B_final": B_final,
                "avg": float((A_final + B_final) / 2),
                "retention": retention, "forgetting": A_init - A_final,
                "accuracy_matrix": acc.tolist(), **extras,
            }
            all_records.append(rec)
            print(f"  A_init={A_init:.2f} A_final={A_final:.2f} "
                  f"B_final={B_final:.2f} Ret={retention:.2f}% "
                  f"F={rec['forgetting']:.2f}")
            C.save_json(rec, marker)

    agg = {}
    for variant in a.variants:
        rows = [r for r in all_records if r["variant"] == variant] or [
            C.load_json(os.path.join(a.out, f"{variant}_seed_{s}.json"))
            for s in a.seeds]
        rows = [r for r in rows if r]
        if not rows:
            continue
        agg[variant] = {
            "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
            "A_A_init": C.mean_std([r["A_A_init"] for r in rows]),
            "A_A_final": C.mean_std([r["A_A_final"] for r in rows]),
            "A_B_final": C.mean_std([r["A_B_final"] for r in rows]),
            "avg": C.mean_std([r["avg"] for r in rows]),
            "retention": C.mean_std([r["retention"] for r in rows]),
            "forgetting": C.mean_std([r["forgetting"] for r in rows]),
            "params_total": rows[-1]["params_total"],
        }
    C.save_json(agg, os.path.join(a.out, "aggregate.json"))
    print("\n=== AGGREGATE Table IV ===")
    for v, d in agg.items():
        print(f"  {v:12s} | A {d['A_A_final']['mean']:.2f}"
              f"±{d['A_A_final']['std']:.2f} | "
              f"B {d['A_B_final']['mean']:.2f}"
              f"±{d['A_B_final']['std']:.2f} | "
              f"Ret {d['retention']['mean']:.2f}% | "
              f"F {d['forgetting']['mean']:.2f} | "
              f"params {d['params_total']:,}")


if __name__ == "__main__":
    main()
