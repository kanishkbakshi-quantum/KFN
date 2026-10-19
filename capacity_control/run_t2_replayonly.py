"""
run_t2_replayonly.py — Capacity-matched shared control on the 10-task benchmark.

This is the experiment the reviewer will demand: a shared-parameter control with
the SAME total parameter count as KFN (28.1M), the SAME m=20/class replay budget,
the SAME per-task epoch schedule, and NO dedicated per-task subspaces.

Implementation: a single shared pathway (bottleneck widened to 64 + 64*(t-1)
channels) trained jointly on all tasks with no channel isolation. This is the
"what if we just gave the same capacity to a shared backbone" control.

Architecture:
  - ONE shared pathway that accumulates width exactly like KFN's total width
  - All specialists feed into a single gated aggregation -> widened bottleneck
  - NO fusion module, NO dedicated expansion block, NO frozen historical branches
  - Everything trained jointly on current + replay data

This matches KFN's total parameter count, total FLOPs, replay budget, and
per-task epoch schedule, so the ONLY difference is structural isolation vs
shared capacity.
"""
import sys, os, json, copy, random, argparse, time, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

# Minimal imports from kfn_core (reuse data loading, novelty, metrics)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kfn_core import (
    Specialist, CosineLinear, ReplayBuffer, set_deterministic, get_device,
    e, get_transforms, load_dataset, split_task, make_test_loader,
    evaluate, TensorImageDataset, EPOCHS_BASE, EPOCHS_SPECIALIST,
    EPOCHS_FUSION, EPOCHS_BIAS, BATCH_SIZE, REPLAY_BUFFER_CAPACITY,
    N_REF, EPS_NORM, EPS_SVD, D_HIGH, KD_TEMPERATURE,
    metrics_from_matrix, save_json, load_json, mean_std,
)

# ============================================================
# CONFIG
# ============================================================
CPT = 10           # classes per task
N_TASKS = 10
TOTAL_WIDTH = 64 + D_HIGH  # 128 at task 2, then +64 each task (accumulated)
DATASET = "cifar100"


class SharedWideModel(nn.Module):
    """Capacity-matched shared control for the 10-task benchmark.
    
    Same total width as KFN (64 + 64*(t-1)), same backbone, same replay,
    same epochs — but NO dedicated per-task subspaces, NO frozen historical
    branches, NO isolation. Everything trained jointly.
    """

    def __init__(self, n_classes: int, n_specs: int, mem_dim: int):
        super().__init__()
        self.n_specs = n_specs
        self.mem_dim = mem_dim
        self.specialists = nn.ModuleList([Specialist(64) for _ in range(n_specs)])
        self.old_proj = nn.ModuleList([nn.Conv2d(64, 64, 1) for _ in range(n_specs)])
        self.old_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True), nn.Conv2d(16, 64, 1), nn.Sigmoid())
        self.old_bottleneck = nn.Sequential(nn.Conv2d(64, mem_dim, 1),
                                            nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = CosineLinear(mem_dim, n_classes, sigma=10.0)

    def forward(self, x):
        feats = [s(x) for s in self.specialists]
        gated = []
        for i, f in enumerate(feats):
            p = self.old_proj[i](f)
            gated.append(p * self.old_gate(p))
        z = torch.stack(gated, 0).sum(0)
        z = self.old_bottleneck(z)
        flat = F.normalize(self.pool(z).flatten(1), p=2, dim=1)
        logits = self.classifier(flat)
        return logits, z

    def expand(self, n_new_classes: int):
        """Add new classifier columns and a new specialist + projection."""
        old_cls = self.classifier
        dev = old_cls.weight.device
        self.classifier = CosineLinear(self.mem_dim,
                                       self.n_classes + CPT, sigma=old_cls.sigma.item()).to(dev)
        with torch.no_grad():
            self.classifier.weight[:self.n_classes].copy_(old_cls.weight)
            self.classifier.sigma.data.copy_(old_cls.sigma.data)
        self.n_classes += CPT

        # add new specialist + projection
        new_spec = Specialist(64).to(dev)
        self.specialists.append(new_spec)
        new_proj = nn.Conv2d(64, 64, 1).to(dev)
        self.old_proj.append(new_proj)

        # widen the bottleneck by 64 channels (matching KFN's expansion)
        old_bn = self.old_bottleneck
        new_mem = self.mem_dim + D_HIGH
        new_bn = nn.Sequential(nn.Conv2d(64, new_mem, 1).to(dev),
                               nn.ReLU(inplace=True))
        with torch.no_grad():
            new_bn[0].weight[:self.mem_dim].copy_(old_bn[0].weight)
            if old_bn[0].bias is not None:
                new_bn[0].bias[:self.mem_dim].copy_(old_bn[0].bias)
        self.old_bottleneck = new_bn
        self.mem_dim = new_mem
        return self

    @property
    def n_classes(self):
        return self.classifier.weight.shape[0]

    @n_classes.setter
    def n_classes(self, v):
        pass  # handled in expand


def compute_novelty_shared(model, spec, ref_loader, device, mem_dim, old_bank):
    """Same novelty estimator as KFN (for protocol parity)."""
    spec.eval()
    new_desc, old_desc = [], []
    with torch.no_grad():
        for x, _ in ref_loader:
            x = x.to(device)
            fn = spec(x)
            new_desc.append(F.adaptive_avg_pool2d(fn, 1).flatten(1))
            if model is not None:
                # use all specialists through proj+gate+bottleneck
                feats = [s(x) for s in model.specialists]
                gated = []
                for i, f in enumerate(feats):
                    p = model.old_proj[i](f)
                    gated.append(p * model.old_gate(p))
                z = model.old_bottleneck(torch.stack(gated, 0).sum(0))
                old_desc.append(F.adaptive_avg_pool2d(z, 1).flatten(1))

    def nrm(v):
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v / torch.clamp(v.norm(p=2, dim=1, keepdim=True), min=EPS_NORM)

    F_new = nrm(torch.cat(new_desc, 0))
    if old_bank is None or old_bank.numel() == 0:
        return {"novelty": 0.0, "realised_rank": 0, "n_ref": int(F_new.size(0)), "n_basis": 0}

    B = old_bank.to(device)
    nb = min(32, B.size(0))
    if B.size(0) > nb:
        idx = torch.linspace(0, B.size(0) - 1, nb, device=B.device).long()
        B = B[idx]
    F_old = nrm(B)
    U, S, Vh = torch.linalg.svd(F_old.T, full_matrices=False)
    keep = S > EPS_SVD
    r = int(keep.sum().item())
    U_r = U[:, keep]
    P = U_r @ U_r.T
    resid = F_new - F_new @ P
    return {"novelty": float(resid.norm(p=2, dim=1).mean().item()),
            "realised_rank": r, "n_ref": int(F_new.size(0)), "n_basis": int(F_old.size(0))}


def build_old_bank_shared(model, buffer, device):
    loader = buffer.loader()
    if loader is None:
        return None
    bank = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feats = [s(x) for s in model.specialists]
            gated = []
            for i, f in enumerate(feats):
                p = model.old_proj[i](f)
                gated.append(p * model.old_gate(p))
            z = model.old_bottleneck(torch.stack(gated, 0).sum(0))
            bank.append(F.adaptive_avg_pool2d(z, 1).flatten(1).cpu())
    return torch.cat(bank, 0) if bank else None


def run_seed(seed: int, epochs_scale: float, out_dir: str, device: torch.device):
    set_deterministic(seed)
    train_ds = load_dataset(DATASET, train=True)
    test_ds = load_dataset(DATASET, train=False)

    ep_first = e(epochs_scale, EPOCHS_BASE)
    ep_incr = e(epochs_scale, EPOCHS_SPECIALIST + EPOCHS_FUSION + EPOCHS_BIAS)

    acc_m = np.zeros((N_TASKS, N_TASKS))
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    model = SharedWideModel(CPT, 1, mem_dim=64).to(device)
    novelty_recs = []
    param_history = []

    for t in range(1, N_TASKS + 1):
        tr_set, _, ref_set, cm = split_task(train_ds, t, CPT, seed, DATASET)
        classes = list(range(CPT * (t - 1), CPT * t))
        loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        ref_loader = DataLoader(ref_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        ep = ep_first if t == 1 else ep_incr

        if t > 1:
            model.expand(CPT)
            model = model.to(device)

        # compute novelty (same protocol as KFN, for comparability)
        old_bank = build_old_bank_shared(model, buffer, device) if t > 1 else None
        nov = compute_novelty_shared(model if t > 1 else None, model.specialists[-1] if t > 1 else model.specialists[0],
                                     ref_loader, device, model.mem_dim, old_bank)

        # all parameters trainable (no freezing, no isolation)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)

        rep = buffer.loader() if t > 1 else None
        for _ in range(ep):
            model.train()
            it_rep = iter(rep) if rep is not None else None
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                if rep is not None:
                    try:
                        xr, yr = next(it_rep)
                    except StopIteration:
                        it_rep = iter(rep)
                        xr, yr = next(it_rep)
                    xr, yr = xr.to(device), yr.to(device)
                    logits_r, _ = model(xr)
                    loss = loss + F.cross_entropy(logits_r, yr)
                loss.backward()
                opt.step()
            sched.step()

        buffer.add(train_ds, cm, seed, DATASET)

        # single test touch
        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, CPT, DATASET)
            acc_m[t - 1, tt - 1] = evaluate(model, tl, device, "cil", CPT, tt,
                                            n_exposed=t * CPT)

        p_total = sum(p.numel() for p in model.parameters())
        param_history.append({"task": t, "P_total": p_total})
        novelty_recs.append({"task": t, "novelty": nov})

        diag_acc = acc_m[t - 1, :t]
        print(f"  Task {t}: mem_dim={model.mem_dim} params={p_total:,} "
              f"nu={nov['novelty']:.4f} avg={np.mean(diag_acc):.2f}")

    m = metrics_from_matrix(acc_m, N_TASKS)
    result = {
        "method": "REPLAY_WIDE_10TASK", "seed": seed,
        "accuracy_matrix": acc_m.tolist(), **m,
        "param_history": param_history,
        "novelty_trajectory": novelty_recs,
        "final_params": param_history[-1]["P_total"],
        "mem_dim_final": model.mem_dim,
    }
    os.makedirs(out_dir, exist_ok=True)
    save_json(result, os.path.join(out_dir, f"REPLAY_WIDE_10TASK_seed_{seed}.json"))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="./results_t2_replaywide")
    args = ap.parse_args()
    device = get_device()
    for s in args.seeds:
        print(f"\n=== REPLAY_WIDE 10-task seed {s} on {device} ===")
        run_seed(s, args.epochs_scale, args.out, device)
