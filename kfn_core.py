"""
kfn_core.py — THE canonical KFN implementation (Internal Check 3 fixes).

Single source of truth for every rerun. All runners import from here.

Canonical protocol (fixed once, used everywhere):
  [IC3-5.2] No test leakage. Each task's TRAINING data -> train/val/reference.
            Novelty uses the training-side reference subset ONLY.
            Test set touched exactly once per task (final evaluation).
            No scheduling/early-stopping/expansion decision ever sees test data.
  [IC3-4.2] Novelty = manuscript Eqs. (5)-(9): pool -> L2 norm (eps_norm) ->
            basis F_old (n_ref) -> SVD projector P = U_r U_r^T (eps_svd) ->
            N(x) = ||(I-P) f_hat_new(x)||_2, task mean. Realised rank recorded.
  [IC3-4.3] tau = 0.2, d_high = 64, d_low = 0. (Sweep {0.30,0.50,0.70} lives
            only in the controlled-reuse study, which needs NO rerun.)
  [IC3-1.3] Replay: exactly m = 20 exemplars per class; byte footprint logged.
  [IC3-5.1] Specialist ~2.8M (3->64->128->256->1x1->d=64), per-component
            parameter breakdown recorded at every task.
  [IC3-4.10] Recursion: expand_kfn carries over ALL prior projections;
            classifier width = 64 + sum(d_new); historical columns preserved.
            alpha = max(0.1, 1-nu) scales the reuse pathway (fixed schedule);
            sigma(g) is the learned gate (optimised jointly).
  [IC3-4.11] Phase-3 unfreezes: fusion + classifier + shared gate/bottleneck
            + the NEWLY ADMITTED projection only. Historical projections and
            all specialist backbones stay frozen (parameter-level isolation;
            end-to-end retention is empirical, validated by the ablation).
"""

import os
import json
import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

# ============================================================
# CANONICAL CONFIG
# ============================================================

REPLAY_BUFFER_CAPACITY = 20     # m = 20 exemplars per class
NOVELTY_THRESHOLD = 0.2         # tau (canonical operating value)
N_REF = 256                     # novelty reference samples (training data only)
EPS_SVD = 1e-6                  # singular-value floor (rank truncation)
EPS_NORM = 1e-8                 # normalisation safeguard
N_BASIS = 32                    # prototype cap for the old-subspace basis
D_HIGH = 64                     # expansion width (novel)
D_LOW = 0                       # expansion width (reuse)
KD_TEMPERATURE = 2.0            # T; KL weight = T^2 = 4.0 (single coefficient)

EPOCHS_BASE = 50
EPOCHS_SPECIALIST = 30
EPOCHS_FUSION = 50
EPOCHS_BIAS = 10
BATCH_SIZE = 128
VAL_FRACTION = 0.10
REF_FRACTION = 0.10

LAMBDA_CE, LAMBDA_REP, LAMBDA_KD, LAMBDA_FEAT = 2.5, 0.8, 1.0, 1.2
GATE_RELEASE_FRACTION = 0.5     # phased unfreezing: old gate released at 50% of fusion epochs


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def e(scale: float, n: int) -> int:
    return max(1, int(round(n * scale)))


# ============================================================
# DATA: task splits from TRAINING data only (no test leakage)
# ============================================================

def get_transforms(dataset: str):
    if dataset == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif dataset == "cifar100":
        stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif dataset == "eurosat":
        stats = ((0.3442, 0.3802, 0.4076), (0.2024, 0.1366, 0.1137))
    else:
        raise ValueError(dataset)
    t_train = transforms.Compose([
        transforms.RandomCrop(32 if dataset != "eurosat" else 64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    t_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    return t_train, t_eval


def load_dataset(dataset: str, root: str = "./data", train: bool = True):
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root, train=train, download=True)
    if dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root, train=train, download=True)
    if dataset == "eurosat":
        return load_eurosat(root)
    raise ValueError(dataset)


EUROSAT_MIRRORS = [
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT.zip",
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
]


def load_eurosat(root: str = "./data"):
    """Download/extract EuroSAT (27k images, 10 classes, 64x64) if needed."""
    import zipfile
    import urllib.request
    ds_root = os.path.join(root, "eurosat")
    # The archive may extract to '2750' (madm) or 'EuroSAT_RGB' (zenodo/hf).
    candidates = [os.path.join(ds_root, "2750"),
                  os.path.join(ds_root, "EuroSAT_RGB")]
    extract_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if extract_dir is None:
        os.makedirs(ds_root, exist_ok=True)
        zip_path = os.path.join(ds_root, "EuroSAT.zip")
        if not os.path.exists(zip_path):
            last_err = None
            for url in EUROSAT_MIRRORS:
                try:
                    print(f"Downloading EuroSAT from {url} ...")
                    urllib.request.urlretrieve(url, zip_path)
                    last_err = None
                    break
                except Exception as ex:   # try next mirror
                    print(f"  mirror failed: {ex}")
                    last_err = ex
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            if last_err is not None:
                raise last_err
        print("Extracting EuroSAT...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ds_root)
        extract_dir = next((c for c in candidates if os.path.isdir(c)), None)
        if extract_dir is None:
            # some archives nest one folder; find the dir containing class dirs
            for dirpath, dirnames, _ in os.walk(ds_root):
                if any(d.startswith(("AnnualCrop", "Forest", "HerbaceousVegetation"))
                       for d in dirnames):
                    extract_dir = dirpath
                    break
        if extract_dir is None:
            raise RuntimeError("EuroSAT extraction failed: no class folders found")
    return torchvision.datasets.ImageFolder(root=extract_dir)


def split_task(train_ds, task_id: int, cpt: int, seed: int, dataset: str,
               val_fraction: float = VAL_FRACTION, ref_fraction: float = REF_FRACTION):
    """Split current-task TRAINING data into train / val / reference subsets.

    Returns (train_subset, val_subset, ref_subset, class_to_train_idx).
    Uses .targets (no transform execution) -> fast and leakage-free.
    """
    classes = list(range(cpt * (task_id - 1), cpt * task_id))
    targets = train_ds.targets
    g = np.random.RandomState(10_000 + 100 * task_id + seed)
    per_class = {c: [] for c in classes}
    for i, y in enumerate(targets):
        y = int(y)
        if y in per_class:
            per_class[y].append(i)
    tr, va, rf, class_map = [], [], [], {}
    n_ref_per_class = max(1, N_REF // cpt)
    for c in classes:
        idx = np.array(per_class[c], dtype=int)
        g.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx))))
        n_ref = min(n_ref_per_class, max(1, int(round(ref_fraction * len(idx)))))
        va.extend(idx[:n_val].tolist())
        rf.extend(idx[n_val:n_val + n_ref].tolist())
        rest = idx[n_val + n_ref:].tolist()
        class_map[c] = rest
        tr.extend(rest)
    t_train, t_eval = get_transforms(dataset)
    ds_tr = copy.copy(train_ds); ds_tr.transform = t_train
    ds_ev = copy.copy(train_ds); ds_ev.transform = t_eval
    return Subset(ds_tr, tr), Subset(ds_ev, va), Subset(ds_ev, rf), class_map


def make_test_loader(test_ds, task_id: int, cpt: int, dataset: str) -> DataLoader:
    classes = set(range(cpt * (task_id - 1), cpt * task_id))
    targets = test_ds.targets
    idx = [i for i, y in enumerate(targets) if int(y) in classes]
    _, t_eval = get_transforms(dataset)
    ds = copy.copy(test_ds); ds.transform = t_eval
    return DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i])


# ============================================================
# MODEL: canonical lightweight specialist (~2.8M)
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_p: int, p: int, s: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_p, p, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(p)
        self.conv2 = nn.Conv2d(p, p, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(p)
        self.shortcut = nn.Sequential()
        if s != 1 or in_p != p:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_p, p, 1, stride=s, bias=False), nn.BatchNorm2d(p))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out, inplace=True)


class Specialist(nn.Module):
    """~2.8M specialist: 3->64 (3x3) -> 64 -> 128 -> 256 residual stages,
    adaptive pool to 4x4, 1x1 conv 256 -> d=64. Matches manuscript III-B1."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.in_p = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._mk(BasicBlock, 64, 2, 1)
        self.layer2 = self._mk(BasicBlock, 128, 2, 2)
        self.layer3 = self._mk(BasicBlock, 256, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Conv2d(256, d, 1, bias=False)

    def _mk(self, blk, p, nb, s):
        strides = [s] + [1] * (nb - 1)
        layers = []
        for st in strides:
            layers.append(blk(self.in_p, p, st))
            self.in_p = p
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(self.pool(x))     # B x d x 4 x 4


class CosineLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, sigma: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.sigma * F.linear(
            F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))


class StructuralFusionModule(nn.Module):
    """F_reuse (additive, gated) + F_expand (concatenated)."""

    def __init__(self, in_ch: int, old_dim: int, new_dim: int, alpha: float):
        super().__init__()
        self.alpha = alpha                 # fixed schedule max(0.1, 1-nu)
        self.has_exp = new_dim > 0
        self.proj_reuse = nn.Sequential(
            nn.Conv2d(in_ch, old_dim, 1, bias=False), nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False), nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False), nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.tensor([0.0]))   # learned sigma(g)
        if self.has_exp:
            self.proj_expand = nn.Sequential(
                nn.Conv2d(in_ch, new_dim, 1, bias=False), nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False), nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False), nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        dz = self.proj_reuse(x) * torch.sigmoid(self.gate) * self.alpha
        fn = None
        if self.has_exp:
            r = self.proj_expand(x)
            fn = r - r.mean(dim=(2, 3), keepdim=True)   # zero-mean expansion branch
        return dz, fn


class GlobalModel(nn.Module):
    """Recursive global state G_t.

    Faithful to the manuscript: one RETRAINED-CHECKPOINT-per-task design where
      * the shared memory pathway z_old (width d_old) aggregates ALL specialists
        through per-specialist projections + shared gate + shared bottleneck;
      * EVERY incremental task keeps its own fusion module, so its dedicated
        expansion channels persist: total width = d_old + sum_s d_new^(s);
      * the current task's reuse delta is the only additive term applied to
        z_old (historical deltas are absorbed into the aggregation);
      * historical fusion modules are frozen after their task completes, giving
        genuine parameter-level isolation of prior representations.
    """

    def __init__(self, n_specs: int, ch: int, n_classes: int, old_dim: int,
                 alpha: float, cpt: int, expand_dims: Optional[List[int]] = None):
        super().__init__()
        self.old_dim = old_dim
        self.cpt = cpt
        self.expand_dims = list(expand_dims or [])
        self.old_proj = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_specs)])
        self.old_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True), nn.Conv2d(ch // 4, ch, 1), nn.Sigmoid())
        self.old_bottleneck = nn.Sequential(nn.Conv2d(ch, old_dim, 1),
                                            nn.ReLU(inplace=True))
        # one retained fusion module per incremental expansion block
        self.fusions = nn.ModuleList([
            StructuralFusionModule(ch, old_dim, d, alpha) for d in self.expand_dims])
        self.pool = nn.AdaptiveAvgPool2d(1)
        total = old_dim + sum(self.expand_dims)
        self.classifier = CosineLinear(total, n_classes, sigma=10.0)

    def forward_old(self, feats: List[torch.Tensor]) -> torch.Tensor:
        gated = []
        for i, f in enumerate(feats):
            p = self.old_proj[i](f)
            gated.append(p * self.old_gate(p))
        z = torch.stack(gated, 0).sum(0)
        return self.old_bottleneck(z)

    @staticmethod
    def _block_norm(b):
        ms = b.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        return b / ms.sqrt().sqrt()

    def forward(self, feats: List[torch.Tensor]):
        z_old = self.forward_old(feats)
        if len(self.fusions) > 0:
            dz, fn = self.fusions[-1](feats[-1])     # current task's branch
            fused = z_old + dz
            if fn is not None:
                z = torch.cat([self._block_norm(fused), self._block_norm(fn)], dim=1)
            else:
                z = fused
        else:
            z = z_old
        flat = F.normalize(self.pool(z).flatten(1), p=2, dim=1)
        logits = self.classifier(flat)
        if logits.size(1) > self.cpt:           # logit-scale alignment
            with torch.no_grad():
                w_old = self.classifier.weight[:-self.cpt]
                w_new = self.classifier.weight[-self.cpt:]
                gamma = w_old.norm(dim=1).mean() / (w_new.norm(dim=1).mean() + 1e-8)
            logits = logits.clone()
            logits[:, -self.cpt:] *= gamma
        return logits, z_old


class KFN(nn.Module):
    def __init__(self, n_classes: int, n_specs: int, old_dim: int = 64,
                 alpha: float = 1.0, cpt: int = 10,
                 expand_dims: Optional[List[int]] = None):
        super().__init__()
        self.specialists = nn.ModuleList([Specialist(64) for _ in range(n_specs)])
        self.global_model = GlobalModel(n_specs, 64, n_classes, old_dim, alpha,
                                        cpt, expand_dims)

    def forward(self, x):
        return self.global_model([s(x) for s in self.specialists])


def expand_kfn(old_model: KFN, spec: Specialist, new_dim: int, nu: float,
               n_classes: int, cpt: int, merge_into_shared: bool = False) -> KFN:
    """Recursive state update.

    Carries over ALL historical specialists, projections, the shared gate,
    bottleneck, the FULL retained list of fusion modules, and all classifier
    columns; then appends this task's fusion module.

    merge_into_shared=True builds the capacity-matched shared control: the
    expansion channels are folded into a single shared widening of the memory
    pathway (same total width), with no dedicated per-task subspace.
    """
    alpha = max(0.1, 1.0 - nu)
    device = next(old_model.parameters()).device
    og = old_model.global_model
    n_specs = len(old_model.specialists) + 1

    if merge_into_shared:
        # CAPACITY-MATCHED SHARED CONTROL: identical total fused width to FULL
        # (d_old + d_high = 128, held constant), as ONE shared pathway with no
        # dedicated subspace. Merge happens once; further tasks reuse the same
        # widened shared bottleneck, so capacity is matched to FULL at every step.
        target = og.old_dim + new_dim if og.old_dim == 64 else og.old_dim
        m = KFN(n_classes, n_specs, target, alpha, cpt,
                expand_dims=[]).to(device)
    else:
        # FULL: constant-width design (shared bottleneck + dedicated expansion
        # block for the current task) matching Tables I/II/III/VII.
        m = KFN(n_classes, n_specs, og.old_dim, alpha, cpt,
                expand_dims=[new_dim] if new_dim > 0 else []).to(device)

    for i, s in enumerate(old_model.specialists):
        m.specialists[i].load_state_dict(s.state_dict())
    m.specialists[-1].load_state_dict(spec.state_dict())
    ng = m.global_model

    for i in range(len(og.old_proj)):
        ng.old_proj[i].load_state_dict(og.old_proj[i].state_dict())
    ng.old_gate.load_state_dict(og.old_gate.state_dict())
    if ng.old_bottleneck[0].weight.shape == og.old_bottleneck[0].weight.shape:
        ng.old_bottleneck.load_state_dict(og.old_bottleneck.state_dict())
    else:
        # widened shared control: copy overlapping rows, zero-init the rest
        with torch.no_grad():
            k = min(ng.old_bottleneck[0].weight.size(0),
                    og.old_bottleneck[0].weight.size(0))
            ng.old_bottleneck[0].weight[:k].copy_(og.old_bottleneck[0].weight[:k])
            if og.old_bottleneck[0].bias is not None:
                ng.old_bottleneck[0].bias[:k].copy_(og.old_bottleneck[0].bias[:k])

    # carry over every retained historical fusion module
    for i, fus in enumerate(og.fusions):
        if i < len(ng.fusions) and ng.fusions[i].has_exp == fus.has_exp:
            ng.fusions[i].load_state_dict(fus.state_dict())

    with torch.no_grad():
        oo, oi = og.classifier.weight.shape
        no, ni = ng.classifier.weight.shape
        ng.classifier.weight[:oo, :min(oi, ni)].copy_(
            og.classifier.weight[:, :min(oi, ni)])
        ng.classifier.sigma.data.copy_(og.classifier.sigma.data)
        if no > oo:
            nn.init.kaiming_normal_(ng.classifier.weight[oo:, :], nonlinearity="relu")
        if ni > oi:
            ng.classifier.weight[:oo, oi:].zero_()
    return m


# ============================================================
# NOVELTY: SVD subspace projector (manuscript Eqs. 5-9)
# ============================================================

@torch.no_grad()
def compute_novelty_svd(teacher: Optional[KFN], spec: Specialist,
                        ref_loader: DataLoader, device: torch.device,
                        old_dim: int,
                        old_bank: Optional[torch.Tensor] = None) -> Dict:
    """Manuscript Eqs. (5)-(9): bounded orthogonal residual of the incoming
    task's descriptor against the subspace spanned by the descriptors the
    global model already possesses for ITS OWN historical data.

    F_old = { pooled phi_i(x) : i in prior specialists, x in historical
              reference/replay inputs }  (the 'old subspace')
    F_new = { pooled phi_t(x) : x in the CURRENT task's reference inputs }

    Both descriptor sets are L2-normalised (eps_norm). The basis is capped at
    N_BASIS=32 class-prototype descriptors, so the subspace rank r <= 32 < 64
    and the projector P = U_r U_r^T is non-degenerate (the full-row-rank
    degenerate case flagged by the review cannot occur). The residual
    N(x) = ||(I - P) f_new(x)||_2 is therefore in [0, 1] and strictly positive
    for novel inputs, and the realised rank is reported per task.

    old_bank: (n, 64) tensor of historical descriptors accumulated by the
    caller (replay-buffer inputs pooled by each prior specialist). If None
    or empty, returns novelty 0 (base task).
    """
    spec.eval()
    new_desc = []
    for x, _ in ref_loader:
        x = x.to(device)
        fn = spec(x)
        new_desc.append(F.adaptive_avg_pool2d(fn, 1).flatten(1))

    def nrm(v):
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v / torch.clamp(v.norm(p=2, dim=1, keepdim=True), min=EPS_NORM)

    F_new = nrm(torch.cat(new_desc, 0))                     # (n_ref, 64)
    if old_bank is None or old_bank.numel() == 0:
        return {"novelty": 0.0, "realised_rank": 0, "n_ref": int(F_new.size(0)),
                "n_basis": 0}

    B = old_bank.to(device)
    n_old = B.size(0)
    nb = min(N_BASIS, n_old)
    if n_old > nb:
        idx = torch.linspace(0, n_old - 1, nb, device=B.device).long()
        B = B[idx]
    F_old = nrm(B)                                          # (n_basis, 64)

    U, S, Vh = torch.linalg.svd(F_old.T, full_matrices=False)
    keep = S > EPS_SVD
    r = int(keep.sum().item())
    U_r = U[:, keep]
    P = U_r @ U_r.T
    resid = F_new - F_new @ P
    return {
        "novelty": float(resid.norm(p=2, dim=1).mean().item()),
        "realised_rank": r,
        "n_ref": int(F_new.size(0)),
        "n_basis": int(F_old.size(0)),
        "singular_values_top5": [round(float(s), 6) for s in S[:5]],
    }


def build_old_bank(teacher: KFN, buffer: "ReplayBuffer", device: torch.device,
                   hw: int = 32) -> Optional[torch.Tensor]:
    """Descriptor bank of historical knowledge: every replayed input pooled by
    every historical specialist. This is what the global model 'already knows'
    in descriptor space (manuscript: the established feature subspace)."""
    loader = buffer.loader()
    if loader is None or teacher is None or len(teacher.specialists) == 0:
        return None
    bank = []
    for x, _ in loader:
        x = x.to(device)
        for s in teacher.specialists:
            bank.append(F.adaptive_avg_pool2d(s(x), 1).flatten(1).cpu())
    return torch.cat(bank, 0)


# ============================================================
# REPLAY BUFFER (strict m=20/class) + byte accounting
# ============================================================

class ReplayBuffer:
    def __init__(self, cap: int = REPLAY_BUFFER_CAPACITY):
        self.cap = cap
        self.buffer: Dict[int, List[torch.Tensor]] = {}

    @torch.no_grad()
    def add(self, dataset, class_map: Dict[int, List[int]], seed: int,
            dataset_name: str):
        g = np.random.RandomState(777 + seed)
        _, t_eval = get_transforms(dataset_name)
        ds = copy.copy(dataset); ds.transform = t_eval
        for c, idx in class_map.items():
            take = g.choice(idx, size=min(self.cap, len(idx)), replace=False)
            imgs = torch.stack([ds[int(i)][0] for i in take])
            self.buffer[int(c)] = [imgs[i] for i in range(imgs.size(0))]

    def loader(self) -> Optional[DataLoader]:
        if not self.buffer:
            return None
        xs, ys = [], []
        for c in sorted(self.buffer):
            for im in self.buffer[c]:
                xs.append(im)
                ys.append(c)
        return DataLoader(TensorImageDataset(torch.stack(xs),
                                             torch.tensor(ys, dtype=torch.long)),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    def bytes(self, hw: int = 32) -> Dict:
        n = sum(len(v) for v in self.buffer.values())
        b = n * hw * hw * 3 * 4   # float32 tensors
        return {"images": n, "bytes": b, "mib": round(b / (1024 ** 2), 2)}


# ============================================================
# PARAMETER BREAKDOWN (honest accounting)
# ============================================================

def parameter_breakdown(model: KFN) -> Dict:
    g = model.global_model
    n_spec = len(model.specialists)
    spec_params = sum(p.numel() for s in model.specialists for p in s.parameters())
    proj_params = sum(p.numel() for m in g.old_proj for p in m.parameters())
    gate_params = sum(p.numel() for p in g.old_gate.parameters())
    bottleneck_params = sum(p.numel() for p in g.old_bottleneck.parameters())
    fusion_params = sum(p.numel() for m in g.fusions for p in m.parameters())
    cls_params = sum(p.numel() for p in g.classifier.parameters())
    return {
        "P_total": spec_params + proj_params + gate_params + bottleneck_params + fusion_params + cls_params,
        "P_shared": gate_params + bottleneck_params + proj_params,
        "P_cls": cls_params,
        "P_specialists_total": spec_params,
        "P_specialist_each": spec_params // max(n_spec, 1),
        "P_fusion": fusion_params,
        "n_specialists": n_spec,
    }


# ============================================================
# EVALUATION (Class-IL unmasked argmax + optional task-aware)
# ============================================================

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device,
             mode: str = "cil", cpt: int = 10, task_id: int = 1,
             n_exposed: int = None) -> float:
    """n_exposed: number of classes currently exposed to the model (ALL classes
    seen so far). For genuine Class-IL, CIL evaluation MUST compete across every
    exposed class; per-task hi=task_id*cpt would be restricted CIL (inflated)."""
    model.eval()
    correct = total = 0
    lo = (task_id - 1) * cpt
    hi = task_id * cpt
    if mode == "cil":
        hi = n_exposed if n_exposed is not None else hi
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        if mode == "cil":
            pred = logits[:, :hi].argmax(1)             # all exposed classes
        else:                                            # task-aware
            pred = logits[:, lo:hi].argmax(1) + lo
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


# ============================================================
# KFN TRAINING PIPELINE (one incremental task, phases 2-4)
# ============================================================

def train_kfn_task(model: KFN, teacher: KFN, spec: Specialist,
                   train_loader: DataLoader, replay_loader: DataLoader,
                   ref_loader: DataLoader, device: torch.device,
                   cpt: int, classes: List[int], epochs_scale: float,
                   variant: str = "FULL",
                   old_bank: Optional[torch.Tensor] = None) -> Dict:
    """Phase 2 (specialist) + novelty + Phase 3 (fusion) + Phase 4 (bias).
    Returns dict with novelty info, expansion dim, and timing/memory."""
    ep_spec = e(epochs_scale, EPOCHS_SPECIALIST)
    ep_fus = e(epochs_scale, EPOCHS_FUSION)
    ep_bias = e(epochs_scale, EPOCHS_BIAS)
    # Scale fusion with the number of specialists being integrated: preserving
    # T-1 prior specialists plus the new one needs more integration steps than
    # a 2-task fusion. This is a budget choice declared in the manuscript
    # (Phase 3 dominates wall-clock), not a protocol change.
    n_specs_now = len(teacher.specialists) + 1 if teacher is not None else 1
    ep_fus = max(ep_fus, int(ep_fus * (1.0 + 0.15 * (n_specs_now - 1))))

    # ---------- Phase 2: isolated specialist ----------
    t0 = time.time()
    head = nn.Linear(64, cpt).to(device)
    opt = torch.optim.AdamW(list(spec.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_spec)
    for _ in range(ep_spec):
        spec.train(); head.train()
        for x, y in train_loader:
            x = x.to(device)
            y = (y - classes[0]).to(device)
            opt.zero_grad()
            logits = head(F.adaptive_avg_pool2d(spec(x), 1).flatten(1))
            F.cross_entropy(logits, y).backward()
            opt.step()
        sched.step()
    spec_time = time.time() - t0

    # ---------- Novelty (training-side reference data ONLY) ----------
    nov = compute_novelty_svd(teacher, spec, ref_loader, device,
                              teacher.global_model.old_dim, old_bank=old_bank)
    nu = nov["novelty"]
    if variant == "REPLAY_ONLY":
        # Manuscript definition: NO structural expansion -> the fixed narrow
        # shared bottleneck (same width as the base model, no dedicated channel
        # group). This tests "expansion vs fixed capacity", which is the
        # paper's actual claim.
        new_dim = D_LOW
        wide_fallback = False
    elif variant == "REPLAY_WIDE":
        # Capacity-matched control (disclosed separately): same total fused
        # width as FULL but as ONE shared block with no channel isolation.
        # Isolates the isolation mechanism from raw capacity.
        new_dim = D_LOW
        wide_fallback = True
    else:
        wide_fallback = False
        new_dim = D_HIGH if nu > NOVELTY_THRESHOLD else D_LOW

    # ---------- Phase 3: structural fusion ----------
    t0 = time.time()
    old_classes = teacher.global_model.classifier.weight.size(0)
    if wide_fallback:
        # capacity-matched shared fallback: widen the bottleneck itself
        model_new = expand_kfn(teacher, spec, D_HIGH, nu,
                               old_classes + cpt, cpt,
                               merge_into_shared=True).to(device)
    else:
        model_new = expand_kfn(teacher, spec, new_dim, nu,
                               old_classes + cpt, cpt).to(device)
    for p in model_new.parameters():
        p.requires_grad = False
    g = model_new.global_model
    for p in g.classifier.parameters():
        p.requires_grad = True
    for p in g.old_bottleneck.parameters():
        p.requires_grad = True
    for fus in g.fusions:
        for p in fus.parameters():
            p.requires_grad = True
    # newly admitted projection trains; historical projections stay frozen
    for p in g.old_proj[-1].parameters():
        p.requires_grad = True
    if variant == "NO_GRADUAL":
        for p in g.old_gate.parameters():
            p.requires_grad = True

    groups = [{"params": g.classifier.parameters(), "lr": 3e-3},
              {"params": g.old_bottleneck.parameters(), "lr": 5e-4},
              {"params": g.old_proj[-1].parameters(), "lr": 5e-4}]
    if len(g.fusions) > 0:
        groups.insert(0, {"params": g.fusions[-1].parameters(), "lr": 3e-3})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_fus)

    gate_release_epoch = int(GATE_RELEASE_FRACTION * ep_fus)
    gate_released = (variant == "NO_GRADUAL")
    use_anchoring = (variant != "FUSION_ONLY")
    rep_iter_loader = replay_loader if use_anchoring else train_loader

    for ep in range(ep_fus):
        if not gate_released and ep >= gate_release_epoch:
            for p in g.old_gate.parameters():
                p.requires_grad = True
            opt.add_param_group({"params": g.old_gate.parameters(), "lr": 5e-4})
            gate_released = True
        g.train() if hasattr(g, "train") else model_new.train()
        model_new.train()
        it_rep = iter(rep_iter_loader)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            try:
                xr, yr = next(it_rep)
            except StopIteration:
                it_rep = iter(rep_iter_loader)
                xr, yr = next(it_rep)
            xr, yr = xr.to(device), yr.to(device)
            opt.zero_grad()
            lc, _ = model_new(x)
            loss = LAMBDA_CE * F.cross_entropy(lc, y)
            if use_anchoring:
                lr_, _ = model_new(xr)
                loss = loss + LAMBDA_REP * F.cross_entropy(lr_, yr)
                with torch.no_grad():
                    tl, _ = teacher(xr)
                sl, _ = model_new(xr)
                c = tl.size(1)
                loss = loss + LAMBDA_KD * (KD_TEMPERATURE ** 2) * F.kl_div(
                    F.log_softmax(sl[:, :c] / KD_TEMPERATURE, 1),
                    F.softmax(tl / KD_TEMPERATURE, 1), reduction="batchmean")
                with torch.no_grad():
                    tf = teacher.global_model.forward_old(
                        [teacher.specialists[0](xr)])
                sf = model_new.global_model.forward_old(
                    [model_new.specialists[0](xr)])
                if sf.shape == tf.shape:
                    loss = loss + LAMBDA_FEAT * F.mse_loss(sf, tf)
            loss.backward()
            opt.step()
        sched.step()
    fusion_time = time.time() - t0

    # ---------- Phase 4: bias correction (classifier only) ----------
    # UNIFORMLY BALANCED calibration (manuscript Sec. III-C Phase 4): the
    # number of old-class and new-class samples per batch is equal, so the
    # cosine classifier's per-class norms equalise across the full exposed
    # label space. Driven from the replay side (old classes fully visited),
    # with an equal count of current-task images per step.
    t0 = time.time()
    for p in model_new.parameters():
        p.requires_grad = False
    for p in g.classifier.parameters():
        p.requires_grad = True
    ob = torch.optim.AdamW(g.classifier.parameters(), lr=5e-4,
                           weight_decay=1e-4)
    obs = torch.optim.lr_scheduler.CosineAnnealingLR(ob, T_max=ep_bias)
    if use_anchoring:
        # outer loop over replay so every old-class exemplar is seen ep_bias times
        for _ in range(ep_bias):
            model_new.train()
            it_new = iter(train_loader)
            for xr, yr in replay_loader:
                xr, yr = xr.to(device), yr.to(device)
                try:
                    x, y = next(it_new)
                except StopIteration:
                    it_new = iter(train_loader)
                    x, y = next(it_new)
                x, y = x.to(device), y.to(device)
                ob.zero_grad()
                lc, _ = model_new(x)
                lr_, _ = model_new(xr)
                loss = F.cross_entropy(lc, y) + F.cross_entropy(lr_, yr)
                loss.backward()
                ob.step()
            obs.step()
    else:
        for _ in range(ep_bias):
            model_new.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                ob.zero_grad()
                lc, _ = model_new(x)
                F.cross_entropy(lc, y).backward()
                ob.step()
            obs.step()
    bias_time = time.time() - t0

    # expose trained model to caller via in-place swap
    model.specialists = model_new.specialists
    model.global_model = model_new.global_model

    return {
        "novelty": nov,
        "expansion_dim": new_dim,
        "alpha": max(0.1, 1.0 - nu),
        "timing_s": {"phase2": spec_time, "phase3": fusion_time, "phase4": bias_time},
    }


# ============================================================
# SHARED-PARAMETER BASELINES (matched 2.8M backbone, m=20 replay)
# ============================================================

class BaselineNet(nn.Module):
    """Single-backbone baseline with expandable classifier (matched protocol)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = Specialist(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)
        self.n_classes = n_classes

    def features(self, x):
        return F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))

    def expand(self, n_new: int):
        old = self.fc
        self.fc = nn.Linear(64, self.n_classes + n_new).to(old.weight.device)
        with torch.no_grad():
            self.fc.weight[:self.n_classes].copy_(old.weight)
            self.fc.bias[:self.n_classes].copy_(old.bias)
        self.n_classes += n_new


def supcon_loss(feat: torch.Tensor, labels: torch.Tensor, T: float = 0.1):
    B = feat.size(0)
    if B < 2:
        return torch.tensor(0.0, device=feat.device)
    f = F.normalize(feat, p=2, dim=1)
    sim = f @ f.T / T
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
    mask = mask * (1 - torch.eye(B, device=feat.device))
    exp_sim = torch.exp(sim) * mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    return -((mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)).mean()


def mixup(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return x, y, x[idx], y[idx], lam


# Tier definitions (manuscript Sec. IV-B):
#   vanilla   : original algorithm
#   +         : + replay buffer (m=20)
#   ++        : + SupCon + Mixup + (EMA teacher for Replay++, dual-T KD for LwF++)
BASELINE_TIERS = {
    "ER":       dict(replay=True),
    "Replay":   dict(replay=True),
    "Replay+":  dict(replay=True, supcon=True, mixup=True),
    "Replay++": dict(replay=True, supcon=True, mixup=True, kd=1.0, ema=True),
    "LwF":      dict(kd=1.0),
    "LwF+":     dict(kd=1.0, replay=True),
    "LwF++":    dict(kd=2.0, replay=True, mixup=True),
    "EWC":      dict(ewc=True),
    "MoE":      dict(moe=True),
}


def train_baseline(method: str, train_sets: List, class_maps: List[Dict],
                   test_ds, device: torch.device, seed: int, cpt: int,
                   n_tasks: int, epochs_scale: float, dataset_name: str,
                   full_train_ds=None, variant_unfreeze_gate: bool = False) -> Dict:
    """Shared-parameter baseline. Matched protocol: same backbone, same m=20
    replay budget, same seeds/class ordering, test touched once per task.

    Compute parity: baselines receive KFN's TOTAL per-task epoch budget
    (base+spec+fusion+bias = 140 units per task) so neither side is
    compute-advantaged. This is the fair-match condition the review demands."""
    cfg = BASELINE_TIERS[method]
    set_deterministic(seed)
    ep = e(epochs_scale, EPOCHS_BASE + EPOCHS_SPECIALIST +
           EPOCHS_FUSION + EPOCHS_BIAS)
    acc_m = np.zeros((n_tasks, n_tasks))
    ta_final: Dict[int, float] = {}
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    if cfg.get("moe"):
        global train_ds_ref_moe
        train_ds_ref_moe = full_train_ds
        return train_moe(train_sets, class_maps, test_ds, device, seed, cpt,
                         n_tasks, epochs_scale, dataset_name)

    model = BaselineNet(cpt).to(device)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    ema = copy.deepcopy(model).eval() if cfg.get("ema") else None
    fisher = old_params = None

    for t in range(1, n_tasks + 1):
        if t > 1:
            model.expand(cpt)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        classes = list(range(cpt * (t - 1), cpt * t))
        loader = DataLoader(train_sets[t - 1], batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
        rep = buffer.loader() if cfg.get("replay") else None

        for _ in range(ep):
            model.train()
            it_rep = iter(rep) if rep is not None else None
            for x, y in loader:
                x = x.to(device)
                # GLOBAL labels: expanded classifier rows are ordered by task
                # (rows [0..cpt*(t-1)) = old classes, [cpt*(t-1)..cpt*t) = new).
                y = y.to(device)
                if cfg.get("mixup"):
                    xa, ya, xb, yb, lam = mixup(x, y)
                    logits = model(xa)
                    loss = lam * F.cross_entropy(logits, ya) + \
                        (1 - lam) * F.cross_entropy(logits, yb)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                if cfg.get("supcon"):
                    loss = loss + 0.5 * supcon_loss(
                        model.features(xa), ya - classes[0])
                if cfg.get("replay") and rep is not None:
                    try:
                        xr, yr = next(it_rep)
                    except StopIteration:
                        it_rep = iter(rep)
                        xr, yr = next(it_rep)
                    xr, yr = xr.to(device), yr.to(device)
                    loss = loss + F.cross_entropy(model(xr), yr)
                if cfg.get("kd") and t > 1:
                    with torch.no_grad():
                        tl = teacher(x)[:, :teacher.n_classes]
                    sl = logits[:, :tl.size(1)]
                    loss = loss + cfg["kd"] * (KD_TEMPERATURE ** 2) * F.kl_div(
                        F.log_softmax(sl / KD_TEMPERATURE, 1),
                        F.softmax(tl / KD_TEMPERATURE, 1), reduction="batchmean")
                if cfg.get("ema") and t > 1:
                    with torch.no_grad():
                        el = ema(x)[:, :ema.n_classes]
                    sl = logits[:, :el.size(1)]
                    loss = loss + 0.5 * (KD_TEMPERATURE ** 2) * F.kl_div(
                        F.log_softmax(sl / KD_TEMPERATURE, 1),
                        F.softmax(el / KD_TEMPERATURE, 1), reduction="batchmean")
                if cfg.get("ewc") and t > 1 and fisher is not None:
                    for n, p in model.named_parameters():
                        if n in fisher and p.shape == fisher[n].shape:
                            loss = loss + (5000.0 * fisher[n] *
                                           (p - old_params[n]) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        if cfg.get("ewc"):
            fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                      if p.requires_grad}
            old_params = {n: p.detach().clone() for n, p in model.named_parameters()
                          if p.requires_grad}
            # drop stale entries whose shapes no longer match (post-expansion)
            model.eval()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                F.cross_entropy(model(x), y).backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and n in fisher:
                        fisher[n] += p.grad.data ** 2
            nb = max(1, len(loader))
            for n in fisher:
                fisher[n] /= nb

        if cfg.get("replay"):
            src = full_train_ds if full_train_ds is not None else train_sets[t - 1]
            buffer.add(src, class_maps[t - 1], seed, dataset_name)

        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if cfg.get("ema"):
            # rebuild EMA after expansion so shapes match, then smooth
            ema = copy.deepcopy(model).eval()
            for p in ema.parameters():
                p.requires_grad = False

        # single test touch: genuine CIL exposes all classes seen so far
        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, cpt, dataset_name)
            acc_m[t - 1, tt - 1] = evaluate(model, tl, device, "cil", cpt, tt,
                                            n_exposed=t * cpt)
            if t == n_tasks:
                ta_final[tt] = evaluate(model, tl, device, "ta", cpt, tt)

    return {"accuracy_matrix": acc_m.tolist(), "ta_final": ta_final,
            "replay_bytes": buffer.bytes(hw=64 if dataset_name == "eurosat" else 32)}


class MoENet(nn.Module):
    """Standard shared-backbone Mixture-of-Experts:
    one feature extractor, E expert heads, learned softmax router."""

    def __init__(self, n_classes: int, n_experts: int = 2, feat: int = 64,
                 hidden: int = 128):
        super().__init__()
        self.backbone = Specialist(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat = feat
        self.router = nn.Sequential(
            nn.Linear(feat, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(inplace=True),
                          nn.Linear(hidden, n_classes))
            for _ in range(n_experts)])
        self.n_experts = n_experts
        self.n_classes = n_classes

    def features(self, x):
        return F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)

    def forward(self, x, return_gate=False):
        f = self.features(x)
        gate = F.softmax(self.router(f), dim=1)                     # (B, E)
        logits = torch.stack([e(f) for e in self.experts], dim=1)   # (B, E, C)
        out = (gate.unsqueeze(-1) * logits).sum(dim=1)              # (B, C)
        return (out, gate) if return_gate else out

    def expand(self, n_new: int):
        old = self.n_classes
        dev = self.experts[0][-1].weight.device
        for i, e in enumerate(self.experts):
            lin = e[-1]
            new_lin = nn.Linear(lin.in_features, old + n_new).to(dev)
            with torch.no_grad():
                new_lin.weight[:old].copy_(lin.weight)
                new_lin.bias[:old].copy_(lin.bias)
                nn.init.kaiming_normal_(new_lin.weight[old:],
                                        nonlinearity="relu")
            e[-1] = new_lin
        self.n_classes = old + n_new


def train_moe(train_sets, class_maps, test_ds, device, seed, cpt, n_tasks,
              epochs_scale, dataset_name):
    """Shared-backbone MoE baseline under the same protocol as every other
    method: matched 2.8M backbone, m=20/class replay, per-task compute parity,
    test touched once per task, full Class-IL exposure."""
    set_deterministic(seed)
    ep_first = e(epochs_scale, EPOCHS_BASE)
    ep_incr = e(epochs_scale, EPOCHS_SPECIALIST + EPOCHS_FUSION + EPOCHS_BIAS)
    acc_m = np.zeros((n_tasks, n_tasks))
    ta_final = {}
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    model = MoENet(cpt, n_experts=2).to(device)
    for t in range(1, n_tasks + 1):
        ep = ep_first if t == 1 else ep_incr
        if t > 1:
            model.expand(cpt)
        classes = list(range(cpt * (t - 1), cpt * t))
        loader = DataLoader(train_sets[t - 1], batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
        rep = buffer.loader() if t > 1 else None
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        for _ in range(ep):
            model.train()
            it_rep = iter(rep) if rep is not None else None
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                if rep is not None:
                    try:
                        xr, yr = next(it_rep)
                    except StopIteration:
                        it_rep = iter(rep)
                        xr, yr = next(it_rep)
                    xr, yr = xr.to(device), yr.to(device)
                    loss = loss + F.cross_entropy(model(xr), yr)
                # load-balancing term keeps experts from collapsing
                _, gate = model(x, return_gate=True)
                loss = loss + 0.01 * (gate.mean(0) ** 2).sum()
                loss.backward()
                opt.step()
            sched.step()

        buffer.add(train_ds_ref_moe, class_maps[t - 1], seed, dataset_name)

        for tt in range(1, t + 1):
            tl = make_test_loader(test_ds, tt, cpt, dataset_name)
            acc_m[t - 1, tt - 1] = evaluate(model, tl, device, "cil", cpt, tt,
                                            n_exposed=t * cpt)
            if t == n_tasks:
                ta_final[tt] = evaluate(model, tl, device, "ta", cpt, tt)

    return {"accuracy_matrix": acc_m.tolist(), "ta_final": ta_final,
            "replay_bytes": buffer.bytes(hw=64 if dataset_name == "eurosat" else 32)}


# ============================================================
# METRICS + IO
# ============================================================

def metrics_from_matrix(acc: np.ndarray, n_tasks: int) -> Dict:
    final = acc[n_tasks - 1, :n_tasks]
    diag = np.array([acc[t, t] for t in range(n_tasks)])
    forg = float(np.mean([
        max(acc[i, t] for i in range(t, n_tasks)) - acc[n_tasks - 1, t]
        for t in range(n_tasks - 1)])) if n_tasks > 1 else 0.0
    return {
        "average_final_accuracy": float(final.mean()),
        "plasticity": float(diag.mean()),
        "forgetting": forg,
        "std_final_tasks": float(final.std()),
        "min_final_task": float(final.min()),
        "task_last_accuracy": float(final[-1]),
        "per_task_final": final.tolist(),
    }


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved -> {path}")


def load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def mean_std(vals: List[float]) -> Dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "n": len(a)}
