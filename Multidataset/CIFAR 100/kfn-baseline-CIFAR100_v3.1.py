import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import warnings
import json
from datetime import datetime
import copy

warnings.filterwarnings("ignore")
os.makedirs("./outputs", exist_ok=True)

# =========================
# CONFIG (FINAL FULL RUN)
# =========================
QUICK_RUN = False
SAVE_PATH = "./outputs/kfn_progress_final.json"

if QUICK_RUN:
    EPOCHS_A = 12
    EPOCHS_B = 30
    EPOCHS_BIAS = 8
    SEEDS = [0]
else:
    EPOCHS_A = 50
    EPOCHS_B = 150
    EPOCHS_BIAS = 15
    SEEDS = [0, 1, 2, 3, 4]

BS = 128
NW = 2
REPLAY_PER_CLASS = 1500
EVAL_TEMP = 1.2

SUPCON_W_A = 0.15
SUPCON_W_B = 0.18

# =========================
# UTILS
# =========================
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_subset(dataset, classes):
    indices = [i for i, label in enumerate(dataset.targets) if label in classes]
    return Subset(dataset, indices)

def save_progress(obj, path=SAVE_PATH):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def build_class_index_map(dataset):
    class_to_indices = {}
    for i, y in enumerate(dataset.targets):
        class_to_indices.setdefault(y, []).append(i)
    return class_to_indices

def sample_replay_indices(class_to_indices, classes, per_class, rng):
    indices = []
    for c in classes:
        pool = class_to_indices[c]
        if len(pool) <= per_class:
            indices.extend(pool)
        else:
            indices.extend(rng.sample(pool, per_class))
    rng.shuffle(indices)
    return indices

def make_dynamic_replay_loader(train_ds, class_to_indices, per_class, bs, nw, rng):
    indices = sample_replay_indices(class_to_indices, range(50), per_class, rng)
    subset = Subset(train_ds, indices)
    return DataLoader(subset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)

# =========================
# SupCon Loss
# =========================
def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    B = features.shape[0]

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mask_sum = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)

    loss = -mean_log_prob_pos
    valid = (mask_sum > 0).float()
    loss = (loss * valid).sum() / (valid.sum() + 1e-12)
    return loss

# =========================
# MODEL
# =========================
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=40.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        return self.sigma * F.linear(
            F.normalize(x, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)

class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class StructuralFusionModule(nn.Module):
    def __init__(self, in_channels, old_dim, new_dim, novelty_score):
        super().__init__()
        self.novelty_score = novelty_score
        self.has_expansion = new_dim > 0

        self.proj_reuse = nn.Sequential(
            nn.Conv2d(in_channels, old_dim, 1, bias=False),
            nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(old_dim), nn.ReLU(inplace=True),
        )
        self.gate_reuse = nn.Parameter(torch.tensor([0.0]))

        if self.has_expansion:
            self.proj_expand = nn.Sequential(
                nn.Conv2d(in_channels, new_dim, 1, bias=False),
                nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(new_dim), nn.ReLU(inplace=True),
            )

    def forward(self, x, old_memory_detached):
        reuse_scale = max(0.1, 0.5 - self.novelty_score * 0.3)
        delta_old = self.proj_reuse(x) * torch.sigmoid(self.gate_reuse) * reuse_scale

        feat_new = None
        if self.has_expansion:
            raw_new = self.proj_expand(x)
            feat_new = raw_new - raw_new.mean(dim=(2, 3), keepdim=True)

        return delta_old, feat_new

class GlobalModel(nn.Module):
    def __init__(self, n_specs, ch, n_classes, old_dim=256, new_dim=0, novelty_score=0.0, use_weight_align=True):
        super().__init__()
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.use_weight_align = use_weight_align

        self.old_proj = nn.ModuleList([nn.Conv2d(ch, ch, 1)])
        self.old_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1), nn.Sigmoid()
        )
        self.old_bottleneck = nn.Sequential(nn.Conv2d(ch, old_dim, 1), nn.ReLU(inplace=True))
        self.fusion = StructuralFusionModule(ch, old_dim, new_dim, novelty_score) if new_dim > 0 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = CosineLinear(old_dim + new_dim, n_classes, sigma=40.0)

    def forward_old(self, feats):
        p = self.old_proj[0](feats[0])
        z = p * self.old_gate(p)
        return self.old_bottleneck(z)

    def forward(self, feats):
        old_memory = self.forward_old(feats)

        if self.fusion is not None:
            spec_new = feats[-1]
            delta_old, feat_new = self.fusion(spec_new, old_memory.detach())
            enhanced_old = old_memory + delta_old
            final_features = torch.cat([enhanced_old, feat_new], dim=1) if feat_new is not None else enhanced_old
        else:
            final_features = old_memory

        W_old = self.classifier.weight[:50, :self.old_dim]
        W_new = self.classifier.weight[50:, :]

        flat_old = self.pool(old_memory).flatten(1)
        flat_old = F.normalize(flat_old, dim=1)
        logits_old = F.linear(flat_old, F.normalize(W_old, p=2, dim=1)) * self.classifier.sigma

        flat_new = self.pool(final_features).flatten(1)
        flat_new = F.normalize(flat_new, dim=1)
        logits_new = F.linear(flat_new, F.normalize(W_new, p=2, dim=1)) * self.classifier.sigma

        logits = torch.cat([logits_old, logits_new], dim=1)

        # SAFE WA ONLY
        if self.use_weight_align and logits.size(1) >= 100:
            with torch.no_grad():
                w_old = self.classifier.weight[:50]
                w_new = self.classifier.weight[50:]
                norm_old = w_old.norm(dim=1).mean()
                norm_new = w_new.norm(dim=1).mean()
                gamma = norm_old / (norm_new + 1e-8)
            logits[:, 50:] *= gamma

        return logits, old_memory

class KFN(nn.Module):
    def __init__(self, n_classes=50, n_specs=1, ch=512, old_dim=256, new_dim=0, novelty_score=0.0):
        super().__init__()
        self.specialists = nn.ModuleList([Specialist() for _ in range(n_specs)])
        self.global_model = GlobalModel(n_specs, ch, n_classes, old_dim, new_dim, novelty_score, use_weight_align=True)

    def forward(self, x):
        return self.global_model([s(x) for s in self.specialists])

# =========================
# TRAIN / EVAL
# =========================
def expand_kfn(old_model, trained_spec, new_dim, novelty_score):
    new_kfn = KFN(
        n_classes=100,
        n_specs=2,
        ch=512,
        old_dim=256,
        new_dim=new_dim,
        novelty_score=novelty_score
    ).to(get_device())

    new_kfn.specialists[0].load_state_dict(old_model.specialists[0].state_dict())
    new_kfn.specialists[1].load_state_dict(trained_spec.state_dict())

    old_g = old_model.global_model
    new_g = new_kfn.global_model
    new_g.old_proj.load_state_dict(old_g.old_proj.state_dict())
    new_g.old_gate.load_state_dict(old_g.old_gate.state_dict())
    new_g.old_bottleneck.load_state_dict(old_g.old_bottleneck.state_dict())

    with torch.no_grad():
        new_g.classifier.weight[:50, :new_g.old_dim] = old_g.classifier.weight[:, :old_g.old_dim]
        new_g.classifier.sigma.data = old_g.classifier.sigma.data.clone()
        nn.init.kaiming_normal_(new_g.classifier.weight[50:, :], nonlinearity="relu")
        if new_g.new_dim > 0:
            new_g.classifier.weight[:50, new_g.old_dim:].zero_()

    return new_kfn

def compute_novelty(model, specialist, loader, device):
    model.eval()
    specialist.eval()
    scores = []

    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    proj_layer = nn.Conv2d(512, model.global_model.old_dim, 1, bias=False).to(device)
    with torch.no_grad():
        w = torch.empty_like(proj_layer.weight)
        w.normal_(generator=gen)
        proj_layer.weight.copy_(w)
    proj_layer.eval()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            f_new = specialist(x)
            f_new = F.normalize(proj_layer(F.adaptive_avg_pool2d(f_new, 1)).flatten(1), p=2, dim=1)

            f_old = model.specialists[0](x)
            f_old = model.global_model.forward_old([f_old])
            f_old = F.normalize(F.adaptive_avg_pool2d(f_old, 1).flatten(1), p=2, dim=1)

            proj = torch.sum(f_new * f_old, dim=1, keepdim=True) * f_old
            residual = f_new - proj
            scores.append(torch.norm(residual, p=2, dim=1).mean().item())

    avg = float(np.mean(scores))
    new_dim = 2048 if avg > 0.2 else 1536
    return new_dim, avg

def evaluate(model, loader, device, mode="cil", temp=1.2):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            logits = logits / temp

            if mode == "task_A":
                pred = logits[:, :50].argmax(1)
            elif mode == "task_B":
                pred = logits[:, 50:].argmax(1) + 50
            else:
                pred = logits.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

def run_kfn_cifar100(seed, loaders, epochs_A, epochs_B, epochs_bias, device, train_ds, class_to_indices):
    set_deterministic(seed)
    train_A, train_B, test_A, test_B = loaders
    rng = random.Random(seed + 999)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # -------- Phase 1
    model = KFN(n_classes=50, n_specs=1, ch=512, old_dim=256).to(device)
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_A)

    for _ in range(epochs_A):
        model.train()
        for x, y in train_A:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad()

            logits, _ = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)

            feat_map = model.specialists[0](x)
            feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
            feat = F.normalize(feat, dim=1)
            supcon = supervised_contrastive_loss(feat, y)

            loss = ce + SUPCON_W_A * supcon

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt1)
                scaler.update()
            else:
                loss.backward()
                opt1.step()

        sched1.step()

    acc_A_init = evaluate(model, test_A, device, mode="task_A", temp=EVAL_TEMP)

    old_model_frozen = copy.deepcopy(model).to(device)
    old_model_frozen.eval()
    for p in old_model_frozen.parameters():
        p.requires_grad = False

    # -------- Phase 2
    spec = Specialist().to(device)
    head = nn.Linear(512, 50).to(device)
    epochs_spec = int(0.8 * epochs_B)

    opt2 = torch.optim.AdamW(list(spec.parameters()) + list(head.parameters()), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_spec)

    for ep in range(epochs_spec):
        spec.train()
        for x, y in train_B:
            x, y_raw = x.to(device), y.to(device)
            y = y_raw - 50
            opt2.zero_grad()

            feat_map = spec(x)
            feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
            feat = F.normalize(feat, dim=1)

            logits = head(feat)
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)

            if ep < epochs_spec // 2:
                loss = ce
            else:
                supcon = supervised_contrastive_loss(feat, y)
                loss = ce + SUPCON_W_B * supcon

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt2)
                scaler.update()
            else:
                loss.backward()
                opt2.step()

        sched2.step()

    # -------- Expansion
    new_dim, novelty = compute_novelty(model, spec, test_B, device)
    model = expand_kfn(model, spec, new_dim, novelty)

    # -------- Phase 3
    for _, p in model.named_parameters():
        p.requires_grad = False

    model.specialists[0].eval()
    model.specialists[1].eval()

    for p in model.global_model.fusion.parameters():
        p.requires_grad = True
    for p in model.global_model.classifier.parameters():
        p.requires_grad = True
    for p in model.global_model.old_gate.parameters():
        p.requires_grad = True
    for p in model.global_model.old_bottleneck.parameters():
        p.requires_grad = True

    opt3 = torch.optim.AdamW([
        {"params": model.global_model.fusion.parameters(), "lr": 2e-3},
        {"params": model.global_model.old_gate.parameters(), "lr": 5e-5},
        {"params": model.global_model.old_bottleneck.parameters(), "lr": 5e-5},
        {"params": model.global_model.classifier.parameters(), "lr": 3e-3},
    ], weight_decay=1e-4)
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=epochs_B)

    for _ in range(epochs_B):
        model.train()
        model.specialists[0].eval()
        model.specialists[1].eval()

        replay_loader_epoch = make_dynamic_replay_loader(train_ds, class_to_indices, REPLAY_PER_CLASS, BS, NW, rng)
        replay_iter = iter(replay_loader_epoch)

        for x_B, y_B in train_B:
            x_B, y_B = x_B.to(device), y_B.to(device)

            try:
                x_A, y_A = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader_epoch)
                x_A, y_A = next(replay_iter)

            x_A, y_A = x_A.to(device), y_A.to(device)
            opt3.zero_grad()

            if rng.random() < 0.5:
                x_mix = torch.cat([x_A, x_B], dim=0)
                logits_mix, _ = model(x_mix)
                logits_A = logits_mix[:x_A.size(0)]
                logits_B = logits_mix[x_A.size(0):]
            else:
                logits_A, _ = model(x_A)
                logits_B, _ = model(x_B)

            logits_B_scaled = logits_B.clone()
            logits_B_scaled[:, 50:] *= 1.3

            loss_B = F.cross_entropy(logits_B_scaled[:, 50:], y_B - 50, label_smoothing=0.1)
            loss_A = F.cross_entropy(logits_A[:, :50], y_A, label_smoothing=0.1)

            with torch.no_grad():
                old_spec = old_model_frozen.specialists[0](x_A)
                old_feat = old_model_frozen.global_model.forward_old([old_spec])

            new_spec = model.specialists[0](x_A)
            new_feat = model.global_model.forward_old([new_spec])

            loss_distill = F.mse_loss(new_feat, old_feat)

            loss = 2.5 * loss_B + 1.0 * loss_A + 0.25 * loss_distill

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt3)
                scaler.update()
            else:
                loss.backward()
                opt3.step()

        sched3.step()

    # -------- Phase 4
    for p in model.parameters():
        p.requires_grad = False
    for p in model.global_model.classifier.parameters():
        p.requires_grad = True

    opt4 = torch.optim.AdamW(model.global_model.classifier.parameters(), lr=5e-5, weight_decay=1e-4)
    sched4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=epochs_bias)

    for _ in range(epochs_bias):
        model.train()
        replay_loader_epoch = make_dynamic_replay_loader(train_ds, class_to_indices, REPLAY_PER_CLASS, BS, NW, rng)
        replay_iter = iter(replay_loader_epoch)

        for x_B, y_B in train_B:
            x_B, y_B = x_B.to(device), y_B.to(device)

            try:
                x_A, y_A = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader_epoch)
                x_A, y_A = next(replay_iter)

            x_A, y_A = x_A.to(device), y_A.to(device)
            x = torch.cat([x_A, x_B], dim=0)
            y = torch.cat([y_A, y_B], dim=0)

            opt4.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt4)
                scaler.update()
            else:
                loss.backward()
                opt4.step()

        sched4.step()

    acc_A_final_taskaware = evaluate(model, test_A, device, mode="task_A", temp=EVAL_TEMP)
    acc_B_final_taskaware = evaluate(model, test_B, device, mode="task_B", temp=EVAL_TEMP)
    acc_A_final_cil = evaluate(model, test_A, device, mode="cil", temp=EVAL_TEMP)
    acc_B_final_cil = evaluate(model, test_B, device, mode="cil", temp=EVAL_TEMP)

    return {
        "acc_A_init": acc_A_init,
        "acc_A_final_taskaware": acc_A_final_taskaware,
        "acc_B_final_taskaware": acc_B_final_taskaware,
        "acc_A_final_cil": acc_A_final_cil,
        "acc_B_final_cil": acc_B_final_cil,
        "retention_taskaware": (acc_A_final_taskaware / acc_A_init) * 100 if acc_A_init > 0 else 0.0,
        "retention_cil": (acc_A_final_cil / acc_A_init) * 100 if acc_A_init > 0 else 0.0,
        "forgetting_taskaware": acc_A_init - acc_A_final_taskaware,
        "forgetting_cil": acc_A_init - acc_A_final_cil,
        "bwt_taskaware": acc_A_final_taskaware - acc_A_init,
        "bwt_cil": acc_A_final_cil - acc_A_init
    }

# =========================
# RUN
# =========================
def run_all():
    device = get_device()
    print(f"🚀 Environment: {device} | QUICK_RUN={QUICK_RUN}")

    stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    root = "/kaggle/input/cifar-100" if os.path.exists("/kaggle/input/cifar-100") else "./data"
    train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=t_train)
    test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=t_test)
    class_to_indices = build_class_index_map(train_ds)

    loaders = (
        DataLoader(get_class_subset(train_ds, range(50)), batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True, persistent_workers=True),
        DataLoader(get_class_subset(train_ds, range(50, 100)), batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True, persistent_workers=True),
        DataLoader(get_class_subset(test_ds, range(50)), batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True),
        DataLoader(get_class_subset(test_ds, range(50, 100)), batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True),
    )

    results = []
    meta = {
        "timestamp": str(datetime.utcnow()),
        "quick_run": QUICK_RUN,
        "epochs_A": EPOCHS_A,
        "epochs_B": EPOCHS_B,
        "epochs_bias": EPOCHS_BIAS,
        "replay_per_class": REPLAY_PER_CLASS,
        "eval_temp": EVAL_TEMP,
        "seeds": SEEDS,
        "results": []
    }

    print("\n[1] Running experiments...")
    try:
        for s in SEEDS:
            print(f" -> Seed {s}...")
            r = run_kfn_cifar100(
                s, loaders, EPOCHS_A, EPOCHS_B, EPOCHS_BIAS,
                device, train_ds, class_to_indices
            )
            results.append(r)
            meta["results"].append({"seed": s, **r})
            save_progress(meta)

            print(
                f"    ↳ Seed {s}: "
                f"A_init(TA)={r['acc_A_init']:.2f} | "
                f"A_final(TA)={r['acc_A_final_taskaware']:.2f} | "
                f"B_final(TA)={r['acc_B_final_taskaware']:.2f} | "
                f"A_final(CIL)={r['acc_A_final_cil']:.2f} | "
                f"B_final(CIL)={r['acc_B_final_cil']:.2f}"
            )

    except KeyboardInterrupt:
        print("\n⛔ Interrupted. Partial results saved to:", SAVE_PATH)
        save_progress(meta)
        return

    if results:
        print("\n" + "=" * 90)
        print("SUMMARY (Mean ± Std)")
        print("=" * 90)

        for k, label in [
            ("acc_A_init", "A Init (Task-aware)"),
            ("acc_A_final_taskaware", "A Final (Task-aware)"),
            ("acc_B_final_taskaware", "B Final (Task-aware)"),
            ("acc_A_final_cil", "A Final (CIL)"),
            ("acc_B_final_cil", "B Final (CIL)")
        ]:
            vals = [r[k] for r in results]
            print(f"{label:26}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

        for k, label in [
            ("retention_taskaware", "Retention (Task-aware)"),
            ("retention_cil", "Retention (CIL)"),
            ("forgetting_taskaware", "Forgetting (Task-aware)"),
            ("forgetting_cil", "Forgetting (CIL)"),
            ("bwt_taskaware", "BWT (Task-aware)"),
            ("bwt_cil", "BWT (CIL)")
        ]:
            vals = [r[k] for r in results]
            print(f"{label:26}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

        print("=" * 90)
        print(f"Saved run log to: {SAVE_PATH}")

if __name__ == "__main__":
    run_all()





''' Results
ubuntu@gt-ubuntu24-04-cmd-v3-2-120gb-100m:~$ python -u KFN.py
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
ubuntu@gt-ubuntu24-04-cmd-v3-2-120gb-100m:~$ python3 KFN.py
🚀 Environment: cuda | QUICK_RUN=False
100.0%

[1] Running experiments...
 -> Seed 0...


    ↳ Seed 0: A_init(TA)=77.08 | A_final(TA)=76.76 | B_final(TA)=76.72 | A_final(CIL)=65.76 | B_final(CIL)=63.72
 -> Seed 1...
    ↳ Seed 1: A_init(TA)=78.24 | A_final(TA)=77.82 | B_final(TA)=76.80 | A_final(CIL)=66.80 | B_final(CIL)=64.26
 -> Seed 2...

    ↳ Seed 2: A_init(TA)=77.06 | A_final(TA)=76.82 | B_final(TA)=76.62 | A_final(CIL)=67.32 | B_final(CIL)=63.54
 -> Seed 3...

'''