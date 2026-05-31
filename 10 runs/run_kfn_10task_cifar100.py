import os
import json
import copy
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# =========================
# CONFIG (STRICT MODE)
# =========================

QUICK_RUN = False
SAVE_PATH = "./outputs/kfn_split_cifar100_10task_final.json"

REPLAY_BUFFER_CAPACITY = 20
N_TASKS = 10
CLASSES_PER_TASK = 10
TOTAL_CLASSES = N_TASKS * CLASSES_PER_TASK

EPOCHS_BASE = 50 if not QUICK_RUN else 5
EPOCHS_SPECIALIST = 30 if not QUICK_RUN else 3
EPOCHS_FUSION = 50 if not QUICK_RUN else 5
EPOCHS_BIAS = 10 if not QUICK_RUN else 2

BATCH_SIZE = 128
NOVELTY_THRESHOLD = 0.25
KD_TEMPERATURE = 2.0
USE_AMP = True
FORCE_CPU = False


# =========================
# UTILS
# =========================

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def save_progress(obj: Dict, path: str = SAVE_PATH) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =========================
# DATASET & DATALOADER SETUP
# =========================

class CIFAR100TaskDataset(Dataset):
    def __init__(self, dataset: Dataset, classes: List[int]):
        self.dataset = dataset
        self.indices = [i for i, label in enumerate(dataset.targets) if label in classes]
        self.classes = classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[self.indices[idx]]
        return img, label


def get_task_loaders(
    train_ds: Dataset,
    test_ds: Dataset,
    task_id: int,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    start_class = (task_id - 1) * CLASSES_PER_TASK
    end_class = task_id * CLASSES_PER_TASK
    task_classes = list(range(start_class, end_class))

    train_subset = CIFAR100TaskDataset(train_ds, task_classes)
    test_subset = CIFAR100TaskDataset(test_ds, task_classes)

    pin_memory = torch.cuda.is_available() and not FORCE_CPU

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


# =========================
# REPLAY BUFFER (STRICT 20 EXEMPLARS/CLASS)
# =========================

class DynamicReplayBuffer:
    def __init__(self, capacity_per_class: int = REPLAY_BUFFER_CAPACITY):
        self.capacity_per_class = capacity_per_class
        self.buffer: Dict[int, List[torch.Tensor]] = {}
        self.labels: Dict[int, List[int]] = {}

    def add_exemplars(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        for img, label in zip(images, labels):
            label = label.item()
            if label not in self.buffer:
                self.buffer[label] = []
                self.labels[label] = []
            if len(self.buffer[label]) < self.capacity_per_class:
                self.buffer[label].append(img.cpu())
                self.labels[label].append(label)

    def get_loader(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = 2,
        shuffle: bool = True
    ) -> Optional[DataLoader]:
        if not self.buffer:
            return None

        buffer_images = []
        buffer_labels = []

        for label in sorted(self.buffer.keys()):
            buffer_images.extend(self.buffer[label])
            buffer_labels.extend(self.labels[label])

        buffer_images = torch.stack(buffer_images)
        buffer_labels = torch.tensor(buffer_labels, dtype=torch.long)

        class BufferDataset(Dataset):
            def __len__(self) -> int:
                return len(buffer_labels)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                return buffer_images[idx], buffer_labels[idx]

        pin_memory = torch.cuda.is_available() and not FORCE_CPU

        return DataLoader(
            BufferDataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self.buffer.values())


# =========================
# MODEL ARCHITECTURE
# =========================

class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma: float = 40.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigma * F.linear(
            F.normalize(x, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class StructuralFusionModule(nn.Module):
    def __init__(self, in_channels: int, old_dim: int, new_dim: int, novelty_score: float):
        super().__init__()
        self.novelty_score = novelty_score
        self.has_expansion = new_dim > 0

        self.proj_reuse = nn.Sequential(
            nn.Conv2d(in_channels, old_dim, 1, bias=False),
            nn.BatchNorm2d(old_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(old_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(old_dim, old_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(old_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_reuse = nn.Parameter(torch.tensor([0.0]))

        if self.has_expansion:
            self.proj_expand = nn.Sequential(
                nn.Conv2d(in_channels, new_dim, 1, bias=False),
                nn.BatchNorm2d(new_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(new_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(new_dim, new_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(new_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor, old_memory_detached: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        reuse_scale = max(0.1, 0.5 - self.novelty_score * 0.3)
        delta_old = self.proj_reuse(x) * torch.sigmoid(self.gate_reuse) * reuse_scale

        feat_new = None
        if self.has_expansion:
            raw_new = self.proj_expand(x)
            feat_new = raw_new - raw_new.mean(dim=(2, 3), keepdim=True)

        return delta_old, feat_new


class GlobalModel(nn.Module):
    def __init__(
        self,
        n_specs: int,
        ch: int,
        n_classes: int,
        old_dim: int = 256,
        new_dim: int = 0,
        novelty_score: float = 0.0,
        use_weight_align: bool = True
    ):
        super().__init__()
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.use_weight_align = use_weight_align

        self.old_proj = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_specs)])
        self.old_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid()
        )
        self.old_bottleneck = nn.Sequential(
            nn.Conv2d(ch, old_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.fusion = StructuralFusionModule(ch, old_dim, new_dim, novelty_score) if new_dim > 0 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = CosineLinear(old_dim + new_dim, n_classes, sigma=40.0)

    def forward_old(self, feats: List[torch.Tensor]) -> torch.Tensor:
        p = self.old_proj[0](feats[0])
        z = p * self.old_gate(p)
        return self.old_bottleneck(z)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        old_memory = self.forward_old(feats)

        if self.fusion is not None:
            spec_new = feats[-1]
            delta_old, feat_new = self.fusion(spec_new, old_memory.detach())
            enhanced_old = old_memory + delta_old
            final_features = torch.cat([enhanced_old, feat_new], dim=1) if feat_new is not None else enhanced_old
        else:
            final_features = old_memory

        flat_feat = F.normalize(self.pool(final_features).flatten(1), p=2, dim=1)
        logits = self.classifier(flat_feat)

        if self.use_weight_align and logits.size(1) > CLASSES_PER_TASK:
            with torch.no_grad():
                w_old = self.classifier.weight[:-CLASSES_PER_TASK]
                w_new = self.classifier.weight[-CLASSES_PER_TASK:]
                gamma = w_old.norm(dim=1).mean() / (w_new.norm(dim=1).mean() + 1e-8)
            logits[:, -CLASSES_PER_TASK:] *= gamma

        return logits, old_memory


class KFN(nn.Module):
    def __init__(
        self,
        n_classes: int = 100,
        n_specs: int = 1,
        ch: int = 512,
        old_dim: int = 256,
        new_dim: int = 0,
        novelty_score: float = 0.0
    ):
        super().__init__()
        self.specialists = nn.ModuleList([Specialist() for _ in range(n_specs)])
        self.global_model = GlobalModel(
            n_specs=n_specs,
            ch=ch,
            n_classes=n_classes,
            old_dim=old_dim,
            new_dim=new_dim,
            novelty_score=novelty_score,
            use_weight_align=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.global_model([s(x) for s in self.specialists])


# =========================
# TRAINING & EVALUATION
# =========================

def expand_kfn(
    old_model: KFN,
    trained_spec: Specialist,
    new_dim: int,
    novelty_score: float,
    n_classes: int
) -> KFN:
    device = old_model.global_model.classifier.weight.device

    new_kfn = KFN(
        n_classes=n_classes,
        n_specs=len(old_model.specialists) + 1,
        ch=512,
        old_dim=old_model.global_model.old_dim,
        new_dim=new_dim,
        novelty_score=novelty_score
    ).to(device)

    for i, spec in enumerate(old_model.specialists):
        new_kfn.specialists[i].load_state_dict(spec.state_dict())
    new_kfn.specialists[-1].load_state_dict(trained_spec.state_dict())

    new_g = new_kfn.global_model
    old_g = old_model.global_model

    new_g.old_proj[0].load_state_dict(old_g.old_proj[0].state_dict())
    new_g.old_gate.load_state_dict(old_g.old_gate.state_dict())
    new_g.old_bottleneck.load_state_dict(old_g.old_bottleneck.state_dict())

    with torch.no_grad():
        old_out, old_in = old_g.classifier.weight.shape
        new_out, new_in = new_g.classifier.weight.shape

        new_g.classifier.weight[:old_out, :old_in].copy_(old_g.classifier.weight)
        new_g.classifier.sigma.data.copy_(old_g.classifier.sigma.data)

        if new_out > old_out:
            nn.init.kaiming_normal_(new_g.classifier.weight[old_out:, :], nonlinearity="relu")

        if new_in > old_in:
            new_g.classifier.weight[:old_out, old_in:].zero_()

    return new_kfn


def compute_novelty(
    model: KFN,
    specialist: Specialist,
    loader: DataLoader,
    device: torch.device
) -> Tuple[int, float]:
    model.eval()
    specialist.eval()
    scores = []

    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    proj_layer = nn.Conv2d(512, model.global_model.old_dim, 1, bias=False).to(device)

    with torch.no_grad():
        w = torch.empty_like(proj_layer.weight, device="cpu")
        w.normal_(generator=gen)
        proj_layer.weight.copy_(w.to(device))

    proj_layer.eval()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            f_new = specialist(x)
            f_new = F.normalize(
                proj_layer(F.adaptive_avg_pool2d(f_new, 1)).flatten(1),
                p=2,
                dim=1
            )

            f_old = model.specialists[0](x)
            f_old = model.global_model.forward_old([f_old])
            f_old = F.normalize(
                F.adaptive_avg_pool2d(f_old, 1).flatten(1),
                p=2,
                dim=1
            )

            proj = torch.sum(f_new * f_old, dim=1, keepdim=True) * f_old
            residual = f_new - proj
            scores.append(torch.norm(residual, p=2, dim=1).mean().item())

    avg_score = float(np.mean(scores))
    new_dim = 48 if avg_score > NOVELTY_THRESHOLD else 4
    return new_dim, avg_score


def evaluate(
    model: KFN,
    loader: DataLoader,
    device: torch.device,
    task_id: Optional[int] = None,
    n_tasks: int = N_TASKS
) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# =========================
# MAIN TRAINING LOOP
# =========================

def train_kfn_split_cifar100(
    seed: int = 0,
    quick_run: bool = QUICK_RUN,
    save_path: str = SAVE_PATH
) -> Dict:
    set_deterministic(seed)
    device = get_device(FORCE_CPU)

    print(f"🚀 Training KFN on Split CIFAR-100 (10 tasks) | Seed: {seed} | Device: {device}")

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

    root = "./data"
    train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=t_train)
    test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=t_test)

    replay_buffer = DynamicReplayBuffer(REPLAY_BUFFER_CAPACITY)

    accuracy_matrix = np.zeros((N_TASKS, N_TASKS))
    param_counts = []
    results = {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "average_accuracy": [],
        "param_counts": param_counts,
        "tasks": []
    }

    print("\n🔹 Task 1: Base Initialization (Classes 0-9)")
    train_loader, test_loader = get_task_loaders(train_ds, test_ds, task_id=1)

    model = KFN(n_classes=CLASSES_PER_TASK, n_specs=1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_BASE)

    amp_ok = USE_AMP and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_ok else None

    for epoch in range(EPOCHS_BASE):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            if amp_ok:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

        sched.step()

    acc = evaluate(model, test_loader, device)
    accuracy_matrix[0, 0] = acc
    print(f"✅ Task 1 Accuracy: {acc:.2f}%")

    for x, y in train_loader:
        replay_buffer.add_exemplars(x, y)
        if len(replay_buffer) >= REPLAY_BUFFER_CAPACITY * CLASSES_PER_TASK:
            break

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    param_counts.append(sum(p.numel() for p in model.parameters()))
    results["average_accuracy"].append(float(acc))
    results["param_counts"] = param_counts
    results["tasks"].append({
        "task_id": 1,
        "classes": list(range(CLASSES_PER_TASK)),
        "novelty_score": 0.0,
        "expansion_dim": 0,
        "param_count": param_counts[-1]
    })
    results["accuracy_matrix"] = accuracy_matrix.tolist()
    save_progress(results, save_path)

    for task_id in range(2, N_TASKS + 1):
        print(f"\n🔹 Task {task_id}: Incremental Learning (Classes {CLASSES_PER_TASK * (task_id - 1)}-{CLASSES_PER_TASK * task_id - 1})")

        current_classes = list(range(CLASSES_PER_TASK * (task_id - 1), CLASSES_PER_TASK * task_id))
        train_loader, test_loader = get_task_loaders(train_ds, test_ds, task_id=task_id)

        print("  🔸 Phase 2: Specialist Training")
        specialist = Specialist().to(device)
        head = nn.Linear(512, CLASSES_PER_TASK).to(device)

        opt_spec = torch.optim.AdamW(list(specialist.parameters()) + list(head.parameters()), lr=1e-3)
        sched_spec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_spec, T_max=EPOCHS_SPECIALIST)

        for epoch in range(EPOCHS_SPECIALIST):
            specialist.train()
            head.train()

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                y_local = y - (task_id - 1) * CLASSES_PER_TASK

                opt_spec.zero_grad()

                if amp_ok:
                    with torch.amp.autocast("cuda"):
                        feat = specialist(x)
                        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                        logits = head(feat)
                        loss = F.cross_entropy(logits, y_local)

                    scaler.scale(loss).backward()
                    scaler.step(opt_spec)
                    scaler.update()
                else:
                    feat = specialist(x)
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    logits = head(feat)
                    loss = F.cross_entropy(logits, y_local)
                    loss.backward()
                    opt_spec.step()

            sched_spec.step()

        print("  🔸 Phase 3A: Novelty Detection & Expansion")
        new_dim, novelty_score = compute_novelty(teacher, specialist, test_loader, device)
        print(f"    Novelty Score: {novelty_score:.4f} | Expansion Dim: {new_dim}")

        print("  🔸 Phase 3B: Structural Fusion")
        model = expand_kfn(
            teacher,
            specialist,
            new_dim,
            novelty_score,
            n_classes=CLASSES_PER_TASK * task_id
        ).to(device)

        for p in model.parameters():
            p.requires_grad = False

        if model.global_model.fusion is not None:
            for p in model.global_model.fusion.parameters():
                p.requires_grad = True

        for p in model.global_model.classifier.parameters():
            p.requires_grad = True
        for p in model.global_model.old_gate.parameters():
            p.requires_grad = True
        for p in model.global_model.old_bottleneck.parameters():
            p.requires_grad = True

        fusion_param_groups = [
            {"params": model.global_model.old_gate.parameters(), "lr": 5e-5},
            {"params": model.global_model.old_bottleneck.parameters(), "lr": 5e-5},
            {"params": model.global_model.classifier.parameters(), "lr": 1e-3},
        ]

        if model.global_model.fusion is not None:
            fusion_param_groups.insert(0, {"params": model.global_model.fusion.parameters(), "lr": 2e-3})

        opt_fusion = torch.optim.AdamW(fusion_param_groups, weight_decay=1e-4)
        sched_fusion = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fusion, T_max=EPOCHS_FUSION)

        replay_loader = replay_buffer.get_loader()
        if replay_loader is None:
            raise RuntimeError("Replay buffer is empty before incremental fusion training.")

        for epoch in range(EPOCHS_FUSION):
            model.train()
            replay_iter = iter(replay_loader)

            for x_curr, y_curr in train_loader:
                x_curr, y_curr = x_curr.to(device), y_curr.to(device)

                try:
                    x_replay, y_replay = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    x_replay, y_replay = next(replay_iter)

                x_replay, y_replay = x_replay.to(device), y_replay.to(device)

                opt_fusion.zero_grad()

                if amp_ok:
                    with torch.amp.autocast("cuda"):
                        logits_curr, _ = model(x_curr)
                        logits_replay, _ = model(x_replay)

                        loss_curr = F.cross_entropy(logits_curr, y_curr)
                        loss_replay = F.cross_entropy(logits_replay, y_replay)

                        with torch.no_grad():
                            teacher_feat = teacher.global_model.forward_old([teacher.specialists[0](x_replay)])

                        student_feat = model.global_model.forward_old([model.specialists[0](x_replay)])
                        loss_distill = F.mse_loss(student_feat, teacher_feat)

                        with torch.no_grad():
                            teacher_logits, _ = teacher(x_replay)

                        student_logits, _ = model(x_replay)
                        common_dim = teacher_logits.size(1)

                        loss_kd = F.kl_div(
                            F.log_softmax(student_logits[:, :common_dim] / KD_TEMPERATURE, dim=1),
                            F.softmax(teacher_logits / KD_TEMPERATURE, dim=1),
                            reduction="batchmean"
                        ) * (KD_TEMPERATURE ** 2)

                        loss = loss_curr + loss_replay + 0.25 * loss_distill + 0.8 * loss_kd

                    scaler.scale(loss).backward()
                    scaler.step(opt_fusion)
                    scaler.update()
                else:
                    logits_curr, _ = model(x_curr)
                    logits_replay, _ = model(x_replay)

                    loss_curr = F.cross_entropy(logits_curr, y_curr)
                    loss_replay = F.cross_entropy(logits_replay, y_replay)

                    with torch.no_grad():
                        teacher_feat = teacher.global_model.forward_old([teacher.specialists[0](x_replay)])

                    student_feat = model.global_model.forward_old([model.specialists[0](x_replay)])
                    loss_distill = F.mse_loss(student_feat, teacher_feat)

                    with torch.no_grad():
                        teacher_logits, _ = teacher(x_replay)

                    student_logits, _ = model(x_replay)
                    common_dim = teacher_logits.size(1)

                    loss_kd = F.kl_div(
                        F.log_softmax(student_logits[:, :common_dim] / KD_TEMPERATURE, dim=1),
                        F.softmax(teacher_logits / KD_TEMPERATURE, dim=1),
                        reduction="batchmean"
                    ) * (KD_TEMPERATURE ** 2)

                    loss = loss_curr + loss_replay + 0.25 * loss_distill + 0.8 * loss_kd
                    loss.backward()
                    opt_fusion.step()

            sched_fusion.step()

        print("  🔸 Phase 4: Bias Correction")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.global_model.classifier.parameters():
            p.requires_grad = True

        opt_bias = torch.optim.AdamW(model.global_model.classifier.parameters(), lr=5e-5)
        sched_bias = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bias, T_max=EPOCHS_BIAS)

        replay_loader = replay_buffer.get_loader()
        if replay_loader is None:
            raise RuntimeError("Replay buffer is empty before bias correction.")

        for epoch in range(EPOCHS_BIAS):
            model.train()

            for x_replay, y_replay in replay_loader:
                x_replay, y_replay = x_replay.to(device), y_replay.to(device)
                opt_bias.zero_grad()

                if amp_ok:
                    with torch.amp.autocast("cuda"):
                        logits, _ = model(x_replay)
                        loss = F.cross_entropy(logits, y_replay)

                    scaler.scale(loss).backward()
                    scaler.step(opt_bias)
                    scaler.update()
                else:
                    logits, _ = model(x_replay)
                    loss = F.cross_entropy(logits, y_replay)
                    loss.backward()
                    opt_bias.step()

            sched_bias.step()

        print("  🔸 Evaluation")
        for t in range(1, task_id + 1):
            _, test_loader_t = get_task_loaders(train_ds, test_ds, task_id=t)
            acc = evaluate(model, test_loader_t, device)
            accuracy_matrix[task_id - 1, t - 1] = acc
            print(f"    Task {t} Accuracy: {acc:.2f}%")

        for x, y in train_loader:
            replay_buffer.add_exemplars(x, y)
            if len(replay_buffer) >= REPLAY_BUFFER_CAPACITY * CLASSES_PER_TASK * task_id:
                break

        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False

        param_counts.append(sum(p.numel() for p in model.parameters()))
        avg_acc = float(np.mean(accuracy_matrix[task_id - 1, :task_id]))

        results["average_accuracy"].append(avg_acc)
        results["accuracy_matrix"] = accuracy_matrix.tolist()
        results["param_counts"] = param_counts
        results["tasks"].append({
            "task_id": task_id,
            "classes": current_classes,
            "novelty_score": novelty_score,
            "expansion_dim": new_dim,
            "param_count": param_counts[-1]
        })

        save_progress(results, save_path)

    print("\n🎉 Training Complete!")
    print(f"📊 Final Accuracy Matrix:\n{np.array(results['accuracy_matrix'])}")
    print(f"📈 Average Accuracy: {results['average_accuracy'][-1]:.2f}%")
    print(f"📏 Parameter Count: {results['param_counts'][-1]:,}")

    return results


# =========================
# RUN EXPERIMENT
# =========================

if __name__ == "__main__":
    train_kfn_split_cifar100(seed=0, quick_run=QUICK_RUN)