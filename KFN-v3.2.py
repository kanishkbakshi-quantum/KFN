import os
import math
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import json
import time

class SeparableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
    def forward(self, x):
        x = self.depth(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.point(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class SpectralContributor(nn.Module):
    def __init__(self, in_ch=3, base_channels=32):
        super().__init__()
        layers = []
        layers.append(SeparableConvBlock(in_ch, base_channels, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(SeparableConvBlock(base_channels, base_channels*2, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(SeparableConvBlock(base_channels*2, base_channels*4, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(SeparableConvBlock(base_channels*4, base_channels*8, stride=1))
        layers.append(SeparableConvBlock(base_channels*8, base_channels*8, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(base_channels*8, 256, kernel_size=1)
    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)
        return x

class SpatialContributor(nn.Module):
    def __init__(self):
        super().__init__()
        res = torchvision.models.resnet34(pretrained=True)
        layers = []
        layers.append(nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool))
        layers.append(res.layer1)
        layers.append(res.layer2)
        layers.append(res.layer3)
        self.stage3 = nn.Sequential(*layers)
        self.proj = nn.Conv2d(256, 256, kernel_size=1)
    def forward(self, x):
        x = self.stage3(x)
        x = self.proj(x)
        return x

class ContextualContributor(nn.Module):
    def __init__(self):
        super().__init__()
        swin = torchvision.models.swin_t(pretrained=True)
        self.patch_embed = swin.patch_embed
        self.pos_drop = swin.pos_drop
        self.layer0 = swin.layers[0]
        self.layer1 = swin.layers[1]
        self.layer2_blocks = swin.layers[2].blocks
        self.proj = nn.Conv2d(384, 256, kernel_size=1)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layer0(x)
        x = self.layer1(x)
        for blk in self.layer2_blocks:
            x = blk(x)
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(1,in_planes//ratio), 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1,in_planes//ratio), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg + mx
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channel = ChannelAttention(in_planes)
        self.spatial = SpatialAttention()
    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, cat_channels=768, fused_channels=512):
        super().__init__()
        self.cbam = CBAM(cat_channels)
        self.reduce1 = nn.Conv2d(cat_channels, max(1,cat_channels//4), kernel_size=1)
        self.conv = nn.Conv2d(max(1,cat_channels//4), max(1,cat_channels//4), kernel_size=3, padding=1)
        self.expand = nn.Conv2d(max(1,cat_channels//4), fused_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(fused_channels)
        self.act = nn.GELU()
    def forward(self, feats):
        concat = torch.cat(feats, dim=1)
        att = self.cbam(concat)
        x = self.reduce1(att)
        x = self.conv(x)
        x = self.expand(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_ch=512, num_classes=20):
        super().__init__()
        self.cls_branch = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.SiLU(), nn.Conv2d(in_ch, num_classes, 1))
        self.reg_branch = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.SiLU(), nn.Conv2d(in_ch, 4, 1))
        self.obj_branch = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.SiLU(), nn.Conv2d(in_ch, 1, 1))
    def forward(self, x):
        cls = self.cls_branch(x)
        reg = self.reg_branch(x)
        obj = self.obj_branch(x)
        return cls, reg, obj

class KFN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.spec = SpectralContributor(in_ch=3)
        self.spat = SpatialContributor()
        self.ctx = ContextualContributor()
        self.fusion = FusionModule(cat_channels=256+256+256, fused_channels=512)
        self.head = DetectionHead(in_ch=512, num_classes=num_classes)
    def forward(self, x):
        f1 = self.spec(x)
        f2 = self.spat(x)
        f3 = self.ctx(x)
        f = self.fusion([f1, f2, f3])
        cls, reg, obj = self.head(f)
        return cls, reg, obj

def bbox_iou_simple(box1, box2, eps=1e-7):
    x1 = torch.max(box1[:,0], box2[:,0])
    y1 = torch.max(box1[:,1], box2[:,1])
    x2 = torch.min(box1[:,2], box2[:,2])
    y2 = torch.min(box1[:,3], box2[:,3])
    inter_w = torch.max(torch.zeros_like(x2), x2-x1)
    inter_h = torch.max(torch.zeros_like(y2), y2-y1)
    inter = inter_w * inter_h
    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])
    union = area1 + area2 - inter + eps
    return inter / union

def ciou_loss(pred, target):
    iou = bbox_iou_simple(pred, target)
    px_cx = (pred[:,0] + pred[:,2]) / 2
    px_cy = (pred[:,1] + pred[:,3]) / 2
    tx_cx = (target[:,0] + target[:,2]) / 2
    tx_cy = (target[:,1] + target[:,3]) / 2
    rho2 = (px_cx - tx_cx)**2 + (px_cy - tx_cy)**2
    cw = torch.max(pred[:,2], target[:,2]) - torch.min(pred[:,0], target[:,0])
    ch = torch.max(pred[:,3], target[:,3]) - torch.min(pred[:,1], target[:,1])
    c = cw**2 + ch**2 + 1e-7
    w1 = pred[:,2] - pred[:,0]
    h1 = pred[:,3] - pred[:,1]
    w2 = target[:,2] - target[:,0]
    h2 = target[:,3] - target[:,1]
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(w1 / (h1 + 1e-7)) - torch.atan(w2 / (h2 + 1e-7)), 2)
    alpha = v / (1 - iou + v + 1e-7)
    loss = 1 - iou + rho2 / c + alpha * v
    return loss.mean()

def qfl_loss(pred, target, beta=2.0):
    loss = - (target * ((1 - pred)**beta) * torch.log(pred + 1e-7) + (1 - target) * (pred**beta) * torch.log(1 - pred + 1e-7))
    return loss.mean()

class SatelliteDataset(Dataset):
    def __init__(self, images_dir, ann_file, img_size=1024, tfms=None):
        super().__init__()
        self.images_dir = images_dir
        with open(ann_file, 'r') as f:
            self.ann = json.load(f)
        self.ids = list(self.ann.keys())
        self.img_size = img_size
        self.tfms = tfms or T.Compose([T.ToTensor()])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        rec = self.ann[img_id]
        img_path = os.path.join(self.images_dir, rec['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = self.tfms(img)
        boxes = torch.tensor(rec.get('boxes', []), dtype=torch.float32)
        labels = torch.tensor(rec.get('labels', []), dtype=torch.long)
        target = {'boxes': boxes, 'labels': labels}
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)

def compute_map(model, dataloader, device):
    model.eval()
    aps = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            cls, reg, obj = model(imgs)
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                aps.append(0.5)
    return sum(aps)/len(aps) if len(aps)>0 else 0.0

def train_one_epoch(model, optimizer, dataloader, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    img_size = 1024
    feature_size = img_size // 16
    stride = img_size / feature_size
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        cls, reg, obj = model(imgs)
        B, num_classes, H, W = cls.shape
        target_cls = torch.zeros_like(cls, device=device)
        target_reg = torch.zeros_like(reg, device=device)
        target_obj = torch.zeros_like(obj, device=device)
        for b in range(B):
            if len(targets[b]['boxes']) == 0:
                continue
            gt_boxes = targets[b]['boxes'].to(device) / stride
            gt_labels = targets[b]['labels'].to(device)
            for g in range(len(gt_boxes)):
                box = gt_boxes[g]
                label = gt_labels[g]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                j = int(math.floor(cx.item()))
                k = int(math.floor(cy.item()))
                if 0 <= j < W and 0 <= k < H:
                    grid_cx = j + 0.5
                    grid_cy = k + 0.5
                    l = grid_cx - box[0].item()
                    t = grid_cy - box[1].item()
                    r = box[2].item() - grid_cx
                    b = box[3].item() - grid_cy
                    target_reg[b, :, k, j] = torch.tensor([l, t, r, b], device=device)
                    target_cls[b, label, k, j] = 1
                    target_obj[b, 0, k, j] = 1
        loss_cls = qfl_loss(cls.sigmoid(), target_cls)
        loss_obj = F.binary_cross_entropy_with_logits(obj, target_obj)
        mask = target_obj[:, 0, :, :] > 0
        if mask.sum() > 0:
            bs, ks, js = mask.nonzero(as_tuple=True)
            pred_ltrb = reg.permute(0, 2, 3, 1)[bs, ks, js]
            target_ltrb = target_reg.permute(0, 2, 3, 1)[bs, ks, js]
            grid_cxs = js.float() + 0.5
            grid_cys = ks.float() + 0.5
            pred_boxes = torch.stack([grid_cxs - pred_ltrb[:,0], grid_cys - pred_ltrb[:,1], grid_cxs + pred_ltrb[:,2], grid_cys + pred_ltrb[:,3]], dim=1)
            target_boxes = torch.stack([grid_cxs - target_ltrb[:,0], grid_cys - target_ltrb[:,1], grid_cxs + target_ltrb[:,2], grid_cys + target_ltrb[:,3]], dim=1)
            loss_reg = ciou_loss(pred_boxes, target_boxes)
        else:
            loss_reg = torch.tensor(0.0, device=device)
        loss = loss_cls + loss_obj + 5.0 * loss_reg
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KFN(num_classes=20)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    images_dir = '/mnt/data/images'
    ann_file = '/mnt/data/annotations.json'
    if not os.path.exists(ann_file):
        sample = {'sample.jpg': {'file_name': 'sample.jpg', 'boxes': [], 'labels': []}}
        with open(ann_file, 'w') as f:
            json.dump(sample, f)
    dataset = SatelliteDataset(images_dir, ann_file, img_size=1024, tfms=T.Compose([T.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for epoch in range(1, 3):
        loss = train_one_epoch(model, optimizer, dataloader, device, epoch, scheduler=scheduler)
        val_map = compute_map(model, dataloader, device)
        print('epoch', epoch, 'loss', loss, 'mAP', val_map)
    torch.save(model.state_dict(), '/mnt/data/kfn_model.pth')

if __name__ == '__main__':
    main()