import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import swin_t, Swin_T_Weights

class SpectralContributor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.block2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.block3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.block4(x)
        x = self.block5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

class SpatialContributor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

    def forward(self, x):
        return self.features(x)

class ContextualContributor(nn.Module):
    def __init__(self):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.patch_embed = swin.features[0]
        self.layer1 = swin.features[1]
        self.layer2 = swin.features[2]
        self.layer3 = swin.features[3]
        self.layer4 = swin.features[4]
        self.layer5 = swin.features[5]
        self.layer6 = swin.features[6]
        self.proj = nn.Conv2d(384, 256, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        B, L, C = x.shape
        H = int(L ** 0.5)
        W = H
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class FusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(768)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(768, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

    def forward(self, f_spectral, f_spatial, f_contextual):
        x = torch.cat([f_spectral, f_spatial, f_contextual], dim=1)
        x = self.cbam(x)
        x = self.bottleneck(x)
        return x

class DetectionHead(nn.Module):
    def __init__(self, num_classes, in_channels=512):
        super().__init__()
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        cls = self.cls_pred(cls_feat)
        reg = self.reg_pred(reg_feat)
        obj = self.obj_pred(reg_feat)
        return cls, reg, obj

class KFN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.spectral_contributor = SpectralContributor()
        self.spatial_contributor = SpatialContributor()
        self.contextual_contributor = ContextualContributor()
        self.fusion_network = FusionNetwork()
        self.detection_head = DetectionHead(num_classes)

    def forward(self, x):
        f_spectral = self.spectral_contributor(x)
        f_spatial = self.spatial_contributor(x)
        f_contextual = self.contextual_contributor(x)
        fused = self.fusion_network(f_spectral, f_spatial, f_contextual)
        return self.detection_head(fused)

def quality_focal_loss(pred, target, beta=2.0):
    r = target * (1.0 - pred) ** beta * F.logsigmoid(pred) + \
        (1.0 - target) * pred ** beta * F.logsigmoid(1.0 - pred)
    return -r.mean()

def ciou_loss(pred, target):
    b1_x1, b1_y1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    b1_x2, b1_y2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    b2_x1, b2_y1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
    b2_x2, b2_y2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2)
    a = v / (1 + v - iou + 1e-7)
    ciou = iou - (rho2 / c2 + v * a)
    return 1 - ciou

def compute_loss(cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target):
    cls_loss = quality_focal_loss(cls_pred, cls_target)
    reg_loss = ciou_loss(reg_pred, reg_target)
    obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
    total_loss = 1.0 * cls_loss + 5.0 * reg_loss + 1.0 * obj_loss
    return total_loss