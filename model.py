"""
model.py — MobileNetV3 + Involution head + LIF SNN for Face PAD

Exports:
    MobileNetV3_INV_SNN  (canonical name)
    MobileNetV3_INV      (alias for backward compat)
    SupConLoss
    ProjectionHead
    AdaptiveCenterCropAndResize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
)
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------

class AdaptiveCenterCropAndResize:
    """Center-crops to the largest square then resizes. Accepts tensor or PIL."""

    def __init__(self, output_size):
        self.output_size = output_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize(self.output_size, Image.Resampling.LANCZOS)
        return self.to_tensor(img)


# ---------------------------------------------------------------------------
# Involution block
# ---------------------------------------------------------------------------

class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, groups=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = max(1, reduction)
        self.kernel_norm = kernel_norm
        self.softmax_temp = softmax_temp
        self.groups = channels if groups is None else int(groups)
        assert self.groups >= 1 and channels % self.groups == 0, \
            f"groups must divide channels: C={channels}, groups={self.groups}"

        hidden = max(1, channels // self.reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.kproj = nn.Conv2d(hidden, kernel_size * kernel_size * self.groups, 1, bias=True)
        self.pool_for_k = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def _normalize_kernel(self, ker):
        if self.kernel_norm == "softmax":
            B, G, k, _, H, W = ker.shape
            ker = F.softmax(ker.view(B, G, k * k, H, W) / self.softmax_temp, dim=2)
            return ker.view(B, G, k, k, H, W)
        if self.kernel_norm == "l2":
            ker = ker - ker.mean(dim=(2, 3), keepdim=True)
            ker = ker / ker.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return ker

    @torch.no_grad()
    def get_kernels(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        xk = self.pool_for_k(x)
        K = self.kproj(self.act(self.bn(self.reduce(xk))))
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=True)
        return self._normalize_kernel(K.view(B, self.groups, k, k, H, W))

    def forward(self, x):
        B, C, H, W = x.shape
        k, G = self.kernel_size, self.groups
        groupC = C // G
        K = self.get_kernels(x)
        x_unfold = F.unfold(x, kernel_size=k, padding=k // 2)
        x_unfold = x_unfold.view(B, C, k, k, H, W).view(B, G, groupC, k, k, H, W)
        return (x_unfold * K.unsqueeze(2)).sum(dim=(3, 4)).view(B, C, H, W)


class InvHead(nn.Module):
    def __init__(self, channels, reduce=4, k=9, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups=None):
        super().__init__()
        hidden = max(8, channels // reduce)
        self.hidden = hidden
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)

        g = hidden if inv_groups is None else int(inv_groups)
        assert hidden % g == 0, f"inv_groups must divide hidden={hidden}; got {g}"
        self.inv = Involution(hidden, k, 1, inv_reduction, kernel_norm, softmax_temp, g)

        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)


# ---------------------------------------------------------------------------
# Spiking neuron
# ---------------------------------------------------------------------------

class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope=25.0):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (ctx.slope * x.abs() + 1.0).pow(2), None


class LIFSpikeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.95, threshold=1.0, slope=25.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def forward(self, x_t, mem):
        mem = self.beta * mem + self.fc(x_t)
        spk = FastSigmoidSpike.apply(mem - self.threshold, self.slope)
        return spk, mem - spk * self.threshold


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MobileNetV3_INV_SNN(nn.Module):
    """
    MobileNetV3 backbone + Involution head + LIF spiking temporal classifier.
    Accepts [B, C, H, W] (single frame) or [B, T, C, H, W] (video).
    """

    def __init__(self, num_classes=2, variant="large", weights=None,
                 k=3, reduce=1, dropout=0.2, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups=None,
                 snn_hidden=256, beta=0.85, threshold=1.0, spike_slope=50.0):
        super().__init__()
        variant = variant.lower()
        if variant == "large":
            base = mobilenet_v3_large(weights=weights or MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        elif variant == "small":
            base = mobilenet_v3_small(weights=weights or MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("variant must be 'large' or 'small'")

        self.features = base.features
        self.feat_dim = base.classifier[0].in_features

        self.inv_head = InvHead(self.feat_dim, reduce, k, inv_reduction,
                                kernel_norm, softmax_temp, inv_groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        self.snn_hidden = snn_hidden
        self.spike_layer = LIFSpikeLayer(self.feat_dim, snn_hidden, beta, threshold, spike_slope)

        self.readout = nn.Linear(snn_hidden + self.feat_dim, num_classes)
        self.fc = self.readout  # backward compat alias

    def _frame_features(self, x):
        x = self.features(x)
        x = self.inv_head(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def extract_intermediate_features(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        frame_feats = self._frame_features(x.view(B * T, C, H, W)).view(B, T, self.feat_dim)

        mem = torch.zeros(B, self.snn_hidden, device=frame_feats.device, dtype=frame_feats.dtype)
        spikes = []
        for t in range(T):
            spk, mem = self.spike_layer(frame_feats[:, t], mem)
            spikes.append(spk)

        avg_spikes = torch.stack(spikes, 1).mean(1)
        cont_feats = frame_feats.mean(1)
        return frame_feats, torch.cat([avg_spikes, cont_feats], dim=1)

    def forward_features(self, x):
        return self.extract_intermediate_features(x)[1]

    def forward(self, x):
        return self.readout(self.dropout(self.forward_features(x)))


MobileNetV3_INV = MobileNetV3_INV_SNN  # alias


# ---------------------------------------------------------------------------
# Loss + projection head for optional SupCon training
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — stable version."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        features = F.normalize(features, dim=-1)
        B, V, D = features.shape
        feats = features.reshape(B * V, D)

        logits = torch.matmul(feats, feats.t()) / self.t
        self_mask = torch.eye(B * V, dtype=torch.bool, device=feats.device)
        logits = logits.masked_fill(self_mask, -1e9)
        log_prob = F.log_softmax(logits, dim=1)

        labels = labels.view(B, 1)
        pos_mask = (labels == labels.t()).float().to(feats.device)
        pos_mask = pos_mask.repeat_interleave(V, 0).repeat_interleave(V, 1)
        pos_mask = pos_mask.masked_fill(self_mask, 0.0)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return logits.new_zeros(())

        mean_log_prob = (log_prob * pos_mask).sum(1) / pos_count.clamp(min=1)
        return -mean_log_prob[valid].mean()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
