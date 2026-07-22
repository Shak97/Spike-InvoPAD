"""
eval_crossset_temporal.py — Video-level cross-dataset evaluation with temporal clips.

Instead of single frames, T consecutive frames from each video are fed as a
[B, T, C, H, W] sequence so the LIF SNN can integrate spike information across
real temporal context.  Clip scores are averaged per video before computing
video-level metrics.

Default: T=3, stride=T (non-overlapping clips).

Usage
-----
python eval_crossset_temporal.py                        # all four protocols, T=3
python eval_crossset_temporal.py --protocol oulu_npu
python eval_crossset_temporal.py --T 5 --stride 2
python eval_crossset_temporal.py --protocol msu_mfsd \\
    --checkpoint checkpoints_cross/test_msu_mfsd/best_hter_0.2100_ep042.pt
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, mobilenet_v3_small,
)


# ---------------------------------------------------------------------------
# Model  (identical to cross_dataset.py / eval_crossset.py)
# ---------------------------------------------------------------------------

class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, groups=None):
        super().__init__()
        self.channels     = channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.reduction    = max(1, reduction)
        self.kernel_norm  = kernel_norm
        self.softmax_temp = softmax_temp
        self.groups = channels if groups is None else int(groups)
        assert self.groups >= 1 and channels % self.groups == 0, \
            f"'groups' must divide channels: got C={channels}, groups={self.groups}"
        hidden       = max(1, channels // self.reduction)
        self.reduce  = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn      = nn.BatchNorm2d(hidden)
        self.act     = nn.ReLU(inplace=True)
        self.kproj   = nn.Conv2d(hidden, kernel_size * kernel_size * self.groups, 1, bias=True)
        self.pool_for_k = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def _normalize_kernel(self, ker):
        if self.kernel_norm == "softmax":
            B, G, k, _, H, W = ker.shape
            ker = F.softmax(ker.view(B, G, k*k, H, W) / self.softmax_temp, dim=2)
            return ker.view(B, G, k, k, H, W)
        if self.kernel_norm == "l2":
            ker = ker - ker.mean(dim=(2, 3), keepdim=True)
            ker = ker / ker.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return ker

    @torch.no_grad()
    def get_kernels(self, x):
        B, C, H, W = x.shape
        k  = self.kernel_size
        xk = self.pool_for_k(x)
        K  = self.kproj(self.act(self.bn(self.reduce(xk))))
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=True)
        return self._normalize_kernel(K.view(B, self.groups, k, k, H, W))

    def forward(self, x):
        B, C, H, W = x.shape
        k, G   = self.kernel_size, self.groups
        groupC = C // G
        K      = self.get_kernels(x)
        xu = F.unfold(x, kernel_size=k, padding=k // 2)
        xu = xu.view(B, C, k, k, H, W).view(B, G, groupC, k, k, H, W)
        return (xu * K.unsqueeze(2)).sum(dim=(3, 4)).view(B, C, H, W)


class InvHead(nn.Module):
    def __init__(self, channels, reduce=1, k=3, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups=None):
        super().__init__()
        hidden      = max(8, channels // reduce)
        self.hidden = hidden
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)
        inv_groups  = hidden if inv_groups is None else int(inv_groups)
        assert hidden % inv_groups == 0, \
            f"'inv_groups' must divide hidden={hidden}; got {inv_groups}"
        self.inv    = Involution(channels=hidden, kernel_size=k, stride=1,
                                 reduction=inv_reduction, kernel_norm=kernel_norm,
                                 softmax_temp=softmax_temp, groups=inv_groups)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)
        self.gamma  = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, slope=25.0):
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        return grad_output / (ctx.slope * input_tensor.abs() + 1.0).pow(2), None


class LIFSpikeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.85, threshold=1.0, slope=50.0):
        super().__init__()
        self.fc        = nn.Linear(input_dim, hidden_dim)
        self.beta      = beta
        self.threshold = threshold
        self.slope     = slope

    def forward(self, x_t, mem):
        cur = self.fc(x_t)
        mem = self.beta * mem + cur
        spk = FastSigmoidSpike.apply(mem - self.threshold, self.slope)
        mem = mem - spk * self.threshold
        return spk, mem


class MobileNetV3_INV_SNN(nn.Module):
    def __init__(self, num_classes=2, variant="large", weights=None,
                 k=3, reduce=1, dropout=0.2, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups=None,
                 snn_hidden=256, beta=0.85, threshold=1.0, spike_slope=50.0):
        super().__init__()
        variant = variant.lower()
        if variant == "large":
            base = mobilenet_v3_large(weights=(weights or MobileNet_V3_Large_Weights.IMAGENET1K_V1))
        elif variant == "small":
            base = mobilenet_v3_small(weights=(weights or MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        else:
            raise ValueError("variant must be 'large' or 'small'")
        self.features    = base.features
        self.feat_dim    = base.classifier[0].in_features
        self.inv_head    = InvHead(channels=self.feat_dim, reduce=reduce, k=k,
                                   inv_reduction=inv_reduction, kernel_norm=kernel_norm,
                                   softmax_temp=softmax_temp, inv_groups=inv_groups)
        self.avgpool     = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(p=dropout)
        self.snn_hidden  = snn_hidden
        self.spike_layer = LIFSpikeLayer(input_dim=self.feat_dim, hidden_dim=snn_hidden,
                                          beta=beta, threshold=threshold, slope=spike_slope)
        self.readout     = nn.Linear(snn_hidden + self.feat_dim, num_classes)
        self.fc          = self.readout

    def _forward_frame_features(self, x_2d):
        x = self.features(x_2d)
        x = self.inv_head(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def extract_intermediate_features(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        frame_feats = self._forward_frame_features(x.view(B * T, C, H, W))
        frame_feats = frame_feats.view(B, T, self.feat_dim)
        mem       = torch.zeros(B, self.snn_hidden, device=frame_feats.device,
                                dtype=frame_feats.dtype)
        spike_seq = []
        for t in range(T):
            spk, mem = self.spike_layer(frame_feats[:, t, :], mem)
            spike_seq.append(spk)
        avg_spike  = torch.stack(spike_seq, dim=1).mean(dim=1)
        continuous = frame_feats.mean(dim=1)
        return frame_feats, torch.cat([avg_spike, continuous], dim=1)

    def forward(self, x):
        _, feats = self.extract_intermediate_features(x)
        return self.readout(self.dropout(feats))


# ---------------------------------------------------------------------------
# Dataset — sequential T-frame clips from each video
# ---------------------------------------------------------------------------

def _vid_id(rel_path):
    """Strip the trailing _NNNN frame index to get a stable video ID."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    return stem.rsplit('_', 1)[0]


def _frame_idx(rel_path):
    """Parse the numeric frame index from the filename suffix."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    try:
        return int(stem.rsplit('_', 1)[1])
    except (IndexError, ValueError):
        return 0


class TemporalClipDataset(Dataset):
    """
    Builds non-overlapping (stride=T by default) sequential clips of T frames
    from each video.  Each item is (clip_tensor [T,C,H,W], binary_label, vid_id).

    Frames with fewer than T available are right-padded with the last frame so
    every video contributes at least one clip.
    """
    def __init__(self, csv_path, image_root, split, transform=None, T=3, stride=None):
        df = pd.read_csv(csv_path)
        self.df         = df[df['filename'].str.startswith(f"{split}/")].reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform
        self.T          = T
        self.stride     = stride if stride is not None else T

        self.clips = self._build_clips()

    def _build_clips(self):
        # Group frames by video, preserving temporal order
        vid_frames = defaultdict(list)  # vid_id -> [(frame_idx, rel_path, binary_label)]
        for _, row in self.df.iterrows():
            rel_path     = row['filename']
            binary_label = 1 if int(row['attack_label']) == 0 else 0
            vid          = _vid_id(rel_path)
            fidx         = _frame_idx(rel_path)
            vid_frames[vid].append((fidx, rel_path, binary_label))

        clips = []
        for vid, frames in vid_frames.items():
            frames.sort(key=lambda x: x[0])          # sort by frame index
            label  = frames[0][2]                     # all frames share the same label
            paths  = [f[1] for f in frames]
            n      = len(paths)

            if n < self.T:
                # Pad to T by repeating the last frame
                padded = paths + [paths[-1]] * (self.T - n)
                clips.append((vid, padded, label))
            else:
                # Non-overlapping (or custom-stride) windows
                i = 0
                while i + self.T <= n:
                    clips.append((vid, paths[i: i + self.T], label))
                    i += self.stride

        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid_id, clip_paths, label = self.clips[idx]
        frames = []
        for rel_path in clip_paths:
            img_path = os.path.join(self.image_root, rel_path)
            if not os.path.exists(img_path) and img_path.endswith('.png'):
                img_path = img_path[:-4] + '.jpg'
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), label, vid_id   # [T, C, H, W], int, str


def collate_clips(batch):
    clips   = torch.stack([b[0] for b in batch])   # [B, T, C, H, W]
    labels  = torch.tensor([b[1] for b in batch])
    vid_ids = [b[2] for b in batch]
    return clips, labels, vid_ids


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Video-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_video_level(model, loader, device):
    """
    Inference on sequential T-frame clips → average clip scores per video → metrics.
    """
    model.eval()

    video_scores = defaultdict(list)   # vid_id -> [clip scores]
    video_labels = {}                  # vid_id -> label

    t_start      = time.time()
    total_clips  = 0
    total_frames = 0

    for clips, batch_labels, vid_ids in loader:
        # clips: [B, T, C, H, W]
        clips = clips.to(device, non_blocking=True)
        logits = model(clips)                                  # [B, 2]
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        for i, vid_id in enumerate(vid_ids):
            video_scores[vid_id].append(float(scores[i]))
            video_labels[vid_id] = int(batch_labels[i])

        total_clips  += clips.size(0)
        total_frames += clips.size(0) * clips.size(1)

    elapsed = time.time() - t_start

    # --- Aggregation: average clip scores per video ---
    final_scores, final_labels = [], []
    for vid_id, sc in video_scores.items():
        final_scores.append(np.mean(sc))
        final_labels.append(video_labels[vid_id])

    scores_np = np.array(final_scores, dtype=np.float64)
    labels_np = np.array(final_labels, dtype=np.int64)
    n_videos  = len(scores_np)

    # --- Metrics ---
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    auc_roc = auc(fpr, tpr)
    fnr     = 1.0 - tpr

    eer_idx           = np.argmin(np.abs(fpr - fnr))
    eer               = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    youden_idx        = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[youden_idx])
    youdens_index     = float(tpr[youden_idx] - fpr[youden_idx])

    preds      = (scores_np >= optimal_threshold).astype(int)
    test_acc   = float((preds == labels_np).mean())
    live_mask  = labels_np == 1
    spoof_mask = labels_np == 0
    far  = float((preds[spoof_mask] == 1).sum() / max(spoof_mask.sum(), 1))
    frr  = float((preds[live_mask]  == 0).sum() / max(live_mask.sum(),  1))
    hter = (far + frr) / 2.0

    return {
        'test_acc':            test_acc,
        'auc_roc':             auc_roc,
        'eer':                 eer,
        'hter':                hter,
        'far':                 far,
        'frr':                 frr,
        'youdens_index':       youdens_index,
        'optimal_threshold':   optimal_threshold,
        'n_videos':            n_videos,
        'n_clips':             total_clips,
        'n_frames':            total_frames,
        'avg_clips_per_video': total_clips / max(n_videos, 1),
        'total_inference_s':   elapsed,
        'fpr':    fpr,
        'tpr':    tpr,
        'labels': labels_np,
        'scores': scores_np,
    }


def print_summary(protocol_name, val, T, stride):
    w = 64
    print(f"\n{'='*w}")
    print(f" Temporal clip results (T={T}, stride={stride}) — {protocol_name.upper()}")
    print(f"{'='*w}")
    print(f"  Videos  : {val['n_videos']}   "
          f"Clips : {val['n_clips']}   "
          f"Frames : {val['n_frames']}   "
          f"Clips/video : {val['avg_clips_per_video']:.1f}")
    print(f"  Accuracy         : {val['test_acc']*100:.2f}%")
    print(f"  AUC-ROC          : {val['auc_roc']:.4f}")
    print(f"  EER              : {val['eer']*100:.4f}%")
    print(f"  HTER             : {val['hter']*100:.4f}%")
    print(f"  FAR              : {val['far']*100:.4f}%")
    print(f"  FRR              : {val['frr']*100:.4f}%")
    print(f"  Youden's J       : {val['youdens_index']:.4f}")
    print(f"  Optimal threshold: {val['optimal_threshold']:.4f}")
    print(f"  Total infer time : {val['total_inference_s']:.2f}s")
    print(f"{'='*w}")


def save_roc_plot(val, save_path, protocol_name, T, stride):
    plt.figure(figsize=(7, 6))
    plt.plot(val['fpr'], val['tpr'], color='darkorange', lw=2,
             label=f"AUC = {val['auc_roc']:.4f}  |  HTER = {val['hter']:.4f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Video-level ROC (T={T}, stride={stride}) — {protocol_name.upper()}")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC saved → {save_path}")


# ---------------------------------------------------------------------------
# Protocol / dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    'oulu_npu': {
        'csv':  'oulu_npu_attack_labels.csv',
        'root': '/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU',
    },
    'casia_fasd': {
        'csv':  'casia_fasd_attack_labels.csv',
        'root': '/media/oem/storage01/Shakeel/facePAD_datasets/datasets/CASIA_FASD_paper_cropped',
    },
    'msu_mfsd': {
        'csv':  'msu_mfsd_attack_labels.csv',
        'root': '/media/oem/storage01/Shakeel/facePAD_datasets/datasets/MSU-MFSD_faces/MSU-MFSD_train_test',
    },
    'idiap': {
        'csv':  'replay_attack_labels_new.csv',
        'root': '/media/oem/storage01/Shakeel/facePAD_datasets/Replay_Attack_imgs_cropped_subsampled',
    },
}

ALL_NAMES = ['oulu_npu', 'casia_fasd', 'msu_mfsd', 'idiap']


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(fold_dir):
    candidates = glob.glob(os.path.join(fold_dir, 'best_hter_*.pt'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: float(os.path.basename(p).split('_')[2]))
    return candidates[0]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Video-level temporal-clip evaluation for MobileNetV3_INV_SNN")
    p.add_argument('--protocol',    default=None, choices=ALL_NAMES,
                   help='Evaluate a single protocol (default: all four)')
    p.add_argument('--checkpoint',  default=None,
                   help='Path to a specific .pt checkpoint (overrides auto-discovery)')
    p.add_argument('--ckpt_dir',    default='checkpoints_cross')
    p.add_argument('--label_dir',   default='/media/oem/storage01/Shakeel/facePAD_datasets/datasets/attack_labels_data_new')
    p.add_argument('--save_dir',    default='eval_crossset_temporal_results')
    p.add_argument('--T',           type=int, default=3,
                   help='Number of sequential frames per clip (default: 3)')
    p.add_argument('--stride',      type=int, default=None,
                   help='Clip stride in frames (default: T, i.e. non-overlapping)')
    p.add_argument('--batch_size',  type=int, default=32,
                   help='Batch size in clips (memory scales with T*batch_size frames)')
    p.add_argument('--num_workers', type=int, default=4)
    # Model — reconstructed from checkpoint args; these are fallbacks
    p.add_argument('--variant',     default='large', choices=['small', 'large'])
    p.add_argument('--k',           type=int,   default=3)
    p.add_argument('--reduce',      type=int,   default=1)
    p.add_argument('--inv_groups',  type=int,   default=None)
    p.add_argument('--dropout',     type=float, default=0.2)
    p.add_argument('--snn_hidden',  type=int,   default=256)
    p.add_argument('--beta',        type=float, default=0.85)
    p.add_argument('--spike_slope', type=float, default=50.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf      = get_test_transform()
    stride  = args.stride if args.stride is not None else args.T
    os.makedirs(args.save_dir, exist_ok=True)

    protocols = ALL_NAMES if args.protocol is None else [args.protocol]

    weights = (MobileNet_V3_Large_Weights.IMAGENET1K_V1
               if args.variant == 'large'
               else MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    print(f"Device: {device}  |  T={args.T}  stride={stride}")

    summary_rows = []

    for test_name in protocols:
        fold_dir  = os.path.join(args.ckpt_dir, f"test_{test_name}")
        ckpt_path = args.checkpoint or find_best_checkpoint(fold_dir)

        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"\n[{test_name}] No checkpoint found in {fold_dir} — skipping.")
            continue

        print(f"\n[{test_name}] Checkpoint: {ckpt_path}")
        ckpt   = torch.load(ckpt_path, map_location=device)
        saved  = ckpt.get('args', {})

        model = MobileNetV3_INV_SNN(
            num_classes  = 2,
            variant      = saved.get('variant',    args.variant),
            weights      = weights,
            k            = saved.get('k',           args.k),
            reduce       = saved.get('reduce',      args.reduce),
            inv_groups   = saved.get('inv_groups',  args.inv_groups),
            dropout      = saved.get('dropout',     args.dropout),
            snn_hidden   = saved.get('snn_hidden',  args.snn_hidden),
            beta         = saved.get('beta',        args.beta),
            spike_slope  = saved.get('spike_slope', args.spike_slope),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()

        meta     = DATASETS[test_name]
        csv_path = os.path.join(args.label_dir, meta['csv'])
        ds       = TemporalClipDataset(csv_path, meta['root'], 'test',
                                       transform=tf, T=args.T, stride=stride)
        loader   = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_clips)

        print(f"  Videos : {len(set(c[0] for c in ds.clips))}   "
              f"Clips : {len(ds)}   "
              f"(T={args.T}, stride={stride})")

        val = evaluate_video_level(model, loader, device)
        print_summary(test_name, val, args.T, stride)

        plot_path = os.path.join(args.save_dir, f"{test_name}_roc.png")
        save_roc_plot(val, plot_path, test_name, args.T, stride)

        summary_rows.append({
            'protocol':          test_name,
            'T':                 args.T,
            'stride':            stride,
            'checkpoint':        os.path.basename(ckpt_path),
            'n_videos':          val['n_videos'],
            'n_clips':           val['n_clips'],
            'n_frames':          val['n_frames'],
            'avg_clips_per_vid': f"{val['avg_clips_per_video']:.1f}",
            'test_acc':          f"{val['test_acc']:.6f}",
            'auc_roc':           f"{val['auc_roc']:.6f}",
            'eer':               f"{val['eer']:.6f}",
            'hter':              f"{val['hter']:.6f}",
            'far':               f"{val['far']:.6f}",
            'frr':               f"{val['frr']:.6f}",
            'youdens_index':     f"{val['youdens_index']:.6f}",
            'optimal_threshold': f"{val['optimal_threshold']:.6f}",
        })

        del model
        torch.cuda.empty_cache()

    if summary_rows:
        import csv
        csv_out = os.path.join(args.save_dir, 'temporal_results.csv')
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary CSV → {csv_out}")


if __name__ == '__main__':
    main()
