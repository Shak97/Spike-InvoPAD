import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import transforms
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
)

# ==========================================
# 1. DATASET DEFINITION & LIFECYCLE HELPERS
# ==========================================

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=16, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['attack', 'real']
        self.samples = self._load_samples()
        self.num_frames = num_frames
        self.is_train = is_train

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.avi'):
                    samples.append((os.path.join(cls_dir, fname), self.classes.index(cls)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path, self.num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames, label

    def _load_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = np.random.randint(0, max(1, total_frames - num_frames + 1)) if self.is_train else 0

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(TF.to_tensor(frame))

        cap.release()

        while len(frames) < num_frames:
            if len(frames) == 0:
                frames.append(torch.zeros(3, 256, 256))
            else:
                frames.append(frames[-1])

        return frames


class AdaptiveCenterCropAndResize:
    def __init__(self, output_size):
        self.output_size = output_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        img = img.crop((left, top, right, bottom))
        img = img.resize(self.output_size, Image.Resampling.LANCZOS)
        return self.to_tensor(img)


def collate_fn(batch):
    max_length = max([frames.size(0) for frames, _ in batch])
    padded_frames = []
    labels = []
    for frames, label in batch:
        if frames.size(0) < max_length:
            padding = torch.zeros((max_length - frames.size(0), frames.size(1), frames.size(2), frames.size(3)))
            padded_frames.append(torch.cat((frames, padding), dim=0))
        else:
            padded_frames.append(frames)
        labels.append(label)

    padded_frames = torch.stack(padded_frames)
    labels = torch.tensor(labels)
    return padded_frames, labels


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
)

# ---------- Involution (group = input channels) ----------
class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4,
                 kernel_norm: str = "l2", softmax_temp: float = 1.0,
                 groups: int | None = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = max(1, reduction)
        self.kernel_norm = kernel_norm
        self.softmax_temp = softmax_temp

        self.groups = channels if (groups is None) else int(groups)
        assert self.groups >= 1 and channels % self.groups == 0,             f"'groups' must divide channels: got C={channels}, groups={self.groups}"

        hidden = max(1, channels // self.reduction)

        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn     = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)
        self.kproj  = nn.Conv2d(hidden, (kernel_size * kernel_size) * self.groups,
                                kernel_size=1, bias=True)

        self.pool_for_k = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def _normalize_kernel(self, ker: torch.Tensor) -> torch.Tensor:
        if self.kernel_norm == "softmax":
            B, G, k, _, H, W = ker.shape
            ker = F.softmax(ker.view(B, G, k*k, H, W) / self.softmax_temp, dim=2)
            return ker.view(B, G, k, k, H, W)
        if self.kernel_norm == "l2":
            ker = ker - ker.mean(dim=(2, 3), keepdim=True)
            denom = ker.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            ker = ker / denom
        return ker

    @torch.no_grad()
    def get_kernels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        xk = self.pool_for_k(x)
        K  = self.kproj(self.act(self.bn(self.reduce(xk))))
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=True)
        K = K.view(B, self.groups, k, k, H, W)
        return self._normalize_kernel(K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, G = self.kernel_size, self.groups
        assert C % G == 0
        groupC = C // G

        K = self.get_kernels(x)
        x_unfold = F.unfold(x, kernel_size=k, padding=k//2)
        x_unfold = x_unfold.view(B, C, k, k, H, W).view(B, G, groupC, k, k, H, W)
        out = (x_unfold * K.unsqueeze(2)).sum(dim=(3,4)).view(B, C, H, W)
        return out


class InvHead(nn.Module):
    def __init__(self, channels, reduce=4, k=9, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups: int | None = None):
        super().__init__()
        hidden = max(8, channels // reduce)
        self.hidden = hidden

        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)

        if inv_groups is None:
            inv_groups = hidden
        else:
            inv_groups = int(inv_groups)
        assert hidden % inv_groups == 0,             f"'inv_groups' must divide hidden={hidden}; got {inv_groups}"

        self.inv = Involution(
            channels=hidden, kernel_size=k, stride=1,
            reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, groups=inv_groups
        )

        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)
        self.gamma  = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)


# ---------- Surrogate-gradient spike function ----------
class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, slope: float = 25.0):
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * input_tensor.abs() + 1.0).pow(2)
        return grad, None


class LIFSpikeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.95, threshold=1.0, slope=25.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def forward(self, x_t, mem):
        cur = self.fc(x_t)
        mem = self.beta * mem + cur
        spk = FastSigmoidSpike.apply(mem - self.threshold, self.slope)
        mem = mem - spk * self.threshold
        return spk, mem


class MobileNetV3_INV_SNN(nn.Module):
    """
    MobileNetV3 backbone + Involution head + LIF spiking temporal classifier.
    Features from the continuous backbone and the discrete SNN are concatenated 
    together before being passed into the final classification readout.
    
    Accepts either:
      - image batch: [B, C, H, W]
      - video batch: [B, T, C, H, W]
    For static images, T=1 is used automatically.
    """
    def __init__(self, num_classes, variant: str = "large",
                 weights=None, k=3, reduce=1, dropout=0.2,            # Optimized defaults: k=3, reduce=1
                 inv_reduction=4, kernel_norm="l2", softmax_temp=1.0,
                 inv_groups: int | None = None,
                 snn_hidden: int = 256, beta: float = 0.85,           # Optimized defaults: beta=0.85
                 threshold: float = 1.0, spike_slope: float = 50.0):  # Optimized defaults: spike_slope=50.0
        super().__init__()
        variant = variant.lower()
        if variant == "large":
            base = mobilenet_v3_large(weights=(weights or MobileNet_V3_Large_Weights.IMAGENET1K_V1))
        elif variant == "small":
            base = mobilenet_v3_small(weights=(weights or MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        else:
            raise ValueError("variant must be 'large' or 'small'")

        self.features = base.features
        self.feat_dim = base.classifier[0].in_features

        self.inv_head = InvHead(
            channels=self.feat_dim, reduce=reduce, k=k,
            inv_reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, inv_groups=inv_groups
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        # temporal spiking head
        self.snn_hidden = snn_hidden
        self.spike_layer = LIFSpikeLayer(
            input_dim=self.feat_dim,
            hidden_dim=snn_hidden,
            beta=beta,
            threshold=threshold,
            slope=spike_slope,
        )
        
        # FIXED: The input size is now the sum of both feature vectors (snn_hidden + self.feat_dim)
        # For example: 256 (spikes) + 576 (backbone) = 832 total dimensions
        self.readout = nn.Linear(snn_hidden + self.feat_dim, num_classes)

        # keep .fc for compatibility with the rest of the notebook
        self.fc = self.readout

    def _forward_frame_features(self, x_2d):
        x = self.features(x_2d)
        x = self.inv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def extract_intermediate_features(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        frame_feats = self._forward_frame_features(x)
        frame_feats = frame_feats.view(B, T, self.feat_dim)

        # Initialize membrane potential for the LIF neurons
        mem = torch.zeros(B, self.snn_hidden, device=frame_feats.device, dtype=frame_feats.dtype)
        spike_seq = []

        # Run the spiking simulation loop across timestamps
        for t in range(T):
            spk, mem = self.spike_layer(frame_feats[:, t, :], mem)
            spike_seq.append(spk)

        # Average spiking features across time frames -> Shape: [B, snn_hidden] (e.g., 256)
        spike_seq = torch.stack(spike_seq, dim=1)
        avg_spike_feats = spike_seq.mean(dim=1)  

        # Mean-pool original continuous backbone features over time -> Shape: [B, feat_dim] (e.g., 576)
        continuous_residual = frame_feats.mean(dim=1)  
        
        # --- FIXED: Concatenate continuous maps and sparse spiking statistics ---
        # Output shape: [B, snn_hidden + feat_dim] (e.g., [B, 832])
        combined_feats = torch.cat([avg_spike_feats, continuous_residual], dim=1)
        
        return frame_feats, combined_feats

    def forward_features(self, x):
        _, feats = self.extract_intermediate_features(x)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.readout(self.dropout(feats))
        return logits


# ==========================================
# 3. METRICS EVALUATION PIPELINE
# ==========================================

def calculate_iso_metrics(fps, fns, tps, tns):
    """Computes basic industrial FacePAD metrics like FAR, FRR, HTER."""
    far = fps / (tns + fps + 1e-7)
    frr = fns / (tps + fns + 1e-7)
    hter = (far + frr) / 2.0
    return far, frr, hter


@torch.no_grad()
def evaluate_all(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    correct = 0
    total = 0

    start_time = time.time()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)

        probs = F.softmax(logits, dim=1)[:, 1]
        _, preds = torch.max(logits, 1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    total_time = time.time() - start_time
    avg_inference_time = total_time / max(total, 1)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_roc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]

    youdens_index_list = tpr - fpr
    best_idx = np.argmax(youdens_index_list)
    optimal_threshold = thresholds[best_idx]
    youdens_index = youdens_index_list[best_idx]

    binary_preds = (all_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(all_labels, binary_preds)

    tn, fp, fn, tp = cm.ravel()
    far, frr, hter = calculate_iso_metrics(fp, fn, tp, tn)
    test_acc = 100. * correct / total

    return {
        'test_acc': test_acc,
        'auc_roc': auc_roc,
        'eer': eer,
        'hter': hter,
        'far': far,
        'frr': frr,
        'youdens_index': youdens_index,
        'optimal_threshold': optimal_threshold,
        'avg_inference_time': avg_inference_time,
        'fpr': fpr,
        'tpr': tpr,
        'labels': all_labels,
        'probs': all_probs
    }


def generate_evaluation_summary(results, save_dir="evaluation_results"):
    os.makedirs(save_dir, exist_ok=True)

    test_acc = results['test_acc']
    auc_roc = results['auc_roc']
    eer = results['eer']
    hter = results['hter']
    far = results['far']
    frr = results['frr']
    youdens_index = results['youdens_index']
    optimal_threshold = results['optimal_threshold']
    avg_inference_time = results['avg_inference_time']
    fpr = results['fpr']
    tpr = results['tpr']

    print("\n--- Evaluation Summary ---")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Half Total Error Rate (HTER): {hter:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"Youden's Index (Max): {youdens_index:.4f}")
    print(f"Optimal Threshold (Youden's Index): {optimal_threshold:.4f}")
    print(f"Average inference time per sample: {avg_inference_time:.6f} seconds")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    output_plot_path = os.path.join(save_dir, "auc_roc_curve.png")
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved evaluation curve directly to disk at: '{output_plot_path}'")


# ==========================================
# 4. EXECUTION DRIVER
# ==========================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using processing unit: {device}")

    dataset_path = '/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU_videos'
    test_path = os.path.join(dataset_path, 'Test_files')

    transform = transforms.Compose([
        AdaptiveCenterCropAndResize((256, 256)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if not os.path.exists(test_path):
        print(f"Error: Target split directory not found at {test_path}")
    else:
        test_dataset = VideoDataset(root_dir=test_path, transform=transform, num_frames=1, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, pin_memory=True)

        model = MobileNetV3_INV_SNN(
            num_classes=2,
            variant="small",
            dropout=0.2,
            snn_hidden=256,
            beta=0.85,
            spike_slope=50.0,
        ).to(device)

        checkpoint_path = r'checkpoints_proposed/best_ce_supervised.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded valid checkpoint model weight file: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Executing evaluation with base weights.")

        results = evaluate_all(model, test_loader, device)
        generate_evaluation_summary(results, save_dir="evaluation_results_spike_without_involution")
