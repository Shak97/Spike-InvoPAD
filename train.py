"""
train.py — OULU-NPU Face PAD training for all four protocols.

Usage
-----
# Protocol 1 (single model, all attack types):
python train.py --protocol 1

# Protocol 2 (single model, limited attack types):
python train.py --protocol 2

# Protocol 3 (Leave-One-Camera-Out, single fold):
python train.py --protocol 3 --fold 2

# Protocol 3 (all 6 folds automatically):
python train.py --protocol 3 --all_folds

# Protocol 4 (all folds, larger model):
python train.py --protocol 4 --all_folds --variant large

All checkpoints and logs go to --ckpt_dir (default: checkpoints/).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_oulu import OULUProtocolDataset, collate_fn
from model import (
    MobileNetV3_INV_SNN,
    ProjectionHead,
    SupConLoss,
    AdaptiveCenterCropAndResize,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OULU-NPU Face PAD Trainer")

    p.add_argument("--protocol",    type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--fold",        type=int, default=None,
                   help="Fold index 1-6 for protocols 3/4. Ignored for 1/2.")
    p.add_argument("--all_folds",   action="store_true",
                   help="Loop all 6 folds for protocols 3/4.")

    # paths
    p.add_argument("--csv_dir",     default="protocol_csvs",
                   help="Root of protocol CSV files from generate_protocol_csvs.py")
    p.add_argument("--ckpt_dir",    default="checkpoints",
                   help="Root directory for saved checkpoints")

    # model
    p.add_argument("--variant",     default="small", choices=["small", "large"])
    p.add_argument("--num_frames",  type=int, default=1)
    p.add_argument("--snn_hidden",  type=int, default=256)
    p.add_argument("--beta",        type=float, default=0.85)
    p.add_argument("--spike_slope", type=float, default=50.0)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--k",           type=int, default=3,
                   help="Involution kernel size")
    p.add_argument("--reduce",      type=int, default=1,
                   help="Channel reduction factor in InvHead")

    # training
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int, default=40,
                   help="Early stopping patience (epochs)")
    p.add_argument("--supcon_weight", type=float, default=0.0,
                   help="Weight for SupCon loss (0 = pure CE)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms():
    train_tf = transforms.Compose([
        AdaptiveCenterCropAndResize((256, 256)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        AdaptiveCenterCropAndResize((256, 256)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, device):
    model = MobileNetV3_INV_SNN(
        num_classes=2,
        variant=args.variant,
        k=args.k,
        reduce=args.reduce,
        dropout=args.dropout,
        snn_hidden=args.snn_hidden,
        beta=args.beta,
        spike_slope=args.spike_slope,
    ).to(device)

    proj_head = ProjectionHead(
        in_dim=model.snn_hidden + model.feat_dim,
        hidden=256,
        out_dim=128,
    ).to(device)

    return model, proj_head


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, proj_head, loader, optimizer,
                ce_criterion, supcon_criterion, supcon_weight, device, epoch):
    model.train()
    proj_head.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc=f"Train E{epoch}", leave=False) as bar:
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()
            _, feats = model.extract_intermediate_features(inputs)

            loss = torch.zeros(1, device=device)
            if supcon_weight > 0:
                z = proj_head(feats)
                loss = loss + supcon_weight * supcon_criterion(z, labels)

            logits = model.fc(feats)
            loss = loss + ce_criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(loss=total_loss / total, acc=100 * correct / total)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate_epoch(model, loader, ce_criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        logits = model(inputs)
        loss = ce_criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full evaluation (FacePAD metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    correct, total = 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        logits = model(inputs)
        probs = torch.softmax(logits, 1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_roc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    best_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[best_idx]
    far = fpr[best_idx]
    frr = fnr[best_idx]
    hter = (far + frr) / 2.0

    return {
        "acc":   100.0 * correct / max(total, 1),
        "auc":   float(auc_roc),
        "eer":   float(eer),
        "hter":  float(hter),
        "far":   float(far),
        "frr":   float(frr),
        "thresh": float(opt_thresh),
    }


def print_metrics(tag, m):
    print(f"  [{tag}] Acc={m['acc']:.2f}%  AUC={m['auc']:.4f}  "
          f"EER={m['eer']:.4f}  HTER={m['hter']:.4f}  "
          f"FAR={m['far']:.4f}  FRR={m['frr']:.4f}")


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(args, protocol, fold, device):
    tag = f"p{protocol}" + (f"_f{fold}" if fold is not None else "")
    csv_root = os.path.join(args.csv_dir, f"Protocol_{protocol}")
    ckpt_root = os.path.join(args.ckpt_dir, f"Protocol_{protocol}")
    os.makedirs(ckpt_root, exist_ok=True)

    suffix = f"_{fold}" if fold is not None else ""
    train_tf, eval_tf = get_transforms()

    def make_loader(split, is_train):
        csv_path = os.path.join(csv_root, f"{split}{suffix}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run generate_protocol_csvs.py first."
            )
        ds = OULUProtocolDataset(csv_path,
                                 transform=train_tf if is_train else eval_tf,
                                 num_frames=args.num_frames,
                                 is_train=is_train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=is_train,
                          num_workers=args.num_workers, collate_fn=collate_fn,
                          pin_memory=True)

    train_loader = make_loader("Train", True)
    dev_loader   = make_loader("Dev",   False)
    test_loader  = make_loader("Test",  False)

    print(f"\n{'='*60}")
    print(f" Protocol {protocol}  |  Fold: {fold if fold else 'N/A'}  |  Tag: {tag}")
    print(f" Train: {len(train_loader.dataset)}  Dev: {len(dev_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")
    print(f"{'='*60}")

    model, proj_head = build_model(args, device)
    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=0.07)

    optimizer = optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_dev_loss = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        proj_head.load_state_dict(ckpt["proj_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev_loss = ckpt.get("best_dev_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    best_ckpt_path = os.path.join(ckpt_root, f"best_{tag}.pth")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, proj_head, train_loader, optimizer,
            ce_criterion, supcon_criterion, args.supcon_weight, device, epoch + 1,
        )
        dev_loss, dev_acc = validate_epoch(model, dev_loader, ce_criterion, device)

        elapsed = time.time() - t0
        print(f"Ep {epoch+1:3d}/{args.epochs} | "
              f"Tr {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Dev {dev_loss:.4f}/{dev_acc:.2f}% | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch + 1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "dev_loss": dev_loss, "dev_acc": dev_acc})

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "proj_head_state_dict": proj_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_loss": best_dev_loss,
                "dev_acc": dev_acc,
                "args": vars(args),
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

    # save training history
    with open(os.path.join(ckpt_root, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # final evaluation with best checkpoint
    print(f"\nLoading best checkpoint: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    dev_metrics  = evaluate(model, dev_loader,  device)
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n--- Final results [{tag}] ---")
    print_metrics("Dev ", dev_metrics)
    print_metrics("Test", test_metrics)

    results = {"tag": tag, "protocol": protocol, "fold": fold,
               "dev": dev_metrics, "test": test_metrics}
    with open(os.path.join(ckpt_root, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    protocol = args.protocol
    needs_folds = protocol in (3, 4)

    if needs_folds:
        if args.all_folds:
            folds = list(range(1, 7))
        elif args.fold is not None:
            folds = [args.fold]
        else:
            print("Protocol 3/4 requires --fold N (1-6) or --all_folds. Defaulting to fold 1.")
            folds = [1]
    else:
        folds = [None]

    all_results = []
    for fold in folds:
        result = train_fold(args, protocol, fold, device)
        all_results.append(result)

    # aggregate across folds
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f" Protocol {protocol} — Aggregate over {len(all_results)} folds")
        print(f"{'='*60}")
        for split in ("dev", "test"):
            for metric in ("acc", "auc", "eer", "hter", "far", "frr"):
                vals = [r[split][metric] for r in all_results]
                print(f"  {split} {metric:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        ckpt_root = os.path.join(args.ckpt_dir, f"Protocol_{protocol}")
        with open(os.path.join(ckpt_root, "all_fold_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
