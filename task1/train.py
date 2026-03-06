# train.py
# Continual learning on ImageNet cat categories (281-293) using pretrained ResNet50
# Evaluates cat accuracy and overall 1000-class accuracy on validation set
# Supports checkpoint save/resume

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

DATA_DIR   = Path(__file__).parent.parent / "data"
TRAIN_DIR  = DATA_DIR / "train"
VAL_DIR    = DATA_DIR / "val"
CKPT_DIR   = Path(__file__).parent / "checkpoints"

CAT_LABELS = list(range(281, 294))  # 281-293 inclusive


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--workers",    type=int,   default=2)
    p.add_argument("--resume",     type=str,   default=None,
                   help="path to checkpoint to resume from")
    return p.parse_args()


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_cat_indices(dataset):
    # ImageFolder remaps folder names to 0-based indices
    # cat folders are named '281'-'293', mapped to whatever ImageFolder assigns
    cat_mapped = {v for k, v in dataset.class_to_idx.items() if k in [str(l) for l in CAT_LABELS]}
    return cat_mapped


def get_cat_subset(dataset, cat_mapped):
    indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl in cat_mapped]
    return Subset(dataset, indices)


def save_checkpoint(model, optimizer, epoch, baseline_all, args):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"ckpt_epoch{epoch}_bs{args.batch_size}_lr{args.lr}.pt"
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "baseline_all": baseline_all,
        "args":         vars(args),
    }, path)
    print(f"  checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["baseline_all"]


def evaluate(model, loader, cat_mapped, device):
    model.eval()
    correct_cat = total_cat = correct_all = total_all = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            preds = out.argmax(dim=1)
            correct_all += (preds == labels).sum().item()
            total_all   += labels.size(0)
            cat_mask = torch.tensor([l.item() in cat_mapped for l in labels], device=device)
            if cat_mask.any():
                correct_cat += (preds[cat_mask] == labels[cat_mask]).sum().item()
                total_cat   += cat_mask.sum().item()
    acc_all = correct_all / total_all * 100
    acc_cat = correct_cat / total_cat * 100 if total_cat else 0
    return acc_all, acc_cat


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"batch_size={args.batch_size}  lr={args.lr}  epochs={args.epochs}")

    train_tf, val_tf = get_transforms()

    # load datasets
    train_ds_full = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds        = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    cat_mapped    = get_cat_indices(val_ds)  # use val since it has all 1000 classes
    train_ds_cat  = get_cat_subset(train_ds_full, set(range(len(train_ds_full.classes))))

    cat_mapped    = get_cat_indices(val_ds)
    train_ds_cat  = get_cat_subset(train_ds_full, set(range(len(train_ds_full.classes))))

    train_loader = DataLoader(train_ds_cat, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    print(f"cat training images : {len(train_ds_cat)}")
    print(f"val images          : {len(val_ds)}")

    # load pretrained resnet50
    model     = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # resume from checkpoint if provided
    start_epoch  = 0
    baseline_all = None
    if args.resume:
        start_epoch, baseline_all = load_checkpoint(args.resume, model, optimizer, device)

    # baseline eval before any training
    if baseline_all is None:
        print("\n--- baseline (before training) ---")
        baseline_all, baseline_cat = evaluate(model, val_loader, cat_mapped, device)
        print(f"overall acc : {baseline_all:.2f}%")
        print(f"cat acc     : {baseline_cat:.2f}%")

    # continual learning loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()
        for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total * 100
        train_loss = total_loss / total
        elapsed    = time.time() - t0

        acc_all, acc_cat = evaluate(model, val_loader, cat_mapped, device)
        degradation = baseline_all - acc_all

        print(f"\nepoch {epoch} | loss {train_loss:.4f} | train acc {train_acc:.2f}%"
              f" | time {elapsed:.1f}s")
        print(f"  val overall : {acc_all:.2f}%  (degradation: {degradation:.3f}%)")
        print(f"  val cat     : {acc_cat:.2f}%")

        save_checkpoint(model, optimizer, epoch, baseline_all, args)

    print("\ndone.")


if __name__ == "__main__":
    main()