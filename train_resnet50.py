"""Train and evaluate an efficient ResNet-50 paddy nutrient classifier.

Expected data layout (the directory names are the class labels):
    dataset/images/nitrogen/*.jpg
    dataset/images/phosphorous/*.jpg
    dataset/images/potassium/*.jpg

Run: python train_resnet50.py [--epochs 20] [--batch-size 32]
Dependencies: pip install -r requirements.txt
First run downloads ImageNet weights unless --no-pretrained is specified.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASSES = ("nitrogen", "phosphorous", "potassium")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


def collect_images(root: Path):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder: {folder}")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 3:
            raise ValueError(f"Need at least 3 images in {folder}; found {len(files)}")
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.asarray(paths, dtype=object), np.asarray(labels, dtype=np.int64)


def make_loaders(paths, labels, size, batch_size, workers, seed, device):
    train_idx, remaining = train_test_split(
        np.arange(len(labels)), test_size=0.30, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(
        remaining, test_size=0.50, random_state=seed, stratify=labels[remaining])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(size + 32), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    pin = device.type == "cuda"
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin,
                  persistent_workers=workers > 0)
    train = DataLoader(LeafDataset(paths[train_idx], labels[train_idx], train_tf),
                       shuffle=True, **common)
    val = DataLoader(LeafDataset(paths[val_idx], labels[val_idx], eval_tf), **common)
    test = DataLoader(LeafDataset(paths[test_idx], labels[test_idx], eval_tf), **common)
    return train, val, test, {"train": train_idx, "validation": val_idx, "test": test_idx}


def build_model(pretrained: bool):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet50(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Pretrained weights unavailable ({exc}); using random initialization.")
        model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, len(CLASSES)))
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        count += targets.size(0)
    return total_loss / count, correct / count


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for images, targets in tqdm(loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            probs = model(images).softmax(dim=1)
        y_true.extend(targets.numpy().tolist())
        y_prob.extend(probs.float().cpu().numpy())
    return np.asarray(y_true), np.asarray(y_prob)


def save_plots(history, y_true, y_prob, out_dir: Path):
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    y_bin = label_binarize(y_true, classes=range(len(CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        axes[0].plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.03); axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["validation_loss"], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_and_training.png", dpi=180)
    plt.close(fig)
    return cm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/resnet50"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0,
                        help="Data loader workers (0 is most compatible on Windows).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the classification head for faster fine-tuning.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when supported.")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    paths, labels = collect_images(args.data)
    print(f"Images: {len(paths)} | classes: {dict(zip(CLASSES, np.bincount(labels)))} | device: {device}")
    train_loader, val_loader, test_loader, splits = make_loaders(
        paths, labels, args.image_size, args.batch_size, args.workers, args.seed, device)
    model = build_model(not args.no_pretrained).to(device)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss); history["validation_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc); history["validation_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1:02}/{args.epochs}: train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict())
    bare_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    bare_model.load_state_dict(best_state)
    artifact = {"state_dict": best_state, "classes": list(CLASSES), "image_size": args.image_size,
                "mean": MEAN, "std": STD, "architecture": "resnet50", "best_epoch": best_epoch}
    torch.save(artifact, args.output / "resnet50_paddy.pth")
    y_true, y_prob = predict(model, test_loader, device)
    cm = save_plots(history, y_true, y_prob, args.output)
    report = classification_report(y_true, y_prob.argmax(axis=1), labels=range(len(CLASSES)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    summary = {"classes": list(CLASSES), "samples": len(paths), "class_counts": dict(zip(CLASSES, map(int, np.bincount(labels)))),
               "split_sizes": {k: len(v) for k, v in splits.items()}, "best_epoch": best_epoch,
               "test_accuracy": float((y_true == y_prob.argmax(axis=1)).mean()),
               "confusion_matrix": cm.tolist(), "classification_report": report,
               "training_seconds": round(time.time() - start, 2), "device": str(device), "history": history}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch}; test accuracy: {summary['test_accuracy']:.3f}")
    print(f"Saved checkpoint, metrics, and plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
