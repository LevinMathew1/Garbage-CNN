"""Main training loop for garbage classification.

Usage:
    python src/train.py --model mobilenet_v2 --epochs 25 --batch-size 64 --lr 1e-3
"""
import argparse
import csv
import sys
import functools
from pathlib import Path

# Flush every print immediately so Colab shows live output
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import build_model, freeze_backbone, unfreeze_all
from src.utils import DEVICE, SEED, ensure_dirs, plot_training_curves, set_seed

FREEZE_EPOCHS = 5
UNFREEZE_LR = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a garbage classifier.")
    p.add_argument("--model",      default="mobilenet_v2",
                   choices=["custom_cnn", "mobilenet_v2", "convnext_tiny"])
    p.add_argument("--data-dir",   default="Garbage classification",
                   help="Path to dataset root")
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=7)
    p.add_argument("--num-workers", type=int,  default=2)
    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--img-size",   type=int,   default=224)
    return p.parse_args()


def compute_class_weights(
    loader: torch.utils.data.DataLoader, num_classes: int
) -> torch.Tensor:
    """Compute inverse-frequency class weights from train loader labels."""
    counts = torch.zeros(num_classes)
    for _, labels in loader:
        for l in labels:
            counts[l] += 1
    weights = 1.0 / counts.clamp(min=1)
    return (weights / weights.sum() * num_classes).to(DEVICE)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    phase: str,
) -> tuple[float, float, float]:
    """One train or eval epoch. Returns (loss, accuracy, macro_f1)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {phase}", leave=False,
                                  disable=not sys.stdout.isatty(), ncols=80):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train(args: argparse.Namespace) -> None:
    set_seed(SEED)
    ensure_dirs("outputs/checkpoints", "outputs/metrics", "outputs/figures")

    print(f"[TRAIN] Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    num_classes = len(class_names)
    print(f"[TRAIN] Classes ({num_classes}): {class_names}")

    # Model
    model = build_model(args.model, num_classes=num_classes).to(DEVICE)

    is_transfer = args.model in ("mobilenet_v2", "convnext_tiny")
    if is_transfer:
        freeze_backbone(model, args.model)
        print(f"[TRAIN] Backbone frozen for first {FREEZE_EPOCHS} epochs.")

    # Loss with class weights
    class_weights = compute_class_weights(train_loader, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda" if DEVICE.type == "cuda" else "cpu")

    # CSV logging
    csv_path = Path(f"outputs/metrics/{args.model}_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "train_loss", "train_acc", "train_f1",
         "val_loss", "val_acc", "val_f1", "lr"]
    )

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_f1 = 0.0
    patience_counter = 0
    best_ckpt = Path(f"outputs/checkpoints/{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        # Two-stage transfer: unfreeze after FREEZE_EPOCHS
        if is_transfer and epoch == FREEZE_EPOCHS + 1:
            unfreeze_all(model)
            for pg in optimizer.param_groups:
                pg["lr"] = UNFREEZE_LR
            print(f"[TRAIN] Epoch {epoch}: backbone unfrozen, lr → {UNFREEZE_LR}")

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scaler, "train"
        )
        vl_loss, vl_acc, vl_f1 = run_epoch(
            model, val_loader, criterion, None, scaler, "val"
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[TRAIN] Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"val loss={vl_loss:.4f} acc={vl_acc:.4f} F1={vl_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        csv_writer.writerow([epoch, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1, current_lr])
        csv_file.flush()

        for k, v in zip(
            ("train_loss", "train_acc", "val_loss", "val_acc"),
            (tr_loss, tr_acc, vl_loss, vl_acc),
        ):
            history[k].append(v)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"[TRAIN] ✓ New best val macro-F1={best_val_f1:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[TRAIN] Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    csv_file.close()

    # Save final model
    final_ckpt = Path(f"outputs/checkpoints/{args.model}_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[TRAIN] Final model saved to {final_ckpt}")

    plot_training_curves(history, args.model, f"outputs/figures/{args.model}_curves.png")
    print(f"[TRAIN] Done. Best val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train(parse_args())
