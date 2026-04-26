# python src/evaluate.py --model mobilenet_v2 --data-dir "Garbage classification"
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import functools

from tqdm import tqdm

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import build_model
from src.utils import DEVICE, SEED, ensure_dirs, plot_training_curves, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained garbage classifier.")
    p.add_argument("--model",    default="mobilenet_v2",
                   choices=["custom_cnn", "mobilenet_v2", "convnext_tiny"])
    p.add_argument("--data-dir", default="Garbage classification")
    p.add_argument("--ckpt",     default=None, help="Override checkpoint path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def _run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  inference", disable=sys.stdout.isatty(), ncols=80):
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str | Path,
) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix (row-normalised)")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] Confusion matrix saved to {save_path}")


def safety_check(cm: np.ndarray, class_names: list[str]) -> None:
    # flag recyclables being sent to trash (metal/glass/plastic → trash)
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    pairs = [("battery", "metal"), ("battery", "trash"), ("metal", "trash"), ("glass", "trash"), ("plastic", "trash")]
    for true_cls, pred_cls in pairs:
        if true_cls not in name_to_idx or pred_cls not in name_to_idx:
            continue
        ti, pi = name_to_idx[true_cls], name_to_idx[pred_cls]
        row_total = cm[ti].sum()
        rate = cm[ti, pi] / max(row_total, 1)
        msg = f"[EVAL] {true_cls}→{pred_cls} misclassification rate: {rate:.2%}"
        if rate > 0.05:
            print(f"WARNING: {msg} (exceeds 5% threshold!)")
        else:
            print(msg)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(SEED)
    ensure_dirs("outputs/figures", "outputs/metrics")

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    model = build_model(args.model, num_classes=num_classes, pretrained=False).to(DEVICE)

    ckpt_path = args.ckpt or f"outputs/checkpoints/{args.model}_best.pt"
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"[EVAL] Checkpoint not found: {ckpt_path}\n"
            "Train the model first with src/train.py."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[EVAL] Loaded checkpoint: {ckpt_path}")

    labels, preds = _run_inference(model, test_loader)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\n[EVAL] Overall Accuracy : {acc:.4f}")
    print(f"[EVAL] Macro F1         : {macro_f1:.4f}")
    print(f"[EVAL] Weighted F1      : {weighted_f1:.4f}\n")

    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    print("[EVAL] Per-class report:\n", report)

    # Save per-class CSV
    report_dict = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = [
        {"class": cls, **metrics}
        for cls, metrics in report_dict.items()
        if cls in class_names
    ]
    df = pd.DataFrame(rows)
    metrics_csv = f"outputs/metrics/{args.model}_per_class.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[EVAL] Per-class metrics saved to {metrics_csv}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, args.model,
                          f"outputs/figures/{args.model}_confusion_matrix.png")

    safety_check(cm, class_names)

    # Save summary metrics
    summary_path = f"outputs/metrics/{args.model}_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "num_classes": num_classes,
            "classes": class_names,
        }, f, indent=2)
    print(f"[EVAL] Summary saved to {summary_path}")

    # Regenerate training curves from history CSV if available
    hist_csv = Path(f"outputs/metrics/{args.model}_history.csv")
    if hist_csv.exists():
        import csv
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        with open(hist_csv) as f:
            for row in csv.DictReader(f):
                hist["train_loss"].append(float(row["train_loss"]))
                hist["train_acc"].append(float(row["train_acc"]))
                hist["val_loss"].append(float(row["val_loss"]))
                hist["val_acc"].append(float(row["val_acc"]))
        plot_training_curves(hist, args.model, f"outputs/figures/{args.model}_curves.png")


if __name__ == "__main__":
    evaluate(parse_args())
