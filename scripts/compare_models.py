"""Aggregate results from all trained models and produce a comparison report.

Usage:
    python scripts/compare_models.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


MODELS = ["custom_cnn", "mobilenet_v2", "convnext_tiny"]
METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")


def _load_summary(model: str) -> dict | None:
    p = METRICS_DIR / f"{model}_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _battery_metal_rate(model: str) -> str:
    """Read metal→trash rate from per-class CSV if available."""
    p = METRICS_DIR / f"{model}_per_class.csv"
    if not p.exists():
        return "N/A"
    # Confusion matrix is needed for the exact rate; approximate from recall
    return "see confusion matrix"


def _param_count(model: str) -> str:
    """Return human-readable param count by importing model."""
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models import build_model
        m = build_model(model, num_classes=6, pretrained=False)
        n = sum(p.numel() for p in m.parameters())
        return f"{n / 1e6:.2f}M"
    except Exception:
        return "N/A"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        summary = _load_summary(model)
        if summary is None:
            print(f"[COMPARE] No summary found for {model}, skipping.")
            continue
        rows.append({
            "Model": model,
            "Params": _param_count(model),
            "Test Acc": f"{summary['accuracy']:.4f}",
            "Test Macro-F1": f"{summary['macro_f1']:.4f}",
            "Weighted F1": f"{summary['weighted_f1']:.4f}",
            "Metal→Trash Error": _battery_metal_rate(model),
        })

    if not rows:
        print("[COMPARE] No model summaries found. Run evaluate.py for each model first.")
        return

    df = pd.DataFrame(rows)
    print("\n[COMPARE] Model Comparison:\n")
    print(df.to_string(index=False))

    # Markdown table
    md_path = METRICS_DIR / "comparison.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"\n[COMPARE] Markdown table saved to {md_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    model_names = df["Model"].tolist()
    f1_values = [float(v) for v in df["Test Macro-F1"].tolist()]
    acc_values = [float(v) for v in df["Test Acc"].tolist()]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width / 2, acc_values, width, label="Test Accuracy")
    ax.bar(x + width / 2, f1_values, width, label="Test Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Accuracy & Macro F1")
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[COMPARE] Bar chart saved to {out_path}")


if __name__ == "__main__":
    main()
