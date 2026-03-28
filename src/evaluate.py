"""
evaluate.py
-----------
Evaluation of the trained SepsisEndToEnd model.

Generates:
    outputs/classification_report.txt
    outputs/confusion_matrix.png
    outputs/roc_curve.png
    outputs/metrics.json

Run:
    python -m src.evaluate
    or
    python src/evaluate.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)
import warnings
warnings.filterwarnings("ignore")

STAGE_LABELS = ["Normal", "Early Sepsis", "Severe Sepsis"]
STAGE_COLORS = ["#22c55e", "#f97316", "#ef4444"]
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models"


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(STAGE_LABELS)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(STAGE_LABELS, fontsize=10, color='#e2e8f0', rotation=15)
    ax.set_yticklabels(STAGE_LABELS, fontsize=10, color='#e2e8f0')

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = 'white' if cm_norm[i, j] > 0.5 else '#e2e8f0'
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xlabel("Predicted label",  color='#9ca3af', fontsize=11)
    ax.set_ylabel("True label",       color='#9ca3af', fontsize=11)
    ax.set_title("Confusion Matrix",  color='white',   fontsize=13, pad=12)
    ax.tick_params(colors='#9ca3af')
    for spine in ax.spines.values():
        spine.set_color('#374151')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Eval] Saved confusion matrix → {save_path}")


# ── ROC Curve Plot ────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_probs, save_path):
    """
    One-vs-rest ROC curves for each class.

    y_true  : array (N,) int labels
    y_probs : array (N, 3) softmax probabilities
    """
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    for i, (label, color) in enumerate(zip(STAGE_LABELS, STAGE_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], '--', color='#6b7280', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color='#9ca3af', fontsize=11)
    ax.set_ylabel("True Positive Rate",  color='#9ca3af', fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", color='white', fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9,
              facecolor='#1f2937', edgecolor='#374151',
              labelcolor='#e2e8f0')
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, color='#1f2937', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#374151')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Eval] Saved ROC curves → {save_path}")


# ── Training History Plot ─────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values():
            spine.set_color('#374151')
        ax.grid(True, color='#1f2937', linewidth=0.5)

    # Loss
    axes[0].plot(epochs, history["train_loss"],
                 color='#3b82f6', lw=2, label='Train loss')
    axes[0].set_xlabel("Epoch", color='#9ca3af')
    axes[0].set_ylabel("Loss",  color='#9ca3af')
    axes[0].set_title("Training loss", color='white', fontsize=11)
    axes[0].legend(facecolor='#1f2937', edgecolor='#374151',
                   labelcolor='#e2e8f0', fontsize=9)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"],
                 color='#22c55e', lw=2, label='Train acc')
    axes[1].plot(epochs, history["test_acc"],
                 color='#f97316', lw=2, label='Test acc',
                 linestyle='--')
    axes[1].set_xlabel("Epoch", color='#9ca3af')
    axes[1].set_ylabel("Accuracy", color='#9ca3af')
    axes[1].set_title("Accuracy over epochs", color='white', fontsize=11)
    axes[1].legend(facecolor='#1f2937', edgecolor='#374151',
                   labelcolor='#e2e8f0', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Eval] Saved training history → {save_path}")


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate(model=None, test_loader=None, config=None):
    """
    Run full evaluation pipeline.
    Can be called from train.py after training,
    or standalone by loading saved model.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model if not passed ───────────────────────────────────────────────
    if model is None:
        model_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            print("[Eval] No saved model found. Run train.py first.")
            return

        print(f"[Eval] Loading model from {model_path}")
        from src.train import SepsisEndToEnd, CONFIG
        cfg   = config or CONFIG
        model = SepsisEndToEnd(
            struct_dim=7,
            lstm_out=cfg["lstm_latent_dim"],
            fused_dim=cfg["fused_dim"],
            latent_dim=cfg["latent_dim"],
        )
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

    # ── Run inference ──────────────────────────────────────────────────────────
    if test_loader is None:
        print("[Eval] No test_loader provided. Generating synthetic test set.")
        from src.train import _synthetic_data, CONFIG
        cfg = config or CONFIG
        X_norm, y, _, _ = _synthetic_data(cfg)

        # Use last 20% as test
        n_test = int(len(X_norm) * 0.2)
        X_test = torch.FloatTensor(X_norm[-n_test:]).unsqueeze(1)
        y_test = torch.LongTensor(y[-n_test:])

        from torch.utils.data import TensorDataset, DataLoader
        test_ds     = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=256)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits, *_ = model(X_batch)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    # ── Classification report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred,
                                   target_names=STAGE_LABELS,
                                   zero_division=0)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_true, y_probs,
        os.path.join(OUTPUT_DIR, "roc_curve.png")
    )

    # ── Training history (if metrics.json exists) ─────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "history" in metrics:
            plot_training_history(
                metrics["history"],
                os.path.join(OUTPUT_DIR, "training_history.png")
            )

    # ── Summary metrics ────────────────────────────────────────────────────────
    acc     = np.mean(y_true == y_pred)
    macro_f1= f1_score(y_true, y_pred, average="macro", zero_division=0)

    summary = {
        "accuracy":   round(float(acc), 4),
        "macro_f1":   round(float(macro_f1), 4),
        "n_test":     int(len(y_true)),
    }
    with open(os.path.join(OUTPUT_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] Accuracy : {acc:.4f}")
    print(f"[Eval] Macro F1  : {macro_f1:.4f}")
    print(f"[Eval] All plots saved to {OUTPUT_DIR}/")
    return summary


if __name__ == "__main__":
    evaluate()