"""
classifier.py
-------------
Learned MLP classifier on top of VAE latent (mu) vectors.
Replaces the rule-based scoring in demo_pipeline.py.

Architecture:
    VAE latent mu (64-dim)
        → FC(64 → 32) → ReLU → Dropout(0.3)
        → FC(32 → 16) → ReLU
        → FC(16 → 3)  → Softmax
    Output: probability over [Normal, Early Sepsis, Severe Sepsis]

Usage:
    from src.classifier import SepsisClassifier, STAGE_LABELS
    clf = SepsisClassifier()
    logits = clf(latent_mu)          # (batch, 3)
    probs  = torch.softmax(logits, dim=-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Labels ───────────────────────────────────────────────────────────────────
STAGE_LABELS = {
    0: "Normal",
    1: "Early Sepsis",
    2: "Severe Sepsis"
}

STAGE_COLORS = {
    "Normal":        "#22c55e",   # green
    "Early Sepsis":  "#f97316",   # orange
    "Severe Sepsis": "#ef4444",   # red
}


# ── Model Definition ─────────────────────────────────────────────────────────

class SepsisClassifier(nn.Module):
    """
    3-class MLP classifier for sepsis staging.

    Input  : latent mu vector from VAE encoder (batch, latent_dim=64)
    Output : raw logits (batch, 3)

    Parameters
    ----------
    latent_dim : int
        Must match VAE latent_dim (default 64).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(self, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(32, 16),
            nn.ReLU(),

            # Output layer
            nn.Linear(16, 3),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, latent_dim)

        Returns
        -------
        logits : torch.Tensor, shape (batch, 3)
        """
        return self.net(x)

    def predict(self, x: torch.Tensor):
        """
        Returns predicted class index and probabilities.

        Returns
        -------
        class_idx : torch.Tensor, shape (batch,)
        probs     : torch.Tensor, shape (batch, 3)
        stage_labels : list[str]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            class_idx = torch.argmax(probs, dim=-1)
        labels = [STAGE_LABELS[i.item()] for i in class_idx]
        return class_idx, probs, labels

    def predict_single(self, latent_mu: np.ndarray):
        """
        Convenience wrapper for a single patient numpy array.

        Parameters
        ----------
        latent_mu : np.ndarray, shape (64,)

        Returns
        -------
        dict with stage, risk_score, probs
        """
        x = torch.FloatTensor(latent_mu).unsqueeze(0)  # (1, 64)
        idx, probs, labels = self.predict(x)

        stage_idx = idx.item()
        prob_vec  = probs[0].numpy()

        # Risk score = weighted combination of probabilities
        # Normal=0, Early=0.5, Severe=1.0
        risk_score = float(
            prob_vec[0] * 0.0 +
            prob_vec[1] * 0.5 +
            prob_vec[2] * 1.0
        )

        return {
            "stage_idx":   stage_idx,
            "stage":       STAGE_LABELS[stage_idx],
            "risk_score":  round(risk_score, 3),
            "prob_normal": round(float(prob_vec[0]), 3),
            "prob_early":  round(float(prob_vec[1]), 3),
            "prob_severe": round(float(prob_vec[2]), 3),
        }


# ── Label generation from clinical thresholds (for training) ─────────────────

def generate_labels_from_features(df) -> np.ndarray:
    """
    Generate pseudo-labels for training using a composite severity score
    with percentile-based splitting.

    Strategy
    --------
    ICU data is inherently sick — fixed thresholds like "HR > 100" will
    label nearly every row as Early Sepsis and produce only 1 class.

    Instead we:
      1. Build a continuous severity score per row (0–6 points).
      2. Split into 3 classes by percentile:
            Bottom 40% by score  → Normal
            Middle 40%           → Early Sepsis
            Top    20%           → Severe Sepsis
      This guarantees all 3 classes are always present regardless of
      how sick the underlying cohort is.

    Score components (each contributes 0 or 1 point):
        lactate   > data 75th pct    → +2  (most discriminative)
        mean_bp   < data 25th pct    → +2
        heart_rate > data 75th pct   → +1
        resp_rate  > data 75th pct   → +1

    Parameters
    ----------
    df : pd.DataFrame with columns matching FEATURE_COLS

    Returns
    -------
    labels : np.ndarray of ints (0=Normal, 1=Early Sepsis, 2=Severe Sepsis)
    """
    # Fill NaN with column medians (safe neutral values)
    hr  = df["heart_rate"].fillna(df["heart_rate"].median())
    mbp = df["mean_bp"].fillna(df["mean_bp"].median())
    rr  = df["resp_rate"].fillna(df["resp_rate"].median())
    lac = df["lactate"].fillna(df["lactate"].median())

    # Compute percentile thresholds from the actual data distribution
    lac_75  = float(np.nanpercentile(lac, 75))
    mbp_25  = float(np.nanpercentile(mbp, 25))
    hr_75   = float(np.nanpercentile(hr,  75))
    rr_75   = float(np.nanpercentile(rr,  75))

    print(f"[Labels] Thresholds — lactate p75: {lac_75:.2f} | "
          f"map p25: {mbp_25:.1f} | hr p75: {hr_75:.1f} | rr p75: {rr_75:.1f}")

    # Build severity score
    score = (
        (lac > lac_75).astype(int) * 2 +
        (mbp < mbp_25).astype(int) * 2 +
        (hr  > hr_75).astype(int)  * 1 +
        (rr  > rr_75).astype(int)  * 1
    )

    # Percentile-based split: always produces all 3 classes
    p40 = np.percentile(score, 40)
    p80 = np.percentile(score, 80)

    labels = np.zeros(len(df), dtype=int)
    labels[score >  p40] = 1   # Early Sepsis
    labels[score >  p80] = 2   # Severe Sepsis

    counts = np.bincount(labels, minlength=3)
    print(f"[Labels] Normal: {counts[0]}  |  "
          f"Early Sepsis: {counts[1]}  |  "
          f"Severe Sepsis: {counts[2]}")

    # Sanity check — warn if any class is empty
    if 0 in counts:
        print("[Labels] WARNING: A class has 0 samples. "
              "Check your feature columns for all-NaN columns.")

    return labels


# ── Standalone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SepsisClassifier — Architecture Demo")
    print("=" * 60)

    clf = SepsisClassifier(latent_dim=64)
    print(clf)

    # Test with a random latent vector (as if from VAE)
    dummy_latent = np.random.randn(64).astype(np.float32)
    result = clf.predict_single(dummy_latent)

    print(f"\nDummy patient prediction:")
    print(f"  Stage      : {result['stage']}")
    print(f"  Risk Score : {result['risk_score']}")
    print(f"  P(Normal)  : {result['prob_normal']}")
    print(f"  P(Early)   : {result['prob_early']}")
    print(f"  P(Severe)  : {result['prob_severe']}")

    total_params = sum(p.numel() for p in clf.parameters())
    print(f"\nTotal parameters: {total_params:,}")