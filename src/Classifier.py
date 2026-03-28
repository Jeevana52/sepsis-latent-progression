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
    Generate pseudo-labels for training using Sepsis-3 inspired thresholds.
    Used when you don't have ground truth sepsis labels.

    Criteria
    --------
    Severe Sepsis  : lactate > 2.0  AND (mean_bp < 65  OR heart_rate > 120)
    Early Sepsis   : (lactate > 1.5 OR heart_rate > 100 OR resp_rate > 20)
                     AND NOT severe
    Normal         : everything else

    Parameters
    ----------
    df : pd.DataFrame with columns heart_rate, mean_bp, resp_rate, lactate, wbc

    Returns
    -------
    labels : np.ndarray of ints (0=Normal, 1=Early, 2=Severe)
    """
    import pandas as pd

    labels = np.zeros(len(df), dtype=int)

    # Fill NaN with neutral values so thresholds don't trigger on missing data
    hr  = df["heart_rate"].fillna(80)
    mbp = df["mean_bp"].fillna(80)
    rr  = df["resp_rate"].fillna(16)
    lac = df["lactate"].fillna(1.0)
    wbc = df["wbc"].fillna(8.0)
    tmp = df["temperature"].fillna(37.0)

    # Severe Sepsis (class 2)
    severe_mask = (
        (lac > 2.0) & ((mbp < 65) | (hr > 120))
    )

    # Early Sepsis (class 1)
    early_mask = (
        (lac > 1.5) | (hr > 100) | (rr > 20) |
        (tmp > 38.3) | (tmp < 36.0) | (wbc > 12) | (wbc < 4)
    ) & ~severe_mask

    labels[early_mask]  = 1
    labels[severe_mask] = 2

    counts = np.bincount(labels, minlength=3)
    print(f"[Labels] Normal: {counts[0]}  |  Early Sepsis: {counts[1]}  |  Severe Sepsis: {counts[2]}")
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