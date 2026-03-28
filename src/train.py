"""
train.py
--------
End-to-end training of:
    LSTM encoder → Fusion → VAE → MLP Classifier

Pipeline:
    Structured features (7-dim)
        → LSTM encoder    → latent_struct (32-dim)
    Text features (768-dim ClinicalBERT)
        → [reused or random init for training]
    Fusion: concat → 800-dim
        → VAE encoder     → mu (64-dim), logvar (64-dim)
        → MLP Classifier  → 3-class logits

Training objective:
    Total loss = CrossEntropy(logits, labels)
               + β * KL_divergence(mu, logvar)
               + Reconstruction_loss(x_hat, x_fused)

Run:
    python -m src.train
    or
    python src/train.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Local imports ─────────────────────────────────────────────────────────────
from src.classifier import SepsisClassifier, generate_labels_from_features, STAGE_LABELS
from src.vae import VAE

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path":        "data/processed/final_structured.csv",
    "model_dir":        "models",
    "output_dir":       "outputs",
    "latent_dim":       64,
    "lstm_latent_dim":  32,
    "fused_dim":        800,   # 32 (LSTM) + 768 (BERT) — we simulate BERT part
    "struct_dim":       7,     # number of structured features
    "epochs":           20,
    "batch_size":       256,
    "learning_rate":    1e-3,
    "beta_kl":          0.001, # KL weight (small = let classification dominate)
    "test_split":       0.2,
    "random_seed":      42,
    "max_rows":         50000, # subset for fast training
}

FEATURE_COLS = [
    "heart_rate", "mean_bp", "resp_rate",
    "systolic_bp", "temperature", "lactate", "wbc"
]


# ── Simple inline LSTM encoder (matches model_lstm.py interface) ───────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, 1, 7)  — single timestep
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))  # (batch, 32)


# ── Full end-to-end pipeline model ───────────────────────────────────────────

class SepsisEndToEnd(nn.Module):
    """
    LSTM → Fusion → VAE → Classifier pipeline.
    
    For training we simulate the text embedding with a learned projection
    so the full 800-dim fusion can be trained without running ClinicalBERT
    on every batch (would require GPU and all notes).
    """

    def __init__(self, struct_dim=7, lstm_out=32, text_sim_dim=768,
                 fused_dim=800, latent_dim=64):
        super().__init__()

        # Structured encoder
        self.lstm_encoder = LSTMEncoder(
            input_dim=struct_dim,
            hidden_dim=64,
            latent_dim=lstm_out
        )

        # Text simulation: learn a 768-dim projection from struct features
        # (replaces ClinicalBERT at training time for speed)
        self.text_sim = nn.Sequential(
            nn.Linear(struct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, text_sim_dim),
        )

        # VAE: encoder
        self.vae_encoder_mu     = nn.Linear(fused_dim, latent_dim)
        self.vae_encoder_logvar = nn.Linear(fused_dim, latent_dim)

        # VAE: decoder
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fused_dim),
        )

        # MLP Classifier on latent mu
        self.classifier = SepsisClassifier(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_struct):
        """
        x_struct : (batch, 1, 7)  — structured features, 1 timestep

        Returns
        -------
        logits   : (batch, 3)
        mu       : (batch, 64)
        logvar   : (batch, 64)
        x_hat    : (batch, 800)   reconstructed fused vector
        x_fused  : (batch, 800)   original fused vector
        """
        # Structured branch
        lstm_out = self.lstm_encoder(x_struct)          # (batch, 32)

        # Text simulation branch
        x_flat   = x_struct.squeeze(1)                  # (batch, 7)
        text_out = self.text_sim(x_flat)                 # (batch, 768)

        # Fusion
        x_fused  = torch.cat([lstm_out, text_out], dim=-1)  # (batch, 800)

        # VAE encode
        mu     = self.vae_encoder_mu(x_fused)
        logvar = self.vae_encoder_logvar(x_fused)
        z      = self.reparameterize(mu, logvar)

        # Decode
        x_hat  = self.vae_decoder(z)

        # Classify from mu (deterministic, not sampled)
        logits = self.classifier(mu)

        return logits, mu, logvar, x_hat, x_fused


# ── Loss function ─────────────────────────────────────────────────────────────

def total_loss(logits, labels, x_hat, x_fused, mu, logvar, beta=0.001):
    # Classification loss
    ce_loss   = nn.CrossEntropyLoss()(logits, labels)

    # Reconstruction loss (MSE on fused vector)
    recon     = nn.MSELoss()(x_hat, x_fused)

    # KL divergence
    kl_loss   = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return ce_loss + recon * 0.01 + beta * kl_loss, ce_loss, recon, kl_loss


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(config):
    import pandas as pd

    path = config["data_path"]
    if not os.path.exists(path):
        print(f"[Train] Data not found at {path}")
        print("[Train] Generating synthetic data for demo...")
        return _synthetic_data(config)

    print(f"[Train] Loading data from {path}")
    df = pd.read_csv(path, nrows=config["max_rows"])

    # Keep only feature columns (drop any missing)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(
        lambda col: col.fillna(col.median())
    )

    X = df[FEATURE_COLS].values.astype(np.float32)

    # Normalize features (z-score)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Generate labels
    y = generate_labels_from_features(df)

    print(f"[Train] Data shape: {X_norm.shape}, Labels: {np.bincount(y)}")
    return X_norm, y, X_mean, X_std


def _synthetic_data(config):
    """Generate synthetic ICU-like data for demo when CSV not present."""
    np.random.seed(config["random_seed"])
    n = 5000

    # Normal patients
    X_normal = np.random.randn(n // 3, 7) * np.array(
        [10, 8, 3, 10, 0.5, 0.3, 2]
    ) + np.array([75, 80, 16, 120, 37.0, 1.0, 7.0])

    # Early sepsis
    X_early = np.random.randn(n // 3, 7) * np.array(
        [12, 10, 4, 12, 0.8, 0.5, 3]
    ) + np.array([105, 70, 22, 100, 38.5, 1.8, 11.0])

    # Severe sepsis
    X_severe = np.random.randn(n // 3, 7) * np.array(
        [15, 8, 5, 15, 1.0, 0.8, 4]
    ) + np.array([130, 55, 28, 85, 39.2, 3.5, 15.0])

    X = np.vstack([X_normal, X_early, X_severe]).astype(np.float32)
    y = np.array(
        [0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3)
    )

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    print(f"[Train] Synthetic data: {X_norm.shape}, Labels: {np.bincount(y)}")
    return X_norm, y, X_mean, X_std


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config=None):
    if config is None:
        config = CONFIG

    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    os.makedirs(config["model_dir"],  exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    result = load_data(config)
    X_norm, y, X_mean, X_std = result

    # Save normalization stats for inference
    np.save(os.path.join(config["model_dir"], "feature_mean.npy"), X_mean)
    np.save(os.path.join(config["model_dir"], "feature_std.npy"),  X_std)

    # Tensors
    X_tensor = torch.FloatTensor(X_norm).unsqueeze(1)  # (N, 1, 7)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    test_size  = int(len(dataset) * config["test_split"])
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SepsisEndToEnd(
        struct_dim=7,
        lstm_out=config["lstm_latent_dim"],
        fused_dim=config["fused_dim"],
        latent_dim=config["latent_dim"],
    )
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    print(f"\n[Train] Starting training — {config['epochs']} epochs")
    print(f"[Train] Train: {train_size} samples | Test: {test_size} samples")
    print("-" * 60)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_f1 = 0.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total, correct = 0, 0
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits, mu, logvar, x_hat, x_fused = model(X_batch)
            loss, ce, recon, kl = total_loss(
                logits, y_batch, x_hat, x_fused, mu, logvar,
                beta=config["beta_kl"]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct   += (preds == y_batch).sum().item()
            total     += len(y_batch)
            epoch_loss += loss.item()

        scheduler.step()

        train_acc  = correct / total
        avg_loss   = epoch_loss / len(train_loader)

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits, *_ = model(X_batch)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y_batch.numpy())

        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Test Acc: {test_acc:.3f} | "
              f"Macro F1: {f1:.3f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),
                       os.path.join(config["model_dir"], "best_model.pt"))

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(
        torch.load(os.path.join(config["model_dir"], "best_model.pt"),
                   map_location="cpu")
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits, *_ = model(X_batch)
            all_preds.extend(logits.argmax(dim=-1).numpy())
            all_labels.extend(y_batch.numpy())

    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2],
        target_names=list(STAGE_LABELS.values()),
        zero_division=0
    )
    print(report)

    # Save report
    report_path = os.path.join(config["output_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[Train] Saved classification report → {report_path}")

    # Save metrics JSON
    metrics = {
        "best_macro_f1": best_f1,
        "final_test_acc": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "history": history,
    }
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save full model for inference
    torch.save(model.state_dict(),
               os.path.join(config["model_dir"], "sepsis_model.pt"))
    print(f"[Train] Saved final model → models/sepsis_model.pt")
    print(f"[Train] Best Macro F1: {best_f1:.4f}")

    return model, X_mean, X_std


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()