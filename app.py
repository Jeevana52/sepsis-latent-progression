"""
app.py
------
Flask web application for the Sepsis Risk Prediction system.

Routes:
    GET  /              → Upload page (index.html)
    POST /predict       → Run full prediction pipeline, return JSON
    GET  /result        → Show result page
    GET  /outputs/<f>   → Serve output files (SHAP plots etc.)

Run:
    python app.py
    → Open http://localhost:5000
"""

import os
import json
import io
import re
import tempfile
import traceback
import numpy as np
import torch

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, session
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "sepsis_secret_2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ── Model globals (loaded once at startup) ────────────────────────────────────
MODEL      = None
X_MEAN     = None
X_STD      = None
EXPLAINER  = None

FEATURE_COLS = [
    "heart_rate", "mean_bp", "resp_rate",
    "systolic_bp", "temperature", "lactate", "wbc"
]

FEATURE_DISPLAY = {
    "heart_rate":  "Heart Rate (bpm)",
    "mean_bp":     "Mean Blood Pressure (mmHg)",
    "resp_rate":   "Respiratory Rate (br/min)",
    "systolic_bp": "Systolic BP (mmHg)",
    "temperature": "Temperature (°C)",
    "lactate":     "Lactate (mmol/L)",
    "wbc":         "WBC Count (×10³/μL)",
}

NORMAL_RANGES = {
    "heart_rate":  (60, 100),
    "mean_bp":     (70, 100),
    "resp_rate":   (12, 20),
    "systolic_bp": (90, 140),
    "temperature": (36.0, 37.5),
    "lactate":     (0.5, 2.0),
    "wbc":         (4.0, 11.0),
}


def load_model():
    """Load trained model and normalization stats at startup."""
    global MODEL, X_MEAN, X_STD, EXPLAINER

    try:
        from src.train import SepsisEndToEnd, CONFIG
        from src.explainability import SepsisExplainer

        model_path = os.path.join("models", "best_model.pt")
        mean_path  = os.path.join("models", "feature_mean.npy")
        std_path   = os.path.join("models", "feature_std.npy")

        if os.path.exists(model_path):
            m = SepsisEndToEnd(
                struct_dim=7,
                lstm_out=CONFIG["lstm_latent_dim"],
                fused_dim=CONFIG["fused_dim"],
                latent_dim=CONFIG["latent_dim"],
            )
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            m.eval()
            MODEL = m
            print("[App] Loaded trained model from models/best_model.pt")
        else:
            print("[App] No trained model found. Will use rule-based fallback.")

        if os.path.exists(mean_path):
            X_MEAN = np.load(mean_path)
            X_STD  = np.load(std_path)

        EXPLAINER = SepsisExplainer(model=None, output_dir="outputs")
        print("[App] Explainer ready (rule-based).")

    except Exception as e:
        print(f"[App] Model load warning: {e}")
        from src.explainability import SepsisExplainer
        EXPLAINER = SepsisExplainer(model=None, output_dir="outputs")


# ── Feature extraction helpers ────────────────────────────────────────────────

def extract_features_from_form(form_data: dict) -> np.ndarray:
    """Parse form fields into a 7-dim feature array."""
    features = []
    for col in FEATURE_COLS:
        try:
            val = float(form_data.get(col, "nan"))
        except (ValueError, TypeError):
            val = float("nan")
        features.append(val)
    return np.array(features, dtype=np.float32)


def extract_features_from_text(text: str) -> np.ndarray:
    features = [float("nan")] * 7
    text = text.lower()

    patterns = {
        "heart_rate": [
            r"heart\s*rate[^0-9]*(\d+)",       # ✅ skips any words before number
            r"(?:hr|pulse)[:\s]+(\d+)",
            r"(\d+)\s*bpm",
        ],
        "mean_bp": [
            r"mean\s*bp[^0-9]*(\d+)",           # ✅ NEW — matches "mean bp (map)"
            r"(?:map|mean\s*(?:arterial|blood)\s*pressure)[^0-9]*(\d+)",
        ],
        "resp_rate": [
            r"resp(?:iratory)?\s*rate[^0-9]*(\d+)",
            r"(?:rr)[:\s]+(\d+)",
            r"(\d+)\s*br(?:eaths?)?\s*/\s*min", # ✅ now matches "br/min"
        ],
        "systolic_bp": [
            r"systolic[^0-9]*(\d+)",            # ✅ skips "bp" before number
            r"(?:sbp)[:\s]+(\d+)",
            r"bp[:\s]+(\d+)/",
        ],
        "temperature": [
            r"temp(?:erature)?[^0-9]*([\d.]+)\s*°?[cf]?",
            r"(3[678]\.\d)",
        ],
        "lactate": [
            r"lactate[^0-9]*([\d.]+)",          # ✅ handles pipe chars too
        ],
        "wbc": [
            r"wbc[^0-9]*([\d.]+)",              # ✅ skips "count" before number
            r"white\s*(?:blood\s*)?cell[^0-9]*([\d.]+)",
        ],
    }

    for i, col in enumerate(FEATURE_COLS):
        for pattern in patterns.get(col, []):
            match = re.search(pattern, text)
            if match:
                try:
                    val = float(match.group(1))
                    if col == "temperature" and val > 45:
                        val = (val - 32) * 5 / 9
                    features[i] = val
                    break
                except ValueError:
                    continue

    return np.array(features, dtype=np.float32)


def extract_features_from_csv(content: str) -> np.ndarray:
    """Parse CSV with header row containing feature column names."""
    import csv, io
    reader = csv.DictReader(io.StringIO(content))
    rows   = list(reader)
    if not rows:
        return np.full(7, float("nan"), dtype=np.float32)

    # Use last row (most recent measurement)
    row = rows[-1]
    features = []
    for col in FEATURE_COLS:
        # Try direct match and common aliases
        aliases = {
            "heart_rate":  ["heart_rate", "hr", "heartrate", "pulse"],
            "mean_bp":     ["mean_bp", "map", "meanbp", "mean_arterial"],
            "resp_rate":   ["resp_rate", "rr", "respiratory_rate"],
            "systolic_bp": ["systolic_bp", "sbp", "systolic"],
            "temperature": ["temperature", "temp"],
            "lactate":     ["lactate", "lac"],
            "wbc":         ["wbc", "white_blood_cell"],
        }
        val = float("nan")
        for alias in aliases.get(col, [col]):
            if alias in row:
                try:
                    val = float(row[alias])
                    break
                except ValueError:
                    pass
        features.append(val)

    return np.array(features, dtype=np.float32)


# ── Prediction pipeline ───────────────────────────────────────────────────────

def run_prediction(features: np.ndarray, subject_id: int = 0) -> dict:
    """
    Full prediction pipeline:
        features (7-dim) → normalize → model → risk score + stage
        → SHAP explanation

    Returns a complete result dict for the frontend.
    """
    # Fill NaN with median values (clinical defaults)
    defaults = np.array([80.0, 80.0, 16.0, 120.0, 37.0, 1.2, 8.0])
    nan_mask = np.isnan(features)
    features_clean = features.copy()
    features_clean[nan_mask] = defaults[nan_mask]

    # ── Model inference ───────────────────────────────────────────────────────
    if MODEL is not None and X_MEAN is not None:
        features_norm = (features_clean - X_MEAN) / (X_STD + 1e-8)
        x_tensor      = torch.FloatTensor(features_norm).unsqueeze(0).unsqueeze(0)

        MODEL.eval()
        with torch.no_grad():
            logits, mu, logvar, _, _ = MODEL(x_tensor)
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        stage_idx  = int(np.argmax(probs))
        risk_score = float(probs[1] * 0.5 + probs[2] * 1.0)
        prob_normal = float(probs[0])
        prob_early  = float(probs[1])
        prob_severe = float(probs[2])

    else:
        # ── Rule-based fallback ───────────────────────────────────────────────
        risk_score = 0.0
        f = features_clean
        if f[5] > 2.0: risk_score += 0.30   # lactate
        if f[0] > 100: risk_score += 0.20   # heart rate
        if f[1] < 65:  risk_score += 0.20   # mean BP
        if f[6] > 12:  risk_score += 0.10   # WBC
        if f[4] > 38.3 or f[4] < 36.0:
            risk_score += 0.10              # temperature
        if f[2] > 20:  risk_score += 0.10   # resp rate

        risk_score = min(risk_score, 1.0)
        if risk_score < 0.25:
            stage_idx = 0
        elif risk_score < 0.60:
            stage_idx = 1
        else:
            stage_idx = 2

        prob_normal = max(0, 1.0 - risk_score)
        prob_early  = risk_score * 0.6 if stage_idx == 1 else 0.1
        prob_severe = risk_score if stage_idx == 2 else 0.05

    stage_names = ["Normal", "Early Sepsis", "Severe Sepsis"]
    stage       = stage_names[stage_idx]

    # ── SHAP / rule-based explanation ─────────────────────────────────────────
    expl = EXPLAINER.explain(
        features_clean,
        subject_id=subject_id,
        risk_score=round(risk_score, 3),
        stage=stage,
    )

    # ── Build feature status for display ──────────────────────────────────────
    feature_status = []
    for i, col in enumerate(FEATURE_COLS):
        raw_val = features[i]
        val     = features_clean[i]
        lo, hi  = NORMAL_RANGES[col]
        if np.isnan(raw_val):
            status = "missing"
        elif val < lo:
            status = "low"
        elif val > hi:
            status = "high"
        else:
            status = "normal"

        feature_status.append({
            "name":    col,
            "display": FEATURE_DISPLAY[col],
            "value":   round(float(val), 2),
            "missing": bool(np.isnan(raw_val)),
            "status":  status,
            "range":   f"{lo}–{hi}",
        })

    # ── SHAP plot path (relative for serving) ─────────────────────────────────
    plot_path = expl.get("plot_path")
    plot_url  = None
    if plot_path and os.path.exists(plot_path):
        plot_url = f"/outputs/{os.path.basename(plot_path)}"

    return {
        "subject_id":       subject_id,
        "risk_score":       round(risk_score, 3),
        "risk_pct":         round(risk_score * 100, 1),
        "stage":            stage,
        "stage_idx":        stage_idx,
        "prob_normal":      round(prob_normal, 3),
        "prob_early":       round(prob_early, 3),
        "prob_severe":      round(prob_severe, 3),
        "explanation":      expl.get("explanation_text", ""),
        "top_features":     expl.get("top_features", []),
        "feature_status":   feature_status,
        "shap_plot_url":    plot_url,
        "nan_filled":       int(nan_mask.sum()),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", feature_cols=FEATURE_COLS,
                           feature_display=FEATURE_DISPLAY,
                           normal_ranges=NORMAL_RANGES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept either:
    - Form fields (manual entry)
    - Uploaded file (PDF note, CSV, or plain text)

    Returns JSON with full prediction result.
    """
    try:
        subject_id = int(request.form.get("subject_id", 0) or 0)
        input_mode = request.form.get("input_mode", "manual")

        if input_mode == "manual":
            features = extract_features_from_form(request.form)

        elif input_mode == "file":
            file = request.files.get("patient_file")
            if not file or file.filename == "":
                return jsonify({"error": "No file uploaded."}), 400

            content = file.read()
            fname   = file.filename.lower()

            if fname.endswith(".csv"):
                features = extract_features_from_csv(content.decode("utf-8", errors="ignore"))

            elif fname.endswith(".pdf"):
                # Try to extract text from PDF
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(content)) as pdf:
                        text = "\n".join(
                            p.extract_text() or "" for p in pdf.pages
                        )
                except Exception:
                    # Fallback: treat bytes as text
                    text = content.decode("utf-8", errors="ignore")
                features = extract_features_from_text(text)

            else:
                # Plain text (.txt or unknown)
                text     = content.decode("utf-8", errors="ignore")
                features = extract_features_from_text(text)

        else:
            return jsonify({"error": f"Unknown input_mode: {input_mode}"}), 400

        result = run_prediction(features, subject_id=subject_id)
        session["last_result"] = result
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/result")
def result_page():
    """Show the result page with last prediction."""
    result = session.get("last_result")
    return render_template("result.html", result=result)


@app.route("/outputs/<filename>")
def serve_output(filename):
    """Serve SHAP plots and other output files."""
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
    })


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Sepsis Risk Prediction — Web App")
    print("=" * 60)
    load_model()
    print("[App] Starting server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)