"""
app.py  —  Sepsis Risk Prediction Web App
Run:  python app.py  →  http://localhost:5000
"""

import os, sys, io, re, json, traceback
import numpy as np
import torch

# ── Always resolve src.* imports from project root ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, render_template, send_from_directory, session

app = Flask(__name__)
app.secret_key = "sepsis_secret_2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["OUTPUT_FOLDER"] = os.path.join(BASE_DIR, "outputs")
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = ["heart_rate","mean_bp","resp_rate","systolic_bp","temperature","lactate","wbc"]

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

DEFAULTS = np.array([80.0, 80.0, 16.0, 120.0, 37.0, 1.2, 8.0], dtype=np.float32)

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL     = None
X_MEAN    = None
X_STD     = None
EXPLAINER = None


def load_model():
    global MODEL, X_MEAN, X_STD, EXPLAINER
    try:
        from src.train import SepsisEndToEnd, CONFIG
        mp = os.path.join(BASE_DIR, "models", "best_model.pt")
        if os.path.exists(mp):
            m = SepsisEndToEnd(struct_dim=7, lstm_out=CONFIG["lstm_latent_dim"],
                               fused_dim=CONFIG["fused_dim"], latent_dim=CONFIG["latent_dim"])
            m.load_state_dict(torch.load(mp, map_location="cpu"))
            m.eval()
            MODEL = m
            print("[App] ✓ Trained model loaded.")
        meanp = os.path.join(BASE_DIR, "models", "feature_mean.npy")
        if os.path.exists(meanp):
            X_MEAN = np.load(meanp)
            X_STD  = np.load(os.path.join(BASE_DIR, "models", "feature_std.npy"))
            print("[App] ✓ Normalization stats loaded.")
    except Exception as e:
        print(f"[App] Model load warning: {e}")

    try:
        from src.explainability import SepsisExplainer
        EXPLAINER = SepsisExplainer(model=None, output_dir=app.config["OUTPUT_FOLDER"])
        print("[App] ✓ Explainer ready.")
    except Exception as e:
        print(f"[App] Explainer warning: {e}")


# ── Feature extraction ────────────────────────────────────────────────────────

PLAUSIBLE = {
    "heart_rate":  (20, 300), "mean_bp":     (20, 200),
    "resp_rate":   (4,  60),  "systolic_bp": (50, 300),
    "temperature": (30, 45),  "lactate":     (0.1, 30),
    "wbc":         (0.1, 100),
}

def _ok(col, val):
    lo, hi = PLAUSIBLE[col]
    return lo <= val <= hi

def extract_features_from_form(form) -> np.ndarray:
    out = []
    for col in FEATURE_COLS:
        raw = form.get(col, "").strip()
        try:   val = float(raw) if raw else float("nan")
        except: val = float("nan")
        out.append(val)
    return np.array(out, dtype=np.float32)

def extract_features_from_text(text: str) -> np.ndarray:
    """
    Handles BOTH formats:
      • Table format:  "Heart Rate 118 BPM 60-100 BPM CRITICAL"
      • Label format:  "HR: 142"  /  "MAP: 59"  /  "Lactate: 3.1"
    """
    feats = [float("nan")] * 7
    t = text.lower()

    patterns = {
        "heart_rate": [
            r"heart\s*rate\s+(\d{2,3})\s*(?:bpm|/min|beats)",   # table
            r"(?:^|\s)hr[:\s=]+(\d{2,3})",                       # HR: 142
            r"pulse[:\s=]+(\d{2,3})",
            r"(\d{2,3})\s*bpm",
        ],
        "mean_bp": [
            r"mean\s*bp\s*\(?map\)?\s+(\d{2,3})\s*mmhg",        # table
            r"(?:map|mean\s*(?:arterial|bp))[:\s=\(]+(\d{2,3})",
        ],
        "resp_rate": [
            r"respiratory\s*rate\s+(\d{1,2})\s*(?:br|breaths)",  # table
            r"(?:rr|resp(?:iratory)?\s*rate)[:\s=]+(\d{1,2})",
            r"(\d{1,2})\s*(?:breaths?|resp(?:irations?)?)\s*(?:per\s*)?(?:min|minute)",
        ],
        "systolic_bp": [
            r"systolic\s*bp\s+(\d{2,3})\s*mmhg",                 # table
            r"(?:sbp|systolic)[:\s=]+(\d{2,3})",
            r"bp[:\s=]+(\d{2,3})\s*/",
            r"(\d{2,3})\s*/\s*\d{2,3}\s*mm",
            r"blood\s*pressure[:\s=]+(\d{2,3})\s*/",
        ],
        "temperature": [
            r"temperature\s+(3[5-9]\.\d|4[0-2]\.\d)\s*°?\s*c",  # table °C
            r"temp(?:erature)?[:\s=]+(3[5-9]\.\d|4[0-2]\.\d)",
            r"t\s*[=:]\s*(3[5-9]\.\d|4[0-2]\.\d)",
            r"(3[5-9]\.\d)\s*°?\s*c\b",
            r"(10[0-4]\.\d)\s*°?\s*f\b",                         # Fahrenheit
        ],
        "lactate": [
            r"lactate\s+([\d.]+)\s*mmol",                         # table
            r"lactate[:\s=]+([\d.]+)",
            r"lactic\s*acid[:\s=]+([\d.]+)",
        ],
        "wbc": [
            r"wbc\s*count\s+([\d.]+)\s*(?:k|x10)",               # table
            r"(?:wbc|white\s*blood\s*cell(?:\s*count)?)[:\s=]+([\d.]+)",
            r"leukocytes?[:\s=]+([\d.]+)",
        ],
    }

    for i, col in enumerate(FEATURE_COLS):
        for pat in patterns[col]:
            m = re.search(pat, t)
            if m:
                try:
                    val = float(m.group(1))
                    if col == "temperature" and val > 45:
                        val = round((val - 32) * 5 / 9, 1)
                    if _ok(col, val):
                        feats[i] = val
                        break
                except ValueError:
                    continue

    return np.array(feats, dtype=np.float32)


def extract_features_from_csv(content: str) -> np.ndarray:
    import csv
    ALIASES = {
        "heart_rate":  ["heart_rate","hr","heartrate","pulse","heart rate"],
        "mean_bp":     ["mean_bp","map","meanbp","mean_arterial","mean bp","mean arterial pressure"],
        "resp_rate":   ["resp_rate","rr","respiratory_rate","resp rate","respiratory rate"],
        "systolic_bp": ["systolic_bp","sbp","systolic","systolic bp"],
        "temperature": ["temperature","temp","body_temp"],
        "lactate":     ["lactate","lac","lactic_acid","lactic acid"],
        "wbc":         ["wbc","white_blood_cell","white blood cell","leukocytes","wbc count"],
    }
    try:
        rows = list(csv.DictReader(io.StringIO(content)))
    except Exception:
        return np.full(7, float("nan"), dtype=np.float32)
    if not rows:
        return np.full(7, float("nan"), dtype=np.float32)

    row = {k.strip().lower(): v for k, v in rows[-1].items()}
    feats = []
    for col in FEATURE_COLS:
        val = float("nan")
        for alias in ALIASES[col]:
            if alias in row:
                try:
                    v = float(row[alias])
                    if _ok(col, v):
                        val = v; break
                except: pass
        feats.append(val)
    return np.array(feats, dtype=np.float32)


def _extract_pdf_text(content: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            t = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if t.strip(): return t
    except Exception: pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        t = "\n".join(p.extract_text() or "" for p in reader.pages)
        if t.strip(): return t
    except Exception: pass
    return content.decode("utf-8", errors="ignore")


# ── Prediction ────────────────────────────────────────────────────────────────

def run_prediction(features: np.ndarray, subject_id: int = 0) -> dict:
    nan_mask = np.isnan(features)
    fc = features.copy()
    fc[nan_mask] = DEFAULTS[nan_mask]

    used_model = False
    risk_score = prob_normal = prob_early = prob_severe = 0.0
    stage_idx  = 0

    # ── Try trained model ─────────────────────────────────────────────────────
    if MODEL is not None and X_MEAN is not None:
        try:
            fn = (fc - X_MEAN) / (X_STD + 1e-8)
            xt = torch.FloatTensor(fn).unsqueeze(0).unsqueeze(0)
            MODEL.eval()
            with torch.no_grad():
                logits, *_ = MODEL(xt)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            stage_idx   = int(np.argmax(probs))
            risk_score  = float(probs[1]*0.5 + probs[2]*1.0)
            prob_normal = float(probs[0])
            prob_early  = float(probs[1])
            prob_severe = float(probs[2])
            used_model  = True
            print(f"[Predict] Model: stage={stage_idx} risk={risk_score:.2f} probs={probs}")
        except Exception as e:
            print(f"[Predict] Model error: {e}")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    if not used_model:
        s = 0.0
        # Lactate — strongest sepsis marker
        if   fc[5] > 4.0:  s += 0.45
        elif fc[5] > 2.0:  s += 0.30
        elif fc[5] > 1.5:  s += 0.10
        # Hypotension (mean BP)
        if   fc[1] < 55:   s += 0.25
        elif fc[1] < 65:   s += 0.15
        # Tachycardia
        if   fc[0] > 130:  s += 0.25
        elif fc[0] > 100:  s += 0.15
        # Tachypnea
        if   fc[2] > 28:   s += 0.15
        elif fc[2] > 20:   s += 0.08
        # WBC
        if   fc[6] > 18:   s += 0.12
        elif fc[6] > 12:   s += 0.06
        elif fc[6] < 4:    s += 0.06
        # Temperature
        if   fc[4] > 39.5: s += 0.12
        elif fc[4] > 38.3: s += 0.06
        elif fc[4] < 36.0: s += 0.06
        # Low systolic BP
        if   fc[3] < 80:   s += 0.10
        elif fc[3] < 90:   s += 0.05

        risk_score = min(float(s), 1.0)
        if   risk_score >= 0.55: stage_idx = 2
        elif risk_score >= 0.20: stage_idx = 1
        else:                    stage_idx = 0

        if stage_idx == 2:
            prob_severe = min(risk_score, 0.97)
            prob_early  = (1-prob_severe)*0.35
            prob_normal = (1-prob_severe)*0.65
        elif stage_idx == 1:
            prob_early  = min(risk_score*0.85, 0.97)
            prob_severe = risk_score*0.15
            prob_normal = max(0, 1-prob_early-prob_severe)
        else:
            prob_normal = max(0, 1-risk_score)
            prob_early  = risk_score*0.65
            prob_severe = risk_score*0.35

        print(f"[Predict] Rule-based: s={s:.2f} stage={stage_idx} risk={risk_score:.2f}")

    stage_names = ["Normal", "Early Sepsis", "Severe Sepsis"]
    stage       = stage_names[stage_idx]

    # ── Explanation ───────────────────────────────────────────────────────────
    expl = {"explanation_text": "", "top_features": [], "plot_path": None}
    if EXPLAINER:
        try:
            expl = EXPLAINER.explain(fc, subject_id=subject_id,
                                     risk_score=round(risk_score,3), stage=stage)
        except Exception as e:
            print(f"[Predict] Explainer error: {e}")

    # ── Feature status ────────────────────────────────────────────────────────
    feat_status = []
    for i, col in enumerate(FEATURE_COLS):
        rv  = features[i]
        val = fc[i]
        lo, hi = NORMAL_RANGES[col]
        if np.isnan(rv):       status = "missing"
        elif val < lo:         status = "low"
        elif val > hi:         status = "high"
        else:                  status = "normal"
        feat_status.append({
            "name": col, "display": FEATURE_DISPLAY[col],
            "value": round(float(val), 2), "missing": bool(np.isnan(rv)),
            "status": status, "range": f"{lo}–{hi}",
        })

    plot_url = None
    pp = expl.get("plot_path")
    if pp and os.path.exists(str(pp)):
        plot_url = f"/outputs/{os.path.basename(pp)}"

    return {
        "subject_id":     subject_id,
        "risk_score":     round(risk_score, 3),
        "risk_pct":       round(risk_score * 100, 1),
        "stage":          stage,
        "stage_idx":      stage_idx,
        "prob_normal":    round(prob_normal,  3),
        "prob_early":     round(prob_early,   3),
        "prob_severe":    round(prob_severe,  3),
        "explanation":    expl.get("explanation_text", ""),
        "top_features":   expl.get("top_features", []),
        "feature_status": feat_status,
        "shap_plot_url":  plot_url,
        "nan_filled":     int(nan_mask.sum()),
        "used_model":     used_model,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", feature_cols=FEATURE_COLS,
                           feature_display=FEATURE_DISPLAY, normal_ranges=NORMAL_RANGES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        subject_id = int(request.form.get("subject_id", 0) or 0)
        mode       = request.form.get("input_mode", "manual")

        if mode == "manual":
            features = extract_features_from_form(request.form)
            print(f"[Predict] Manual: {list(zip(FEATURE_COLS, features))}")

        elif mode == "file":
            f = request.files.get("patient_file")
            if not f or f.filename == "":
                return jsonify({"error": "No file uploaded."}), 400
            content = f.read()
            fname   = f.filename.lower()
            print(f"[Predict] File: {fname} ({len(content)} bytes)")

            if fname.endswith(".csv"):
                features = extract_features_from_csv(content.decode("utf-8", errors="ignore"))
            elif fname.endswith(".pdf"):
                text     = _extract_pdf_text(content)
                print(f"[Predict] PDF text snippet: {text[:200]}")
                features = extract_features_from_text(text)
            else:
                text     = content.decode("utf-8", errors="ignore")
                features = extract_features_from_text(text)

            print(f"[Predict] Extracted: {list(zip(FEATURE_COLS, features))}")
        else:
            return jsonify({"error": "Unknown input_mode"}), 400

        result = run_prediction(features, subject_id=subject_id)
        session["last_result"] = result
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/result")
def result_page():
    return render_template("result.html", result=session.get("last_result"))

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    print("=" * 60)
    print("Sepsis Risk Prediction — Web App")
    print("=" * 60)
    load_model()
    print("[App] Starting at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)