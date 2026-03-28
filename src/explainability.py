"""
explainability.py
-----------------
SHAP-based feature explainability for sepsis risk prediction.
Replaces the old rule-based text explanations.

Usage:
    from src.explainability import SepsisExplainer
    explainer = SepsisExplainer(model)
    result = explainer.explain(patient_features, subject_id)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import json

# ── Try importing SHAP (graceful fallback if not installed) ──────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Run: pip install shap")

FEATURE_NAMES = [
    "heart_rate",
    "mean_bp",
    "resp_rate",
    "systolic_bp",
    "temperature",
    "lactate",
    "wbc"
]

# Clinical thresholds for fallback rule-based labels
CLINICAL_THRESHOLDS = {
    "heart_rate":   {"high": 100, "label_high": "tachycardia"},
    "mean_bp":      {"low":   65, "label_low":  "hypotension"},
    "resp_rate":    {"high":  20, "label_high": "tachypnea"},
    "systolic_bp":  {"low":   90, "label_low":  "hypotension"},
    "temperature":  {"high": 38.3, "label_high": "fever",
                     "low":  36.0, "label_low":  "hypothermia"},
    "lactate":      {"high":  2.0, "label_high": "elevated lactate"},
    "wbc":          {"high": 12.0, "label_high": "leukocytosis",
                     "low":   4.0, "label_low":  "leukopenia"},
}


class SepsisExplainer:
    """
    Generates SHAP-based or rule-based explanations for sepsis predictions.
    
    Parameters
    ----------
    model : callable
        Any model with a predict/forward method that accepts numpy arrays.
        If None, uses rule-based fallback only.
    output_dir : str
        Directory to save SHAP plots.
    """

    def __init__(self, model=None, output_dir="outputs"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.shap_explainer = None
        self._background_data = None

    # ── Setup ────────────────────────────────────────────────────────────────

    def fit_background(self, X_background: np.ndarray):
        """
        Fit SHAP KernelExplainer on background data (sample of training data).
        
        Parameters
        ----------
        X_background : np.ndarray, shape (n_samples, 7)
            A small representative sample of training features (50–200 rows).
        """
        if not SHAP_AVAILABLE:
            print("[SHAP] Not available, skipping background fit.")
            return

        if self.model is None:
            print("[SHAP] No model provided, skipping background fit.")
            return

        print(f"[SHAP] Fitting KernelExplainer on {len(X_background)} background samples...")
        self._background_data = shap.sample(X_background, min(100, len(X_background)))

        def predict_fn(X):
            """Wrapper: returns risk score (float) for each row."""
            import torch
            tensor = torch.FloatTensor(X).unsqueeze(1)
            with torch.no_grad():
                out = self.model(tensor)
            # If model returns (risk, stage) tuple
            if isinstance(out, tuple):
                out = out[0]
            return out.numpy().flatten()

        self.shap_explainer = shap.KernelExplainer(
            predict_fn,
            self._background_data
        )
        print("[SHAP] Explainer ready.")

    # ── Core Explain ─────────────────────────────────────────────────────────

    def explain(self, features: np.ndarray, subject_id: int = 0,
                risk_score: float = None, stage: str = None) -> dict:
        """
        Generate explanation for a single patient.

        Parameters
        ----------
        features : np.ndarray, shape (7,)
            One row of [heart_rate, mean_bp, resp_rate, systolic_bp,
                        temperature, lactate, wbc]
        subject_id : int
        risk_score : float, optional
        stage : str, optional

        Returns
        -------
        dict with keys:
            subject_id, shap_values, feature_names, feature_values,
            top_features, explanation_text, plot_path
        """
        features = np.array(features).flatten()
        assert len(features) == len(FEATURE_NAMES), \
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"

        result = {
            "subject_id":    subject_id,
            "risk_score":    risk_score,
            "stage":         stage,
            "feature_names": FEATURE_NAMES,
            "feature_values": features.tolist(),
            "shap_values":   None,
            "top_features":  [],
            "explanation_text": "",
            "plot_path":     None,
        }

        # ── SHAP explanation ─────────────────────────────────────────────────
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(
                features.reshape(1, -1), nsamples=100
            ).flatten()
            result["shap_values"] = shap_values.tolist()

            # Rank features by absolute SHAP value
            ranked_idx = np.argsort(np.abs(shap_values))[::-1]
            top_features = []
            for i in ranked_idx[:3]:
                direction = "↑ increases" if shap_values[i] > 0 else "↓ decreases"
                top_features.append({
                    "feature":    FEATURE_NAMES[i],
                    "value":      float(features[i]),
                    "shap":       float(shap_values[i]),
                    "direction":  direction,
                })
            result["top_features"] = top_features

            # ── Save per-patient SHAP waterfall plot ─────────────────────────
            plot_path = self._save_shap_plot(
                shap_values, features, subject_id, risk_score, stage
            )
            result["plot_path"] = plot_path

            # ── Plain English explanation ─────────────────────────────────────
            result["explanation_text"] = self._shap_to_text(top_features, stage)

        else:
            # ── Fallback: rule-based explanation ─────────────────────────────
            result["top_features"] = self._rule_based_features(features)
            result["explanation_text"] = self._rule_based_text(features, stage)

        return result

    # ── Batch Explain ────────────────────────────────────────────────────────

    def explain_batch(self, X: np.ndarray, subject_ids=None,
                      risk_scores=None, stages=None) -> list:
        """
        Explain multiple patients and save a global summary plot.
        Returns list of per-patient result dicts.
        """
        results = []
        n = len(X)
        if subject_ids is None:
            subject_ids = list(range(n))
        if risk_scores is None:
            risk_scores = [None] * n
        if stages is None:
            stages = [None] * n

        for i in range(n):
            r = self.explain(
                X[i], subject_id=subject_ids[i],
                risk_score=risk_scores[i], stage=stages[i]
            )
            results.append(r)
            print(f"[SHAP] Explained patient {subject_ids[i]} ({i+1}/{n})")

        # Save global summary if SHAP is available and we have shap_values
        if SHAP_AVAILABLE and any(r["shap_values"] for r in results):
            self._save_summary_plot(X, results)

        return results

    # ── Plot Helpers ─────────────────────────────────────────────────────────

    def _save_shap_plot(self, shap_values, features, subject_id,
                        risk_score, stage):
        """Save a horizontal bar SHAP waterfall plot for one patient."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in shap_values]
        y_pos = range(len(FEATURE_NAMES))

        bars = ax.barh(
            y_pos, shap_values,
            color=colors, edgecolor='none', height=0.6
        )

        # Value labels on bars
        for bar, val, feat_val in zip(bars, shap_values, features):
            x = bar.get_width()
            ax.text(
                x + (0.005 if x >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va='center',
                ha='left' if x >= 0 else 'right',
                fontsize=9, color='white'
            )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [f"{n}  ({features[i]:.1f})"
             for i, n in enumerate(FEATURE_NAMES)],
            fontsize=10, color='#e2e8f0'
        )
        ax.axvline(0, color='#6b7280', linewidth=0.8)
        ax.tick_params(colors='#9ca3af')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.xaxis.label.set_color('#9ca3af')

        title = f"Patient {subject_id}"
        if risk_score is not None:
            title += f"  |  Risk: {risk_score:.2f}"
        if stage:
            title += f"  |  {stage}"
        ax.set_title(title, color='white', fontsize=11, pad=10)
        ax.set_xlabel("SHAP value  (red = raises risk, blue = lowers risk)",
                      color='#9ca3af', fontsize=9)

        plt.tight_layout()
        path = os.path.join(
            self.output_dir, f"shap_explanation_{subject_id}.png"
        )
        plt.savefig(path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SHAP] Saved plot → {path}")
        return path

    def _save_summary_plot(self, X: np.ndarray, results: list):
        """Save global SHAP summary (mean |SHAP|) across all patients."""
        all_shap = np.array([
            r["shap_values"] for r in results if r["shap_values"]
        ])
        if len(all_shap) == 0:
            return

        mean_abs = np.mean(np.abs(all_shap), axis=0)
        ranked = np.argsort(mean_abs)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        colors = plt.cm.YlOrRd(
            np.linspace(0.3, 0.9, len(FEATURE_NAMES))
        )
        ax.barh(
            range(len(FEATURE_NAMES)),
            mean_abs[ranked],
            color=[colors[i] for i in range(len(FEATURE_NAMES))],
            edgecolor='none', height=0.6
        )
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_yticklabels(
            [FEATURE_NAMES[i] for i in ranked],
            fontsize=10, color='#e2e8f0'
        )
        ax.set_title("Global feature importance (mean |SHAP|)",
                     color='white', fontsize=11, pad=10)
        ax.set_xlabel("Mean |SHAP value|", color='#9ca3af', fontsize=9)
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values():
            spine.set_color('#374151')

        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SHAP] Saved global summary → {path}")

    # ── Text Generation ──────────────────────────────────────────────────────

    def _shap_to_text(self, top_features: list, stage: str = None) -> str:
        """Convert top SHAP features into a plain-English sentence."""
        if not top_features:
            return "No significant risk drivers identified."

        parts = []
        for f in top_features[:3]:
            name = f["feature"].replace("_", " ")
            val  = f["value"]
            shap = f["shap"]
            if shap > 0:
                parts.append(f"elevated {name} ({val:.1f})")
            else:
                parts.append(f"normal {name} ({val:.1f}) reducing risk")

        risk_parts = [p for p in parts if "reducing" not in p]
        safe_parts = [p for p in parts if "reducing" in p]

        sentence = ""
        if risk_parts:
            sentence = "Risk driven by: " + ", ".join(risk_parts) + "."
        if safe_parts:
            sentence += " " + " and ".join(safe_parts).capitalize() + "."

        if stage == "Severe Sepsis":
            sentence += " Immediate clinical attention recommended."
        elif stage == "Early Sepsis":
            sentence += " Close monitoring advised."

        return sentence.strip()

    def _rule_based_features(self, features: np.ndarray) -> list:
        """Fallback: return top abnormal features based on thresholds."""
        findings = []
        for i, name in enumerate(FEATURE_NAMES):
            val = features[i]
            if np.isnan(val):
                continue
            thresh = CLINICAL_THRESHOLDS.get(name, {})
            if "high" in thresh and val > thresh["high"]:
                findings.append({
                    "feature":   name,
                    "value":     float(val),
                    "shap":      0.1,
                    "direction": "↑ increases",
                    "label":     thresh["label_high"],
                })
            elif "low" in thresh and val < thresh["low"]:
                findings.append({
                    "feature":   name,
                    "value":     float(val),
                    "shap":      0.1,
                    "direction": "↑ increases",
                    "label":     thresh["label_low"],
                })
        return findings[:3]

    def _rule_based_text(self, features: np.ndarray,
                         stage: str = None) -> str:
        """Fallback plain-English text from clinical thresholds."""
        findings = self._rule_based_features(features)
        if not findings:
            return "Vitals within normal range. Continue monitoring."

        labels = [f.get("label", f["feature"]) for f in findings]
        text = "Indicators: " + " + ".join(labels) + "."
        if stage == "Severe Sepsis":
            text += " Immediate clinical attention recommended."
        elif stage == "Early Sepsis":
            text += " Close monitoring advised."
        return text


# ── Standalone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SHAP Explainability — Demo (no model, rule-based fallback)")
    print("=" * 60)

    # Simulate a high-risk patient
    demo_features = np.array([142.0, 59.0, 24.0, 85.0, 38.8, 3.1, 14.2])

    explainer = SepsisExplainer(model=None, output_dir="outputs")
    result = explainer.explain(
        demo_features,
        subject_id=10119770,
        risk_score=0.72,
        stage="Early Sepsis"
    )

    print(f"\nSubject ID  : {result['subject_id']}")
    print(f"Risk Score  : {result['risk_score']}")
    print(f"Stage       : {result['stage']}")
    print(f"Explanation : {result['explanation_text']}")
    print(f"\nTop Features:")
    for f in result["top_features"]:
        print(f"  {f['feature']:15s} = {f['value']:.1f}")
    print("\nDone. Install SHAP + provide a trained model for full SHAP values.")