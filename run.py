"""
run.py
------
Quick start script.

Usage:
    python run.py train        → Train the model
    python run.py evaluate     → Run evaluation + save plots
    python run.py app          → Launch web app
    python run.py all          → Train → Evaluate → Launch app
    python run.py demo         → Quick demo without training
"""

import sys
import os


def train():
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING THE MODEL")
    print("=" * 60)
    from src.train import train as run_train
    run_train()


def evaluate():
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATING THE MODEL")
    print("=" * 60)
    from src.evaluate import evaluate as run_eval
    run_eval()


def launch_app():
    print("\n" + "=" * 60)
    print("STEP 3: LAUNCHING WEB APP")
    print("Open: http://localhost:5000")
    print("=" * 60)
    from app import app, load_model
    load_model()
    app.run(debug=False, host="0.0.0.0", port=5000)


def demo():
    """Quick demo using rule-based fallback (no training needed)."""
    import numpy as np
    from src.explainability import SepsisExplainer

    print("\n" + "=" * 60)
    print("DEMO — Rule-Based Prediction (no model required)")
    print("=" * 60)

    patients = [
        {"id": 10119770, "name": "High Risk",   "feats": [142.0, 59.0, 24.0, 85.0, 38.8, 3.1, 14.2]},
        {"id": 10012345, "name": "Moderate",    "feats": [108.0, 68.0, 21.0, 100.0, 38.3, 1.8, 11.5]},
        {"id": 10099999, "name": "Normal",      "feats": [72.0,  82.0, 16.0, 120.0, 37.0, 1.0,  7.0]},
    ]

    explainer = SepsisExplainer(output_dir="outputs")

    for p in patients:
        feats = np.array(p["feats"])
        risk  = 0.0
        f = feats
        if f[5] > 2.0: risk += 0.30
        if f[0] > 100: risk += 0.20
        if f[1] < 65:  risk += 0.20
        if f[6] > 12:  risk += 0.10
        if f[4] > 38.3 or f[4] < 36.0: risk += 0.10
        if f[2] > 20:  risk += 0.10
        risk = min(risk, 1.0)

        stage = ("Severe Sepsis" if risk >= 0.6
                 else "Early Sepsis" if risk >= 0.25
                 else "Normal")

        result = explainer.explain(feats, subject_id=p["id"],
                                   risk_score=round(risk, 3), stage=stage)

        print(f"\n── Patient {p['id']} ({p['name']}) ──────────────")
        print(f"  Risk Score : {risk:.2f} ({risk*100:.1f}%)")
        print(f"  Stage      : {stage}")
        print(f"  Explanation: {result['explanation_text']}")
        print(f"  Top Findings:")
        for feat in result["top_features"][:3]:
            print(f"    {feat['feature']:15s} = {feat['value']:.1f}")

    print("\n✓ Demo complete. Outputs saved to outputs/")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "app":
        launch_app()
    elif cmd == "all":
        train()
        evaluate()
        launch_app()
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)
        print("\nExample:")
        print("  python run.py demo    ← try this first (no data needed)")
        print("  python run.py train")
        print("  python run.py app")


if __name__ == "__main__":
    main()