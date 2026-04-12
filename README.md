# SepsisProg — Early Sepsis Detection Using VAE + ClinicalBERT

> **Learning Latent Sepsis Progression Patterns Using Variational Autoencoders and ClinicalBERT: An Explainable Generative Approach**

---

## Project Overview

This project detects early-stage sepsis in ICU patients by combining:

- **Structured vitals + lab biomarkers** (heart rate, blood pressure, lactate, WBC, etc.) from MIMIC-IV
- **Clinical discharge notes** encoded using ClinicalBERT
- A **Variational Autoencoder (VAE)** that learns hidden disease progression states
- A **learned MLP classifier** that predicts: Normal / Early Sepsis / Severe Sepsis
- **SHAP explainability** showing which features drove each prediction
- A **Flask web app** where a patient (or clinician) can upload a medical document and instantly see their risk score, stage, and explanation

---

## Architecture

```
MIMIC-IV Data
    ├── chartevents.csv  ──► Preprocessing ──► LSTM Encoder (→ 32-dim)
    ├── labevents.csv    ──► Lab Merge      ──┤
    └── discharge.csv    ──► ClinicalBERT   ──► 768-dim text embedding
                                                       │
                                             Fusion Layer (800-dim)
                                                       │
                                               VAE Encoder
                                           mu (64-dim), logvar (64-dim)
                                                       │
                                             MLP Classifier
                                                       │
                              ┌────────────────────────┤
                              │                        │
                         Risk Score             SHAP Explanation
                       Stage Prediction        Feature Attribution
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jeevana52/sepsis-latent-progression.git
cd sepsis-latent-progression
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Train the model

```bash
python -m src.train
```

This trains the end-to-end pipeline (LSTM → Fusion → VAE → Classifier) and saves:
- `models/best_model.pt`
- `models/feature_mean.npy` and `feature_std.npy`
- `outputs/classification_report.txt`
- `outputs/metrics.json`

### Step 2 — Evaluate

```bash
python -m src.evaluate
```

Generates:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/training_history.png`

### Step 3 — Launch the web app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Web App — How It Works

The web app accepts two types of input:

**Option A — Manual Entry:**  
Enter vitals directly (heart rate, blood pressure, lactate, etc.)

**Option B — Upload a Document:**  
Upload a discharge PDF, CSV, or text note. The app extracts vital signs automatically using regex pattern matching on clinical text.

**Output shown:**
- Risk score (0–100%)
- Sepsis stage badge (Normal / Early Sepsis / Severe Sepsis)
- Probability bars for each class
- Plain-English clinical explanation
- SHAP feature contribution plot
- Color-coded vital status grid

---

## Project Structure

```
sepsis_project/
├── app.py                      ← Flask web app (main entry point)
├── requirements.txt
├── config.yaml
├── README.md
│
├── src/
│   ├── data_loader.py          ← Load MIMIC-IV CSVs
│   ├── preprocessing.py        ← Clean structured vitals
│   ├── lab_preprocessing.py    ← Extract lactate, WBC
│   ├── text_preprocessing.py   ← Clean clinical notes
│   ├── model_lstm.py           ← LSTM structured encoder
│   ├── model_text.py           ← ClinicalBERT text encoder
│   ├── fusion.py               ← Combine LSTM + BERT → 800-dim
│   ├── vae.py                  ← Variational Autoencoder
│   ├── classifier.py           ← MLP classifier on VAE latent
│   ├── train.py                ← End-to-end training script
│   ├── evaluate.py             ← Metrics, confusion matrix, ROC
│   ├── explainability.py       ← SHAP explanations
│   └── demo_pipeline.py        ← Demo with real patient examples
│
├── templates/
│   ├── index.html              ← Upload / manual entry page
│   └── result.html             ← Prediction result page
│
├── data/
│   ├── raw/                    ← Original MIMIC-IV files (not committed)
│   └── processed/              ← Cleaned subsets
│
├── models/                     ← Saved model weights
├── outputs/                    ← Predictions, plots, metrics
└── notebooks/
    └── exploration.ipynb
```

---

## Key Results

| Metric         | Value |
|---------------|-------|
| Model         | VAE + MLP Classifier |
| Features      | 7 structured + ClinicalBERT text |
| Training set  | 80% of 50,000 ICU rows |
| Classes       | Normal / Early Sepsis / Severe Sepsis |
| Outputs       | Risk score, stage, SHAP explanation |

---

## Dataset

This project uses the **MIMIC-IV** dataset (Medical Information Mart for Intensive Care).  
Access requires credentialing via PhysioNet: https://physionet.org/content/mimiciv/

Files used:
- `chartevents.csv.gz` — ICU vitals
- `labevents.csv.gz` — Lab biomarkers
- `icustays.csv.gz` — ICU stay metadata
- `patients.csv.gz` — Demographics
- `discharge.csv.gz` — Clinical notes

---

## Tech Stack

| Component       | Technology |
|----------------|-----------|
| Deep Learning  | PyTorch |
| Text Encoding  | HuggingFace Transformers (ClinicalBERT) |
| Explainability | SHAP |
| Web Framework  | Flask |
| Evaluation     | scikit-learn |
| Visualization  | matplotlib |
