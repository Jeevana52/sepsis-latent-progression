import pandas as pd

# -----------------------
# Load real processed structured data
# -----------------------
data = pd.read_csv("data/processed/final_structured.csv")

# -----------------------
# Select first 5 real patients
# -----------------------
# Select clinically abnormal rows
patients = data[
    (data["heart_rate"] > 120) &
    (
        (data["lactate"] >= 2) |
        (data["wbc"] >= 11) |
        (data["mean_bp"] <= 70)
    )
]

patients = patients.drop_duplicates(subset=["subject_id"]).head(5)

# -----------------------
# Real clinical scoring
# -----------------------
for i, row in patients.iterrows():
    score = 0
    explanations = []

    if row["lactate"] >= 2:
        score += 0.35
        explanations.append("elevated lactate")

    if row["heart_rate"] >= 120:
        score += 0.25
        explanations.append("tachycardia")

    if row["wbc"] >= 11:
        score += 0.20
        explanations.append("elevated WBC")

    if row["mean_bp"] <= 70:
        score += 0.20
        explanations.append("hypotension")

    if row["temperature"]>=100.4:
        score+=0.15
        explanations.append("fever")

    score = min(score, 0.99)

    # stage classification
    if score < 0.25:
        stage = "Normal"
    elif score < 0.6:
        stage = "Early Sepsis"
    else:
        stage = "Severe Sepsis"

    # -----------------------
    # Output
    # -----------------------
    print(f"\nPatient {i+1}")
    print(f"Subject ID: {row['subject_id']}")
    print(f"Heart Rate: {row['heart_rate']}")
    print(f"Lactate: {row['lactate']}")
    print(f"WBC: {row['wbc']}")
    print(f"Mean BP: {row['mean_bp']}")

    print(f"Sepsis Risk: {score:.2f}")
    print(f"Predicted Stage: {stage}")

    if explanations:
        print("Explanation:", ", ".join(explanations))
    else:
        print("Explanation: stable physiological state")