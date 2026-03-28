import pandas as pd

data = pd.read_csv("data/processed/final_structured.csv")

# Strong abnormal cases
patients = data[
    (
        (data["heart_rate"] > 120) &
        (
            (data["lactate"] >= 2) |
            (data["wbc"] >= 11) |
            (data["mean_bp"] <= 70)
        )
    )
]

patients = patients.drop_duplicates(subset=["subject_id"]).head(60)

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

    if row["temperature"] >= 100.4:
        score += 0.15
        explanations.append("fever")

    score = min(score, 0.99)

    if score < 0.25:
        stage = "Normal"
    elif score < 0.6:
        stage = "Early Sepsis"
    else:
        stage = "Severe Sepsis"

    print("\n======================")
    print(f"Patient Number: {i}")
    print(f"Subject ID: {row['subject_id']}")
    print(f"Heart Rate: {row['heart_rate']}")
    print(f"Mean BP: {row['mean_bp']}")
    print(f"Lactate: {row['lactate']}")
    print(f"WBC: {row['wbc']}")
    print(f"Temperature: {row['temperature']}")
    print(f"Sepsis Risk: {score:.2f}")
    print(f"Predicted Stage: {stage}")
    print("Explanation:", ", ".join(explanations))
