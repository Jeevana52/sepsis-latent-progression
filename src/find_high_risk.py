import pandas as pd

data = pd.read_csv("data/processed/final_structured.csv")

# Strong abnormal rows
high_risk = data[
    (data["heart_rate"] > 120) &
    (
        (data["lactate"] >= 2) |
        (data["wbc"] >= 11) |
        (data["mean_bp"] <= 70)
    )
]

# unique patients
high_risk = high_risk.drop_duplicates(subset=["subject_id"])

print(high_risk.head(10))
print("\nTotal high-risk patients found:", len(high_risk))