import pandas as pd

# -----------------------
# Load vitals
# -----------------------
vitals = pd.read_csv("data/raw/vitals.csv")

feature_map = {
    220045: "heart_rate",
    220179: "systolic_bp",
    220050: "mean_bp",
    220210: "resp_rate",
    223761: "temperature"
}

vitals["feature"] = vitals["itemid"].map(feature_map)

# -----------------------
# Pivot to wide format
# -----------------------
vitals_wide = vitals.pivot_table(
    index=["subject_id", "stay_id", "charttime"],
    columns="feature",
    values="valuenum"
).reset_index()

# -----------------------
# Sort by patient and time
# -----------------------
vitals_wide = vitals_wide.sort_values(["subject_id", "stay_id", "charttime"])

# -----------------------
# Forward fill only feature columns
# -----------------------
feature_cols = ["heart_rate", "systolic_bp", "mean_bp", "resp_rate", "temperature"]

vitals_wide[feature_cols] = (
    vitals_wide.groupby(["subject_id", "stay_id"])[feature_cols].ffill()
)

# -----------------------
# Fill remaining NaNs with median
# -----------------------
vitals_wide[feature_cols] = vitals_wide[feature_cols].fillna(
    vitals_wide[feature_cols].median()
)

# -----------------------
# Save
# -----------------------
vitals_wide.to_csv("data/processed/structured_clean.csv", index=False)

print("\nStructured preprocessing complete")
print(vitals_wide.head())
print(vitals_wide.shape)

print("\nMissing values after cleaning:")
print(vitals_wide.isnull().sum())