import pandas as pd

# -----------------------
# Load structured data
# -----------------------
structured = pd.read_csv("data/processed/structured_clean.csv")

# -----------------------
# Load labs
# -----------------------
labs = pd.read_csv("data/raw/labs.csv")

# -----------------------
# Correct lab item mapping
# -----------------------
lab_map = {
    50813: "lactate",
    51300: "wbc"
}

labs["lab_feature"] = labs["itemid"].map(lab_map)

labs = labs.dropna(subset=["lab_feature"])

# -----------------------
# Pivot labs
# -----------------------
labs_wide = labs.pivot_table(
    index=["subject_id", "charttime"],
    columns="lab_feature",
    values="valuenum"
).reset_index()

# -----------------------
# Merge with structured data
# -----------------------
merged = pd.merge(
    structured,
    labs_wide,
    on=["subject_id", "charttime"],
    how="left"
)

# -----------------------
# Fill missing values
# -----------------------
merged["lactate"] = merged["lactate"].fillna(merged["lactate"].median())
merged["wbc"] = merged["wbc"].fillna(merged["wbc"].median())

# -----------------------
# Save final structured dataset
# -----------------------
merged.to_csv("data/processed/final_structured.csv", index=False)

print("\nLab merge complete")
print(merged.head())
print(merged.shape)

print("\nMissing values:")
print(merged.isnull().sum())