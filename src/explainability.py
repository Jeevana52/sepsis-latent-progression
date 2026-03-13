# Example patient physiological values

patient = {
    "heart_rate": 137,
    "lactate": 2.8,
    "wbc": 14,
    "mean_bp": 62
}

# -----------------------
# Explanation engine
# -----------------------
explanations = []

if patient["lactate"] > 2:
    explanations.append("elevated lactate")

if patient["heart_rate"] > 120:
    explanations.append("tachycardia")

if patient["wbc"] > 12:
    explanations.append("high white blood cell count")

if patient["mean_bp"] < 65:
    explanations.append("hypotension")

# -----------------------
# Final explanation
# -----------------------
print("\nClinical Explanation:")

if explanations:
    print("High sepsis risk due to " + ", ".join(explanations))
else:
    print("No major sepsis indicators detected")