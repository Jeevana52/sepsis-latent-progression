import pandas as pd

labs = pd.read_csv("data/raw/labs.csv")

print("\nAvailable itemids:")
print(labs["itemid"].value_counts().head(30))