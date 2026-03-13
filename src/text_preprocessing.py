import pandas as pd
import re

# -----------------------
# Load notes
# -----------------------
notes = pd.read_csv("data/raw/notes.csv")

# -----------------------
# Keep needed columns
# -----------------------
notes = notes[["subject_id", "hadm_id", "charttime", "text"]]

# -----------------------
# Remove missing text
# -----------------------
notes = notes.dropna(subset=["text"])

# -----------------------
# Basic cleaning
# -----------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'___', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9., ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

notes["clean_text"] = notes["text"].apply(clean_text)

# -----------------------
# Keep only first 500 chars
# (fast for ClinicalBERT later)
# -----------------------
notes["clean_text"] = notes["clean_text"].str[:500]

# -----------------------
# Save processed notes
# -----------------------
notes.to_csv("data/processed/notes_clean.csv", index=False)

print("\nNotes preprocessing complete")
print(notes[["subject_id", "clean_text"]].head())
print(notes.shape)