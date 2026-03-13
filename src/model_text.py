import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------
# Load cleaned notes
# -----------------------
notes = pd.read_csv("data/processed/notes_clean.csv")

# -----------------------
# Take small sample first (fast run)
# -----------------------
texts = notes["clean_text"].head(20).tolist()

# -----------------------
# Load ClinicalBERT
# -----------------------
model_name = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# -----------------------
# Tokenize
# -----------------------
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# -----------------------
# Generate embeddings
# -----------------------
with torch.no_grad():
    outputs = model(**inputs)

# CLS token embedding
embeddings = outputs.last_hidden_state[:, 0, :]

print("ClinicalBERT embedding shape:", embeddings.shape)
print(embeddings[:3])