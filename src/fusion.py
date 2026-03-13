import torch

# -----------------------
# Simulated encoder outputs
# -----------------------

# structured latent (from LSTM)
structured = torch.randn(20, 32)

# text embedding (from ClinicalBERT)
text = torch.randn(20, 768)

# -----------------------
# Fusion
# -----------------------
fused = torch.cat((structured, text), dim=1)

print("Structured shape:", structured.shape)
print("Text shape:", text.shape)
print("Fused shape:", fused.shape)

print(fused[:3])