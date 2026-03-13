import pandas as pd
import torch
import torch.nn as nn

# -----------------------
# Load structured data
# -----------------------
data = pd.read_csv("data/processed/final_structured.csv")

# -----------------------
# Select features
# -----------------------
features = [
    "heart_rate",
    "mean_bp",
    "resp_rate",
    "systolic_bp",
    "temperature",
    "lactate",
    "wbc"
]

X = data[features].values

# -----------------------
# Convert to tensor
# -----------------------
X_tensor = torch.tensor(X, dtype=torch.float32)

# reshape for LSTM
X_tensor = X_tensor.unsqueeze(1)

print("Input shape:", X_tensor.shape)

# -----------------------
# LSTM Model
# -----------------------
class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=32,
            batch_first=True
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return hidden[-1]

# -----------------------
# Run model
# -----------------------
model = LSTMEncoder()

with torch.no_grad():
    latent = model(X_tensor)

print("Latent shape:", latent.shape)
print(latent[:5])