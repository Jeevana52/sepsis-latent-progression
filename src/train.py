import torch
import torch.nn as nn

# -----------------------
# Simulate actual latent output from VAE
# -----------------------
latent = torch.randn(5, 64)

# -----------------------
# Predictor
# -----------------------
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.risk_head = nn.Linear(64, 1)
        self.stage_head = nn.Linear(64, 3)

    def forward(self, x):
        risk = torch.sigmoid(self.risk_head(x))
        stage_logits = self.stage_head(x)
        stage = torch.argmax(stage_logits, dim=1)
        return risk, stage

# -----------------------
# Run prediction
# -----------------------
model = Predictor()

risk, stage = model(latent)

# -----------------------
# Stage labels
# -----------------------
stage_labels = {
    0: "Normal",
    1: "Early Sepsis",
    2: "Severe Sepsis"
}

# -----------------------
# Display realistic output
# -----------------------
for i in range(len(risk)):
    print(f"\nPatient {i+1}")
    print(f"Sepsis Risk: {risk[i].item():.2f}")
    print(f"Predicted Stage: {stage_labels[int(stage[i])]}")