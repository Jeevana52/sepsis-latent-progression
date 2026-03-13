import torch
import torch.nn as nn

# -----------------------
# Simulated fused input
# -----------------------
x = torch.randn(20, 800)

# -----------------------
# VAE Model
# -----------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(800, 256)

        self.fc_mu = nn.Linear(256, 64)
        self.fc_logvar = nn.Linear(256, 64)

        self.fc_decode = nn.Linear(64, 256)
        self.fc_out = nn.Linear(256, 800)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_decode(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -----------------------
# Run VAE
# -----------------------
model = VAE()

recon, mu, logvar = model(x)

print("Input shape:", x.shape)
print("Latent mu shape:", mu.shape)
print("Latent logvar shape:", logvar.shape)
print("Reconstructed shape:", recon.shape)