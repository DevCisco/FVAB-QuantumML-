import torch
import torch.nn as nn

# B2: Autoencoder Standard
class VanillaAE(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, d_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# B3: Shallow Regularized AE (con vincolo di sparsità)
class RegularizedAE(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.encoder = nn.Linear(512, d_latent) # Shallow
        self.decoder = nn.Linear(d_latent, 512)

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x)) # Sigmoide per facilitare sparsità
        return self.decoder(z), z