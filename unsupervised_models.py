import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Costante di configurazione condivisa
# ---------------------------------------------------------------------------
AS_RGB: bool = True


# ---------------------------------------------------------------------------
# B2: Autoencoder Standard (VanillaAE)
# ---------------------------------------------------------------------------
class VanillaAE(nn.Module):
    """
    Autoencoder a due strati nascosti.

    Architettura fissa (verificata dai checkpoint esistenti):
        Encoder: 512 → 128 → d_latent
        Decoder: d_latent → 128 → 512

    Input: feature ResNet18 a 512 dim (PRIMA della PCA).
    I modelli AE operano sulle feature raw del backbone, non sulle
    feature già ridotte dalla PCA.

    Args:
        d_latent (int): dimensione del collo di bottiglia (4/8/16/32).
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            reconstruction (Tensor): shape (B, 512).
            z              (Tensor): shape (B, d_latent).
        """
        z = self.encoder(x)
        return self.decoder(z), z


# ---------------------------------------------------------------------------
# B3: Shallow Regularized AE (con vincolo di sparsità L1 esplicito)
# ---------------------------------------------------------------------------
class RegularizedAE(nn.Module):
    """
    Autoencoder shallow (encoder/decoder a singolo strato lineare).

    Architettura fissa (verificata dai checkpoint esistenti):
        Encoder: 512 → d_latent  (shallow, un solo strato)
        Decoder: d_latent → 512

    Input: feature ResNet18 a 512 dim (PRIMA della PCA).

    La sparsità è imposta tramite penalità L1 su z nel training loop:
        loss = MSELoss(recon, x) + model.sparsity_loss(z)

    Args:
        d_latent        (int):   dimensione del collo di bottiglia.
        sparsity_weight (float): peso λ della penalità L1 (default 1e-3).
    """

    def __init__(self, d_latent: int, sparsity_weight: float = 1e-3):
        super().__init__()
        self.encoder = nn.Linear(512, d_latent)
        self.decoder = nn.Linear(d_latent, 512)
        self.sparsity_weight = sparsity_weight

    def forward(self, x: torch.Tensor):
        """
        Returns:
            reconstruction (Tensor): shape (B, 512).
            z              (Tensor): shape (B, d_latent).
        """
        z = torch.sigmoid(self.encoder(x))
        return self.decoder(z), z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Penalità L1: λ * mean(|z|).
        Da sommare alla reconstruction loss nel training loop.
        """
        return self.sparsity_weight * z.abs().mean()