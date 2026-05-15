import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unsupervised_models import VanillaAE, RegularizedAE
from seed_manager import set_seed
import numpy as np

def train_ae(model_type, d, train_data, seed, epochs=20, batch_size=256):
    """
    FIX BUG 3: la versione originale caricava l'intero dataset in un unico
    tensore ed eseguiva forward/backward su tutto in un solo step (batch
    gradient descent puro). Su dataset grandi causa OOM; su qualsiasi
    dimensione è instabile per autoencoder profondi e molto più lento di
    mini-batch SGD. Ora usiamo DataLoader con mini-batch.

    FIX BUG 3b: il parametro 'seed' veniva ricevuto ma mai usato, rendendo
    l'inizializzazione dei pesi non riproducibile tra run. Ora set_seed viene
    chiamato prima della costruzione del modello.
    """
    # Riproducibilità: seed applicato prima della costruzione del modello
    set_seed(seed)

    device = "cpu"
    if model_type == 'B2':
        model = VanillaAE(d).to(device)
    else:
        model = RegularizedAE(d).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    data_tensor = torch.tensor(train_data).float()
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch)

            # FIX BUG 4: torch.norm(z, 1) con tensore 2D è ambiguo e NON calcola
            # la norma L1 per-campione. La penalità L1 corretta sul bottleneck è
            # z.abs().mean(): differenziabile, stabile, semanticamente corretta
            # (media dei valori assoluti delle attivazioni sul batch).
            if model_type == 'B3':
                l1_penalty = z.abs().mean()
                loss = loss + 1e-4 * l1_penalty

            loss.backward()
            optimizer.step()

    return model