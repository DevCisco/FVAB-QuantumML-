import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unsupervised_models import VanillaAE, RegularizedAE
from seed_manager import set_seed
import numpy as np

def train_ae(model_type, d, train_data, seed, epochs=20, batch_size=256):
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

            if model_type == 'B3':
                l1_penalty = z.abs().mean()
                loss = loss + 1e-4 * l1_penalty

            loss.backward()
            optimizer.step()

    return model