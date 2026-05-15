import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_engine import HybridModel
from data_loader import get_data_loaders
import pandas as pd
import time 
import os
from test import get_weights_path

flag = 'octmnist'
percorso_pesi = get_weights_path(flag)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    # FIX #5: la funzione originale aveva `return` dentro il loop al primo batch,
    # e non restituiva nulla se il loader era vuoto o se i >= 10 al batch 0.
    # Un return None causa AttributeError su loss.backward() nel chiamante.
    # Soluzione: accumulare la loss e restituire la media fuori dal loop,
    # con un valore di fallback (0.0) se nessun batch viene processato.
    total_loss = 0.0
    n_batches = 0

    for i, (images, labels) in enumerate(loader):
        if i >= 10:
            break
        images = images[:32]
        labels = labels[:32].squeeze().long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(">>> Forward pass OK. Calcolo gradiente (attendere...)")
        loss.backward()
        optimizer.step()
        print(">>> Backward pass OK. Pesi aggiornati.")

        total_loss += loss.item()
        n_batches += 1

    # Restituisce sempre un float, mai None
    return total_loss / n_batches if n_batches > 0 else 0.0

def run_experiment(d, backbone='resnet'):
    config = {'d_latent': d, 'n_qubits': 4, 'n_layers': 1}
    device = torch.device("cpu")
    
    model = HybridModel(config, backbone_type=backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_loader, _, test_loader = get_data_loaders(batch_size=128)
    
    print(f"--- Starting Real Run: VQC + {backbone} (d={d}) ---")
    start_time = time.time()
    
    for epoch in range(3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    
    end_time = time.time()
    
    os.makedirs("experiments/models", exist_ok=True)
    torch.save(model.state_dict(), f"experiments/models/vqc_{backbone}_d{d}_mock.pth")
    
    return {"model": f"VQC_{backbone}", "d": d, "time": end_time - start_time}

if __name__ == "__main__":
    result = run_experiment(d=32, backbone='resnet')
    print(f"Run completato in {result['time']:.2f}s. Artifact salvato.")
    result = run_experiment(d=16, backbone='resnet')
    print(f"Run completato in {result['time']:.2f}s. Artifact salvato.")
    result = run_experiment(d=8, backbone='resnet')
    print(f"Run completato in {result['time']:.2f}s. Artifact salvato.")
    result = run_experiment(d=4, backbone='resnet')
    print(f"Run completato in {result['time']:.2f}s. Artifact salvato.")