import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_engine import HybridModel
from data_loader import get_data_loaders
import pandas as pd
import time
import os
# FIX E: rimosso "from config import Config" — inutilizzato ovunque nel file e
# causa ModuleNotFoundError se config.py non esiste, bloccando l'intero modulo
# (incluse le chiamate da run_production_suite.py).

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss / len(loader), 100 * correct / total

def train_production(d, backbone, epochs=5):
    device = torch.device("cpu")
    train_loader, val_loader, _ = get_data_loaders(batch_size=4)

    model = HybridModel(
        {'d_latent': d, 'n_qubits': 4, 'n_layers': 1},
        backbone_type=backbone
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    history = []
    os.makedirs("experiments/models", exist_ok=True)

    print(f"\n>>> Inizio Produzione: {backbone.upper()} + VQC | d={d}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0          # FIX F: contatore reale, non divisore fisso
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            if i > 50:
                break
            images, labels = images.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches if n_batches > 0 else 0.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        # FIX D: ":.4,f" è una SyntaxError — lo specificatore corretto è ":.4f"
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"experiments/models/best_vqc_{backbone}_d{d}.pth"
            )

        history.append([epoch+1, d, backbone, avg_train_loss, val_acc])

    df_history = pd.DataFrame(
        history, columns=['epoch', 'd', 'backbone', 'loss', 'val_acc']
    )
    log_file = "experiments/production_log.csv"
    os.makedirs("experiments", exist_ok=True)
    df_history.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

if __name__ == "__main__":
    train_production(d=32, backbone='resnet', epochs=3)
    train_production(d=16, backbone='resnet', epochs=3)
    train_production(d=8,  backbone='resnet', epochs=3)
    train_production(d=4,  backbone='resnet', epochs=3)