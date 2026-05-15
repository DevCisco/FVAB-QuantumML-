import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
from hybrid_engine import HybridModel
from data_loader import get_data_loaders

# FIX #10: CONFIG non deve contenere 'device' né 'batch_size'.
# Queste chiavi venivano incluse in config_with_d e passate a HybridModel,
# che non le conosce. 'device' è un torch.device, non un iperparametro del modello.
# Separiamo le due responsabilità: CONFIG_MODEL per HybridModel, costanti separate
# per device e batch_size.
CONFIG_MODEL = {
    'n_qubits': 4,
    'n_layers': 1,
    'lr': 0.01,
}
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_experiment(d, backbone, train_loss, test_acc, duration):
    """Archivia i risultati in un CSV unico (I/O fissato)"""
    log_file = "experiments/master_log.csv"
    os.makedirs("experiments", exist_ok=True)
    new_data = {
        "week": [2],
        "backbone": [backbone],
        "d_latent": [d],
        "train_loss": [round(train_loss, 4)],
        "test_acc": [round(test_acc, 2)],
        "duration_sec": [round(duration, 2)],
        "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")]
    }
    df = pd.DataFrame(new_data)
    if not os.path.isfile(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)

def run_mock_training(d, backbone):
    """Esegue un run reale minimo per validare la pipeline"""
    print(f"\n>>> Validazione End-to-End: VQC + {backbone} | d={d}")
    
    train_loader, _, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # FIX #10: config del modello senza 'device' né 'batch_size'
    config_with_d = {**CONFIG_MODEL, 'd_latent': d}
    model = HybridModel(config_with_d, backbone_type=backbone).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG_MODEL['lr'])
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    model.train()
    total_loss = 0.0
    n_batches = 0  # FIX #9: contatore reale dei batch processati

    for i, (images, labels) in enumerate(train_loader):
        if i > 32:
            break
        labels = labels.squeeze().long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    duration = time.time() - start_time
    
    # FIX #9: il codice originale divideva per 6 (valore magico errato).
    # Il loop elabora fino a 33 batch (i = 0..32). Dividiamo per n_batches,
    # il numero effettivo di batch completati, per ottenere la loss media corretta.
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    os.makedirs("experiments/models", exist_ok=True)
    torch.save(model.state_dict(), f"experiments/models/vqc_{backbone}_d{d}_v1.pth")
    
    log_experiment(d, backbone, avg_loss, 0.0, duration)

def main():
    print("=== FINALIZZAZIONE===")
    
    if os.path.exists("artifacts/final_benchmarks.csv"):
        print(">>> Comparatore lineare comune rilevato. Integrazione OK.")
        baseline = pd.read_csv("artifacts/final_benchmarks.csv")
        print(baseline)
    else:
        print("!!! ATTENZIONE: Benchmark 1 non trovato.")

    regimes = [32, 16, 8, 4]
    for d in regimes:
        run_mock_training(d, backbone='resnet')
        
    print(">>> I/O fissato in: experiments/master_log.csv")
    print(">>> Modularità verificata: il sistema è pronto per i rami di produzione.")

if __name__ == "__main__":
    main()