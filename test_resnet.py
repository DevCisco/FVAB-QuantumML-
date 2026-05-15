import torch
import torch.nn as nn
import os
import time
import psutil
from data_loader import get_data_loaders
from pca_res_compressors import ResNetCompressor
from test import get_weights_path

flag = 'octmnist'
percorso_pesi = get_weights_path(flag)

def print_mem():
    process = psutil.Process(os.getpid())
    print(f"   [RAM Usata: {process.memory_info().rss / 1024**2:.2f} MB]")

def main():
    # FIX #3: "gpu" non è un identificatore di device valido in PyTorch.
    # Il device MPS (Apple Silicon) si specifica come "mps", non "gpu".
    # Con "gpu" torch.device() lancia un RuntimeError immediato.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Utilizzo device: {device}")
    
    print("Caricamento dati OCTMNIST...")
    train_loader, _, _ = get_data_loaders(batch_size=128)

    latent_dims = [32, 16, 8, 4]
    
    if not os.path.exists("artifacts/resnet"):
        os.makedirs("artifacts/resnet")

    for d in latent_dims:
        print(f"\n--- Test ResNet18 Feature Extraction per d={d} ---")
        print_mem()
        
        model = ResNetCompressor(d_latent=d, data_flag=flag).to(device)
        model.eval()
        
        start_time = time.time()
        
        print(f"Inizio estrazione feature per un campione di test...")
        
        with torch.no_grad():
            batch_idx = 0
            for images, _ in train_loader:
                batch_idx += 1
                images = images.to(device)
                
                u = model(images)
                
                if batch_idx % 100 == 0:
                    print(f"   > Processati {batch_idx} batch. Shape output: {u.shape}")
                
                if batch_idx >= 50:
                    break
        
        save_path = f"artifacts/resnet/resnet_propper_d{d}.pth"
        torch.save(model.state_dict(), save_path)
        
        end_time = time.time()
        print(f"Completato d={d} in {end_time - start_time:.2f} secondi.")
        print(f"Modello salvato in: {save_path}")

if __name__ == "__main__":
    main()