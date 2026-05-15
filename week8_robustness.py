import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from unsupervised_models import VanillaAE, RegularizedAE
from vqc_fewshot_engine import FewShotVQC, get_octmnist_dataset


# FIX BUG 9: una lambda in transforms.Compose non è pickle-serializzabile.
# Con DataLoader(num_workers > 0) causa AttributeError al momento del fork.
# Sostituita con una classe callable top-level, che è sempre serializzabile.
class AddGaussianNoise:
    """Applica rumore gaussiano al tensore immagine già normalizzato."""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


def run_robustness_test(latent_dim, seed, compression_type='B1'):
    """
    Esegue il test di robustezza per il Team B.
    compression_type: 'B1' (PCA), 'B2' (VanillaAE), 'B3' (RegAE)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Backbone ResNet18 bloccato
    backbone = models.resnet18(weights='IMAGENET1K_V1')
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()

    # 2. Dataset con rumore gaussiano
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(std=0.1),   # FIX BUG 9: classe callable, non lambda
    ])

    test_dataset = get_octmnist_dataset(split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Caricamento Compressore
    # FIX BUG 6: VanillaAE e RegularizedAE accettano solo 'd_latent' come
    # argomento posizionale (definiti in unsupervised_models.py). Passare
    # 'input_dim=' e 'latent_dim=' causava TypeError immediato.
    # FIX BUG 7: weights_only=True per sicurezza (RCE) e compatibilità PyTorch 2.4+.
    if compression_type == 'B2':
        compressor = VanillaAE(d_latent=latent_dim).to(device)
        compressor.load_state_dict(
            torch.load(f"models/B2_d{latent_dim}_s{seed}.pt", weights_only=True)
        )
        compressor.eval()
    elif compression_type == 'B3':
        compressor = RegularizedAE(d_latent=latent_dim).to(device)
        compressor.load_state_dict(
            torch.load(f"models/B3_d{latent_dim}_s{seed}.pt", weights_only=True)
        )
        compressor.eval()
    else:
        import joblib
        compressor = joblib.load(f"models/B1_pca_d{latent_dim}.pkl")

    # 4. Caricamento VQC
    # FIX BUG 8: FewShotVQC.__init__ accetta solo 'd_latent', non 'n_qubits'
    # né 'latent_dim'. Passare kwargs inesistenti causava TypeError immediato.
    # FIX BUG 7: weights_only=True anche qui.
    vqc = FewShotVQC(d_latent=latent_dim)
    vqc.load_state_dict(
        torch.load(f"models/vqc_d{latent_dim}_s{seed}.pt", weights_only=True)
    )
    vqc = vqc.to(device).eval()

    # 5. Loop di Inferenza
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = backbone(images)

            if compression_type == 'B1':
                compressed = torch.tensor(
                    compressor.transform(features.cpu().numpy()),
                    dtype=torch.float32
                ).to(device)
            else:
                _, compressed = compressor(features)

            logits = vqc(compressed)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    return all_preds, all_targets


if __name__ == "__main__":
    # FIX BUG 7 applicato sopra; import metriche spostato in testa al file
    dimensions = [32, 16, 8, 4]
    robustness_seeds = [11, 23]

    results_log = []

    for d in dimensions:
        for s in robustness_seeds:
            for comp in ['B1', 'B2', 'B3']:
                print(f"Audit Week 8: Test d={d}, Seed={s}, Comp={comp}")
                preds, targets = run_robustness_test(d, s, comp)

                acc = accuracy_score(targets, preds)
                f1  = f1_score(targets, preds, average='macro')

                results_log.append({
                    'dim': d, 'seed': s, 'compression': comp,
                    'accuracy': acc, 'macro_f1': f1
                })

    os.makedirs("artifacts", exist_ok=True)
    df = pd.DataFrame(results_log)
    df.to_csv("artifacts/week8_robustness_report.csv", index=False)
    print("Freeze Week 8 completato. Artifact generati.")