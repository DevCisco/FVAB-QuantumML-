from data_loader import get_data_loaders
from pca_res_compressors import PCACompressor, ResNetCompressor
import os, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

#artifact salvati su disco:
#res_raw_{split}.npz                          → feature 512-dim raw per training AE (B2/B3)
#{split}_ids_dimensione{d}_seed{s}_.csv       → feature d-dim (PCA), input VQC/few-shot

os.makedirs("artifacts/resnet/features", exist_ok=True)
os.makedirs("artifacts/sweep", exist_ok=True)

AS_RGB = True



def make_loaders(seed, batch_size=128):
    return get_data_loaders(seed=seed, batch_size=batch_size)


def extract_embeddings(backbone, loader):
    """Estrae embedding 512-dim in RAM senza salvare su disco."""
    all_features, all_labels = [], []
    backbone.eval()
    import torch
    with torch.no_grad():
        for images, labels in loader:
            all_features.append(backbone.extract_backbone(images).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.vstack(all_labels)


def save_raw_features(cache_512, labels_512, splits, d, seed):
    """Feature ResNet18 raw a 512-dim per training AE (B2/B3)."""
    for split in splits:
        path = f"artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz"
        np.savez_compressed(path, features=cache_512[split], labels=labels_512[split])
    print("  feature raw 512-dim salvate → res_raw_" f"d{d}_" "{train|val|test}_s{seed}.npz")


def save_pca_features_csv(features, labels, split, d, seed):
    """
    Salva le feature PCA d-dim in formato CSV con la convenzione:
        {split}_ids_dimensione{d}_seed{seed}_.csv

    Colonne: feat_0, feat_1, ..., feat_{d-1}, label
    """
    path = f"artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv"
    cols = [f"feat_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=cols)
    df['label'] = labels.ravel()
    df.to_csv(path, index=False)


def main():
    latent_dims = [32, 16, 8, 4]
    seeds = [11, 17, 29]
    flag = 'octmnist'
    splits = ['train', 'val', 'test']

    res_backbone = ResNetCompressor(data_flag=flag, as_rgb=AS_RGB)

    for d in latent_dims:
        for s in seeds:
            print(f"\nseed {s} ".upper() + "=" * 40)

            # ── fase 1: immagine → ResNet-18 → 512-dim (in RAM) ───────────
            t0 = time.time()
            train_loader, val_loader, test_loader = make_loaders(s)

            cache_512, labels_512 = {}, {}
            for split, loader in zip(splits, [train_loader, val_loader, test_loader]):
                feats, labels = extract_embeddings(res_backbone, loader)
                cache_512[split]  = feats
                labels_512[split] = labels
            print(f"  resnet 512-dim: fatto ({time.time()-t0:.0f}s)")

            # ── fase 2: salvataggio feature raw per training AE (B2/B3) ───
            save_raw_features(cache_512, labels_512, splits, d, s)

            # ── fase 3: preprocessing (fit solo su train) ──────────────────
            scaler = StandardScaler()
            x_tr  = scaler.fit_transform(cache_512['train'])
            x_val = scaler.transform(cache_512['val'])
            x_te  = scaler.transform(cache_512['test'])
            
            # ── fase 4: PCA → d-dim → salvataggio CSV ──────────────────────
            
            t0 = time.time()

            pca = PCACompressor(n_components=d)
            x_tr_d  = pca.fit_transform(x_tr)
            x_val_d = pca.transform(x_val)
            x_te_d  = pca.transform(x_te)

            ev = sum(pca.explained_variance_ratio_) * 100
            print(f"  d={d}: varianza spiegata = {ev:.1f}%  ({time.time()-t0:.0f}s)")

            for split, feats in zip(splits, [x_tr_d, x_val_d, x_te_d]):
                save_pca_features_csv(feats, labels_512[split], split, d, s)


        print(f"  seed {s} completato")


if __name__ == "__main__":
    main()