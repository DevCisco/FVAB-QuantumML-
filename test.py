from data_loader import get_data_loaders
from pca_res_compressors import PCACompressor, ResNetCompressor
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

os.makedirs("artifacts/pca/features",    exist_ok=True)
os.makedirs("artifacts/resnet/features", exist_ok=True)

# Costante condivisa: deve corrispondere all'impostazione in data_loader.py
AS_RGB = True


def make_loaders(seed, batch_size=128):
    """
    Crea una coppia di loader freschi per il seed dato.

    Ogni chiamata costruisce un nuovo DataLoader con un Generator
    appena inizializzato. In questo modo lo stato del generatore non si
    'esaurisce' tra un'iterazione e l'altra del ciclo su latent_dims,
    garantendo che l'ordine di shuffle sia sempre quello atteso per quel seed.
    """
    return get_data_loaders(seed=seed, batch_size=batch_size)


def main():
    latent_dims = [32, 16, 8, 4]
    seeds       = [11, 17, 29]
    flag        = 'octmnist'

    # as_rgb=True comunicato esplicitamente al compressore,
    # così la logica di adattamento conv1 usa i canali reali delle immagini
    # e non il valore nativo di INFO['n_channels'] (che sarebbe 1).
    print("Inizializzazione backbone ResNet18 con pesi MedMNIST...")
    res_backbone = ResNetCompressor(data_flag=flag, as_rgb=AS_RGB)

    for s in seeds:
        print(f"\n{'='*55}")
        print(f"  SEED: {s}")
        print(f"{'='*55}")

        splits = ['train', 'val', 'test']

        # -----------------------------------------------------------
        # FASE 1: PCA PURA SULLE IMMAGINI RAW
        # -----------------------------------------------------------
        print(f"\n>>> FASE 1: PCA raw (Seed {s})")
        for d in latent_dims:
            t0 = time.time()

            # BUG 2 FIX: loader freschi per ogni d → generator non esaurito
            train_loader, val_loader, test_loader = make_loaders(s)
            loaders = [train_loader, val_loader, test_loader]

            pca_raw = PCACompressor(n_components=d, whiten=True)
            pca_raw.fit(train_loader)
            pca_raw.save_model(f"artifacts/pca/pca_raw_d{d}_seed{s}.pkl")

            for split, loader in zip(splits, loaders):
                save_path = f"artifacts/pca/features/pca_raw_d{d}_seed{s}_{split}.npz"
                pca_raw.extract_and_save_features(loader, save_path)

            ev = sum(pca_raw.explained_variance_ratio_) * 100
            print(f"  d={d}: varianza spiegata = {ev:.2f}%  ({time.time()-t0:.1f}s)")

        # -----------------------------------------------------------
        # FASE 2: ESTRAZIONE FEATURE BACKBONE RESNET (512-dim)
        # -----------------------------------------------------------
        print(f"\n>>> FASE 2: Estrazione backbone ResNet 512-dim (Seed {s})")
        t0 = time.time()

        # BUG 2 FIX: loader freschi anche per la fase ResNet
        train_loader, val_loader, test_loader = make_loaders(s)
        loaders = [train_loader, val_loader, test_loader]

        cache_512  = {}
        labels_512 = {}

        for split, loader in zip(splits, loaders):
            base_path = f"artifacts/resnet/resnet_base_512_seed{s}_{split}.npz"
            res_backbone.extract_and_save_features(loader, base_path)
            data              = res_backbone.load_features(base_path)
            cache_512[split]  = data['features']
            labels_512[split] = data['labels']

        print(f"  Backbone 512-dim completato ({time.time()-t0:.1f}s)")

        # -----------------------------------------------------------
        # FASE 3: SCALING + PCA SULLE FEATURE RESNET
        # -----------------------------------------------------------
        print(f"\n>>> FASE 3: Scaling e riduzione PCA su feature ResNet (Seed {s})")

        selector   = VarianceThreshold(threshold=0.01)
        x_train_fs = selector.fit_transform(cache_512['train'])
        x_val_fs   = selector.transform(cache_512['val'])
        x_test_fs  = selector.transform(cache_512['test'])

        scaler         = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_fs)
        x_val_scaled   = scaler.transform(x_val_fs)
        x_test_scaled  = scaler.transform(x_test_fs)

        for d in latent_dims:
            pca_res = PCACompressor(n_components=d, whiten=True)

            x_train_red = pca_res.fit_transform(x_train_scaled)
            x_val_red   = pca_res.transform(x_val_scaled)
            x_test_red  = pca_res.transform(x_test_scaled)

            ev = sum(pca_res.explained_variance_ratio_) * 100
            print(f"  d={d}: varianza spiegata (ResNet→PCA) = {ev:.2f}%")

            for split, feats in zip(splits, [x_train_red, x_val_red, x_test_red]):
                save_path = f"artifacts/resnet/features/resnet_pca_d{d}_seed{s}_{split}.npz"
                pca_res.save_features(save_path, feats, labels_512[split])

            pca_res.save_model(f"artifacts/resnet/pca_res_d{d}_seed{s}.pkl")

        print(f"\n  Seed {s} completato.")


if __name__ == "__main__":
    main()