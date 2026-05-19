from data_loader import get_data_loaders
from pca_res_compressors import PCACompressor, ResNetCompressor
import os, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

os.makedirs("artifacts/pca/features", exist_ok=True)
os.makedirs("artifacts/resnet/features", exist_ok=True)

AS_RGB = True

# soglia per rimuovere feature quasi costanti dal backbone
VARIANCE_THRESHOLD = 0.05

# rumore gaussiano sul train prima della PCA (calibrato su griglia 0.40-0.60,
# valore scelto: 0.48 -> d=32 ~83%, d=4 ~37%)
# serve solo sul train, su val/test non si tocca niente
NOISE_RATIO = 0.48

# whiten=False: le componenti mantengono la scala originale proporzionale
# a sqrt(lambda), non viene normalizzata la varianza in output
USE_WHITEN = False


def make_loaders(seed, batch_size=128):
    # ricrea i loader da zero ad ogni chiamata così il generator non si
    # esaurisce tra un d e l'altro nel ciclo
    return get_data_loaders(seed=seed, batch_size=batch_size)


def main():
    latent_dims = [32, 16, 8, 4]
    seeds = [11, 17, 29]
    flag = 'octmnist'

    res_backbone = ResNetCompressor(data_flag=flag, as_rgb=AS_RGB)
    splits = ['train', 'val', 'test']

    for s in seeds:
        print(f"\nseed {s} ".upper() + "=" * 40)

        # --- fase 1: PCA diretta sui pixel raw ---
        for d in latent_dims:
            t0 = time.time()
            train_loader, val_loader, test_loader = make_loaders(s)
            loaders = [train_loader, val_loader, test_loader]

            pca_raw = PCACompressor(n_components=d, whiten=USE_WHITEN)
            pca_raw.fit(train_loader)
            pca_raw.save_model(f"artifacts/pca/pca_raw_d{d}_seed{s}.pkl")

            for split, loader in zip(splits, loaders):
                pca_raw.extract_and_save_features(
                    loader, f"artifacts/pca/features/pca_raw_d{d}_seed{s}_{split}.npz"
                )

            ev = sum(pca_raw.explained_variance_ratio_) * 100
            print(f"  pca raw d={d}: {ev:.1f}% var spiegata  ({time.time()-t0:.0f}s)")

        # --- fase 2: embedding 512-dim con ResNet ---
        t0 = time.time()
        train_loader, val_loader, test_loader = make_loaders(s)
        loaders = [train_loader, val_loader, test_loader]

        cache_512, labels_512 = {}, {}
        for split, loader in zip(splits, loaders):
            path = f"artifacts/resnet/resnet_base_512_seed{s}_{split}.npz"
            res_backbone.extract_and_save_features(loader, path)
            data = res_backbone.load_features(path)
            cache_512[split] = data['features']
            labels_512[split] = data['labels']
        print(f"  resnet 512-dim: fatto  ({time.time()-t0:.0f}s)")

        # --- fase 3: preprocessing + PCA sulle feature ResNet ---

        # rimozione feature a bassa varianza
        sel = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        x_tr = sel.fit_transform(cache_512['train'])
        x_val = sel.transform(cache_512['val'])
        x_te = sel.transform(cache_512['test'])
        print(f"  rimosse {cache_512['train'].shape[1] - x_tr.shape[1]} feature "
              f"(soglia var={VARIANCE_THRESHOLD})")

        # scaling
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_val = scaler.transform(x_val)
        x_te = scaler.transform(x_te)

        # rumore sul solo training set
        rng = np.random.default_rng(seed=s)
        x_tr = x_tr + rng.normal(0.0, NOISE_RATIO, x_tr.shape)

        for d in latent_dims:
            pca = PCACompressor(n_components=d, whiten=USE_WHITEN)
            x_tr_d = pca.fit_transform(x_tr)
            x_val_d = pca.transform(x_val)
            x_te_d = pca.transform(x_te)

            ev = sum(pca.explained_variance_ratio_) * 100
            print(f"  pca resnet d={d}: {ev:.1f}% var spiegata")

            for split, feats in zip(splits, [x_tr_d, x_val_d, x_te_d]):
                pca.save_features(
                    f"artifacts/resnet/features/resnet_pca_d{d}_seed{s}_{split}.npz",
                    feats, labels_512[split]
                )
            pca.save_model(f"artifacts/resnet/pca_res_d{d}_seed{s}.pkl")

        print(f"  seed {s} completato")


if __name__ == "__main__":
    main()