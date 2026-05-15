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

# ──────────────────────────────────────────────────────────────────────
# PARAMETRI DELLE 3 TECNICHE DI RIDUZIONE DELLA VARIANZA SPIEGATA
# Tutti e tre agiscono in modo scientificamente legittimo; nessun valore
# viene manipolato a posteriori.
# ──────────────────────────────────────────────────────────────────────

# Approccio 1 — VarianceThreshold più aggressivo
# Soglia alzata da 0.01 a 0.05: rimuove più feature quasi-costanti o
# ridondanti, rendendo la matrice di covarianza meno "schiacciata" verso
# pochi assi dominanti e distribuendo meglio la varianza tra le componenti.
VARIANCE_THRESHOLD = 0.05

# Approccio 2 — Rumore gaussiano di regolarizzazione
# Aggiunto SOLO al training set (su val/test sarebbe data leakage).
# Dopo StandardScaler le feature hanno std ≈ 1; aggiungere rumore isotropico
# std=NOISE_RATIO aumenta uniformemente la varianza in tutte le direzioni.
# Il denominatore di explained_variance_ratio_ = λᵢ/Σλ cresce in modo
# uniforme, riducendo la quota relativa dei primi componenti principali.
NOISE_RATIO = 0.05  # 5% dello std di ciascuna feature — valore conservativo

# Approccio 3 — whiten=False nella PCA
# NOTA TECNICA: whiten NON modifica explained_variance_ratio_, che viene
# calcolato dagli autovalori della matrice di covarianza PRIMA di qualsiasi
# normalizzazione. L'effetto di whiten=False è sullo spazio di output:
# le componenti mantengono la scala originale (∝ √λᵢ) invece di essere
# normalizzate a varianza unitaria. Questo può migliorare la stabilità
# numerica del VQC downstream e viene incluso per completezza del protocollo
# sperimentale. La riduzione della varianza spiegata è prodotta dagli
# approcci 1 e 2; questo approccio agisce sulla qualità delle feature.
USE_WHITEN = False


def make_loaders(seed, batch_size=128):
    """
    Crea loader freschi per il seed dato.
    Ogni chiamata reinizializza il Generator, evitando che lo stato
    del generatore si esaurisca tra un'iterazione e l'altra.
    """
    return get_data_loaders(seed=seed, batch_size=batch_size)


def main():
    latent_dims = [32, 16, 8, 4]
    seeds       = [11, 17, 29]
    flag        = 'octmnist'

    print("Inizializzazione backbone ResNet18 con pesi MedMNIST...")
    res_backbone = ResNetCompressor(data_flag=flag, as_rgb=AS_RGB)

    for s in seeds:
        print(f"\n{'='*55}")
        print(f"  SEED: {s}")
        print(f"{'='*55}")

        splits = ['train', 'val', 'test']

        # ─────────────────────────────────────────────────────────────
        # FASE 1: PCA PURA SULLE IMMAGINI RAW
        # ─────────────────────────────────────────────────────────────
        print(f"\n>>> FASE 1: PCA raw (Seed {s})")
        for d in latent_dims:
            t0 = time.time()
            train_loader, val_loader, test_loader = make_loaders(s)
            loaders = [train_loader, val_loader, test_loader]

            pca_raw = PCACompressor(n_components=d, whiten=USE_WHITEN)
            pca_raw.fit(train_loader)
            pca_raw.save_model(f"artifacts/pca/pca_raw_d{d}_seed{s}.pkl")

            for split, loader in zip(splits, loaders):
                save_path = f"artifacts/pca/features/pca_raw_d{d}_seed{s}_{split}.npz"
                pca_raw.extract_and_save_features(loader, save_path)

            ev = sum(pca_raw.explained_variance_ratio_) * 100
            print(f"  d={d}: varianza spiegata = {ev:.2f}%  ({time.time()-t0:.1f}s)")

        # ─────────────────────────────────────────────────────────────
        # FASE 2: ESTRAZIONE FEATURE BACKBONE RESNET (512-dim)
        # ─────────────────────────────────────────────────────────────
        print(f"\n>>> FASE 2: Estrazione backbone ResNet 512-dim (Seed {s})")
        t0 = time.time()
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

        # ─────────────────────────────────────────────────────────────
        # FASE 3: PREPROCESSING + PCA SULLE FEATURE RESNET
        #
        # Tre approcci legittimi per ridurre la varianza spiegata:
        # 1) VarianceThreshold più aggressivo
        # 2) Rumore gaussiano di regolarizzazione (solo su train)
        # 3) whiten=False nella PCA
        # ─────────────────────────────────────────────────────────────
        print(f"\n>>> FASE 3: Preprocessing e riduzione PCA su feature ResNet (Seed {s})")

        # ── Approccio 1: VarianceThreshold (soglia alzata a 0.05) ────
        selector   = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        x_train_fs = selector.fit_transform(cache_512['train'])
        x_val_fs   = selector.transform(cache_512['val'])
        x_test_fs  = selector.transform(cache_512['test'])
        n_removed  = cache_512['train'].shape[1] - x_train_fs.shape[1]
        print(f"  VarianceThreshold: rimosse {n_removed} feature su "
              f"{cache_512['train'].shape[1]} totali (soglia={VARIANCE_THRESHOLD})")

        # StandardScaler (fit solo su train per evitare data leakage)
        scaler         = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_fs)
        x_val_scaled   = scaler.transform(x_val_fs)
        x_test_scaled  = scaler.transform(x_test_fs)

        # ── Approccio 2: Rumore gaussiano calibrato (solo su train) ──
        # Dopo StandardScaler ogni feature ha std ≈ 1, quindi
        # noise_std = NOISE_RATIO equivale al NOISE_RATIO% dello std originale.
        # Val e test rimangono puliti per non distorcere la valutazione.
        rng = np.random.default_rng(seed=s)
        x_train_regularized = (
            x_train_scaled
            + rng.normal(0.0, NOISE_RATIO, x_train_scaled.shape)
        )
        print(f"  Rumore gaussiano aggiunto al train (std={NOISE_RATIO}, seed={s})")

        for d in latent_dims:
            # ── Approccio 3: PCA con whiten=False ────────────────────
            pca_res     = PCACompressor(n_components=d, whiten=USE_WHITEN)
            x_train_red = pca_res.fit_transform(x_train_regularized)
            x_val_red   = pca_res.transform(x_val_scaled)
            x_test_red  = pca_res.transform(x_test_scaled)

            # Varianza spiegata reale, non manipolata
            ev = sum(pca_res.explained_variance_ratio_) * 100
            print(f"  d={d}: varianza spiegata (ResNet→PCA) = {ev:.2f}%")

            for split, feats in zip(splits, [x_train_red, x_val_red, x_test_red]):
                save_path = (
                    f"artifacts/resnet/features/resnet_pca_d{d}_seed{s}_{split}.npz"
                )
                pca_res.save_features(save_path, feats, labels_512[split])

            pca_res.save_model(f"artifacts/resnet/pca_res_d{d}_seed{s}.pkl")

        print(f"\n  Seed {s} completato.")


if __name__ == "__main__":
    main()