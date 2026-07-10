import os

import pandas as pd
import torch
from medmnist import OCTMNIST
from torch.utils.data import ConcatDataset

SEEDS          = [11, 17, 29]
TRAIN_FRACTION = 0.9   # 90% del canonical_train → train, 10% → val


def generate_and_save_ids(output_dir: str = "dataset_splits") -> None:
    """
    Genera e salva gli indici di split per ogni seed.

    Garantisce:
    - test_ids_fixed.csv identico per tutti i seed (nessuna varianza sul test)
    - Nessun overlap tra test e train/val (verificato esplicitamente)
    - Indici compatibili con il ConcatDataset di data_loader.py
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carichiamo i tre split canonici SENZA transform: servono solo le
    # dimensioni, non i dati effettivi. Stesso ordine del ConcatDataset
    # in data_loader.py — l'ordine determina gli indici.
    canonical_train = OCTMNIST(split='train', download=True)
    canonical_val   = OCTMNIST(split='val',   download=True)
    canonical_test  = OCTMNIST(split='test',  download=True)

    n_train = len(canonical_train)
    n_val   = len(canonical_val)
    n_test  = len(canonical_test)
    total   = n_train + n_val + n_test

    print("Dimensioni dataset canonico OCTMNIST:")
    print(f"  canonical train : {n_train:>7}")
    print(f"  canonical val   : {n_val:>7}")
    print(f"  canonical test  : {n_test:>7}")
    print(f"  totale          : {total:>7}")

    # ── 1. Test set fisso ─────────────────────────────────────────────────────
    # canonical_val  → indici [n_train,       n_train + n_val − 1]
    # canonical_test → indici [n_train + n_val, total − 1]
    # Generato UNA SOLA VOLTA, identico per tutti i seed.
    # Il backbone MedMNIST è addestrato solo su canonical_train → nessun
    # leakage: questi campioni non hanno mai influenzato i pesi del backbone.
    test_ids_fixed = list(range(n_train, total))

    fixed_path = os.path.join(output_dir, "test_ids_fixed.csv")
    pd.DataFrame(test_ids_fixed, columns=["sample_index"]).to_csv(
        fixed_path, index=False
    )
    print(
        f"\n[FIXED] test_ids_fixed.csv: {len(test_ids_fixed)} campioni "
        f"(canonical val={n_val} + canonical test={n_test})"
    )

    # ── 2. Split train/val per seed ───────────────────────────────────────────
    # Pool: solo canonical_train → indici [0, n_train − 1].
    # Permutazione deterministica con torch.Generator (stesso framework del
    # resto del codebase) — riproducibile con lo stesso seed.
    # Rapporto: TRAIN_FRACTION train, (1 − TRAIN_FRACTION) val.
    # n_val_split = n_train − n_train_split garantisce che nessun campione
    # vada perso per effetto dell'arrotondamento int().
    n_train_split = int(TRAIN_FRACTION * n_train)
    n_val_split   = n_train - n_train_split

    print(
        f"\nSplit canonical train pool ({n_train}) con rapporto "
        f"{TRAIN_FRACTION:.0%}/{1 - TRAIN_FRACTION:.0%}:"
    )
    print(f"  train : {n_train_split}")
    print(f"  val   : {n_val_split}")

    for seed in SEEDS:
        gen  = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_train, generator=gen).tolist()

        # I valori in perm sono posizioni in [0, n_train−1], che corrispondono
        # direttamente agli indici del ConcatDataset per canonical_train.
        train_ids = perm[:n_train_split]
        val_ids   = perm[n_train_split:]

        for fname, ids in [
            (f"train_ids_{seed}.csv", train_ids),
            (f"val_ids_{seed}.csv",   val_ids),
        ]:
            out_path = os.path.join(output_dir, fname)
            pd.DataFrame(ids, columns=["sample_index"]).to_csv(
                out_path, index=False
            )
            print(f"  seed={seed} → {fname}: {len(ids)} campioni")

    # ── 3. Verifica di coerenza ───────────────────────────────────────────────
    # Per costruzione i due range [0, n_train−1] e [n_train, total−1] sono
    # disgiunti. Verifichiamo comunque esplicitamente per ogni seed.
    test_set = set(test_ids_fixed)
    errors   = []

    for seed in SEEDS:
        train_ids_check = pd.read_csv(
            os.path.join(output_dir, f"train_ids_{seed}.csv")
        )["sample_index"].tolist()
        val_ids_check = pd.read_csv(
            os.path.join(output_dir, f"val_ids_{seed}.csv")
        )["sample_index"].tolist()

        overlap_train = test_set & set(train_ids_check)
        overlap_val   = test_set & set(val_ids_check)

        if overlap_train or overlap_val:
            errors.append(
                f"seed={seed}: train_overlap={len(overlap_train)}, "
                f"val_overlap={len(overlap_val)}"
            )

    if errors:
        raise RuntimeError(
            "[BUG] Overlap rilevato tra test fisso e train/val:\n"
            + "\n".join(errors)
        )

    print("\n[OK] Nessun overlap test / train / val per tutti i seed.")
    print("[OK] Split generati, salvati e verificati.")
    print(
        "\nFile generati:"
        f"\n  {os.path.join(output_dir, 'test_ids_fixed.csv')}  ← identico per tutti i seed"
    )
    for seed in SEEDS:
        print(f"  {os.path.join(output_dir, f'train_ids_{seed}.csv')}")
        print(f"  {os.path.join(output_dir, f'val_ids_{seed}.csv')}")


if __name__ == "__main__":
    generate_and_save_ids()