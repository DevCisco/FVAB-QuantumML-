import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from stabilized_vqc_model import StabilizedVQC


def load_split_csv(split, d, seed):
    """
    Carica un CSV prodotto da test.py e restituisce features e labels
    come tensori PyTorch.
    """
    path = os.path.join("artifacts", "sweep", f"B1_pca_{split}_d{d}_seed{seed}.csv")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df['label'].values,   dtype=torch.long)
    return X, y


def train_regime(d, seed):
    torch.set_num_threads(1)

    X_train, y_train = load_split_csv('train', d, seed)
    X_val,   y_val   = load_split_csv('val',   d, seed)

    # subset per velocizzare: coerente con la policy del bando
    X_train, y_train = X_train[:5000], y_train[:5000]
    X_val,   y_val   = X_val[:1000],   y_val[:1000]

    model     = StabilizedVQC(n_qubits=4, d_latent=d)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    epochs   = 5

    BATCH_TRAIN = 128
    BATCH_VAL   = 200
    rng = torch.Generator()
    rng.manual_seed(seed)

    for epoch in range(epochs):
        idx_train = torch.randperm(len(X_train), generator=rng)[:BATCH_TRAIN]
        idx_val   = torch.randperm(len(X_val),   generator=rng)[:BATCH_VAL]

        model.train()
        optimizer.zero_grad()
        output = model(X_train[idx_train])
        loss   = criterion(output, y_train[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val[idx_val])
            acc = (val_out.argmax(1) == y_val[idx_val]).float().mean().item() * 100

            if acc > best_acc:
                best_acc = acc
                os.makedirs("artifacts/checkpoints", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"artifacts/checkpoints/vqc_d{d}_seed{seed}_best.pth" # salvato per uso in week8_robustness.py
                )

        print(f"d={d} seed={seed} | Epoch {epoch+1} | Val Acc: {acc:.2f}%", flush=True)

    return {"d": d, "seed": seed, "VQC_Best_Acc": best_acc}


def main():
    os.makedirs("artifacts/checkpoints", exist_ok=True)

    dims  = [32, 16, 8, 4]
    seeds = [11, 17, 29]

    # ogni combinazione (d, seed) gira su un processo separato
    combos = [(d, s) for d in dims for s in seeds]

    max_workers = min(len(combos), os.cpu_count() or 1)
    print(f"[INFO] Avvio {len(combos)} run su {max_workers} processi paralleli...\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_regime, d, s): (d, s)
            for d, s in combos
        }
        for future in as_completed(futures):
            d, s = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(
                    f"[OK] d={d} seed={s} → Best Acc: {result['VQC_Best_Acc']:.2f}%",
                    flush=True
                )
            except Exception as e:
                print(f"[ERROR] d={d} seed={s} → {e}", flush=True)

    df = pd.DataFrame(all_results)
    df = df.sort_values(["d", "seed"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv("artifacts/week4_vqc_results.csv", index=False)
    print("\n[DONE] Training completato.")


if __name__ == "__main__":
    main()