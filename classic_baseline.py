# ---------------------------------------------------------------------------
# FIX freeze_support come prima istruzione — obbligatorio su Windows
# con ProcessPoolExecutor (spawn).
# ---------------------------------------------------------------------------
import multiprocessing
multiprocessing.freeze_support()

import logging
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import f1_score as sk_f1


# ---------------------------------------------------------------------------
# Costanti globali
# Allineate con train_vqc_production.py per confronto diretto.
# ---------------------------------------------------------------------------
DIMS              = [32, 16, 8, 4]
SEEDS             = [11, 17, 29]
COMPRESSORS       = ['B1', 'B2', 'B3']   # FIX: prima solo B1 — vedi nota sotto
EPOCHS            = 100        # Adam converge molto più velocemente di NFT
PATIENCE          = 15         # più alto del VQC: Adam può oscillare prima di convergere
LR                = 0.01
WEIGHT_DECAY      = 1e-4       # L2 regularization: riduce overfitting su 32 campioni
HIDDEN_DIM        = 32         # neuroni strato nascosto
DROPOUT           = 0.3        # dropout: necessario con soli 32 campioni di training
MAX_WORKERS       = 4
N_CLASSES         = 4
SAMPLES_PER_CLASS = 8          # = VQC → 32 campioni bilanciati per training

JOB_TIMEOUT_SEC   = 2 * 3600   # classico è molto più veloce del VQC

# Percorsi CSV per ogni compressore — identici a train_vqc_production.py e
# master_sweep_team_b.py, unica fonte di verità per questa convenzione di
# naming duplicata su tre file (nessuno dei tre importa dagli altri due,
# per restare eseguibili come script standalone indipendenti).
COMPRESSOR_PATHS = {
    'B1': "artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv",
    'B2': "artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv",
    'B3': "artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv",
}


# ---------------------------------------------------------------------------
# Logging per worker
# ---------------------------------------------------------------------------
def get_logger(d: int, seed: int, compressor: str) -> logging.Logger:
    os.makedirs("experiments_classical/logs", exist_ok=True)
    name   = f"worker_{compressor}_d{d}_s{seed}"
    logger = logging.getLogger(f"classical_{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        f"experiments_classical/logs/{name}.log", mode='w', encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Caricamento feature pre-salvate da test.py (B1) / b2_b3_training.py (B2, B3)
#
# Colonne CSV: feat_0..feat_{d-1} (B1) oppure latent_0..latent_{d-1} (B2/B3),
# label. Il filtro [c != 'label'] gestisce entrambe le convenzioni di nome
# senza bisogno di distinguerle esplicitamente.
# ---------------------------------------------------------------------------
def load_pca_features(d: int, seed: int, split: str, compressor: str) -> tuple:
    """
    Carica feature d-dim (PCA o autoencoder) seed-specifiche dal CSV pre-salvato.

    Args:
        d          (int): dimensione latente (4/8/16/32).
        seed       (int): seed usato durante la generazione (test.py / b2_b3_training.py).
        split      (str): 'train' | 'val' | 'test'.
        compressor (str): 'B1' | 'B2' | 'B3'.

    Returns:
        X (ndarray): shape (N, d), dtype float32.
        y (ndarray): shape (N,), dtype int.
    """
    path = COMPRESSOR_PATHS[compressor].format(split=split, d=d, seed=seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            f"Per B1: eseguire test.py\n"
            f"Per B2/B3: eseguire b2_b3_training.py"
        )
    df   = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Campionamento bilanciato dal training set
# Allineato con make_balanced_subset di train_vqc_production.py.
# ---------------------------------------------------------------------------
def make_balanced_batch(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> tuple:
    """
    Estrae un batch bilanciato con esattamente samples_per_class per classe.
    replace=True come fallback se una classe ha meno campioni del richiesto.

    Returns:
        X_batch (Tensor): shape (N_CLASSES * samples_per_class, d).
        y_batch (Tensor): shape (N_CLASSES * samples_per_class,).
    """
    rng     = np.random.default_rng(seed)
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            raise ValueError(
                f"Classe {cls} assente. Presenti: {np.unique(y).tolist()}"
            )
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())
    return (
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Pesi per classe — formula sklearn 'balanced'
# Identico a train_vqc_production.py per confronto equo della loss.
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=N_CLASSES).astype(float)
    weights = len(y) / (N_CLASSES * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Modello classico — MLP a due strati
#
# Architettura:
#     Linear(input_dim, HIDDEN_DIM) → ReLU → Dropout(DROPOUT)
#     → Linear(HIDDEN_DIM, N_CLASSES)
#
# Input: d componenti compresse complete (non troncate a 4 come nel VQC).
# Dropout e weight_decay riducono l'overfitting su soli 32 campioni.
#
# Parametri:
#     d=4:  (4*32+32) + (32*4+4) =  160+132 =  292
#     d=8:  (8*32+32) + (32*4+4) =  288+132 =  420
#     d=16: (16*32+32)+ (32*4+4) =  544+132 =  676
#     d=32: (32*32+32)+ (32*4+4) = 1056+132 = 1188
# ---------------------------------------------------------------------------
class ClassicalMLP(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = HIDDEN_DIM,
        n_classes:  int = N_CLASSES,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model,
    X:        torch.Tensor,
    labels_np: np.ndarray,
    criterion,
    device,
) -> tuple:
    """
    Valuta il modello su feature pre-caricate.

    Returns:
        loss (float), acc (float), macro_f1 (float), per_class_f1 (list).
    """
    model.eval()
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(X)
        loss    = criterion(outputs, labels_t)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()

    acc          = float((preds == labels_np).mean() * 100)
    macro_f1     = float(sk_f1(labels_np, preds, average='macro',    zero_division=0))
    per_class_f1 = sk_f1(labels_np, preds, average=None, zero_division=0).tolist()

    return loss.item(), acc, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# CSV per-worker — no lock, no race condition tra processi
# Stesse colonne di train_vqc_production.py per confronto diretto.
# ---------------------------------------------------------------------------
def save_worker_csv(history: list, d: int, seed: int, compressor: str) -> None:
    os.makedirs("experiments_classical/history", exist_ok=True)
    path = f"experiments_classical/history/log_{compressor}_d{d}_s{seed}.csv"
    pd.DataFrame(
        history,
        columns=[
            'epoch', 'd', 'seed', 'compressor',
            'train_loss',
            'val_loss',  'val_acc',  'val_macro_f1',
            'test_loss', 'test_acc', 'test_macro_f1',
            'per_class_f1',
        ],
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def train_production(d: int, seed: int, compressor: str) -> dict:
    """
    Addestra ClassicalMLP per una tripla (d, seed, compressor).

    Differenze rispetto al VQC:
    - Feature: d-dim complete (non a blocchi via re-uploading)
    - Optimizer: Adam con backpropagation (non NFT)
    - Epoche: fino a 100 con early stopping patience=15
    - Nessuna dipendenza da Qiskit o backbone live
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed, compressor)
    logger.info(f"Avvio job d={d} seed={seed} compressor={compressor}")

    # — Caricamento feature pre-salvate -------------------------------------
    try:
        X_train, y_train = load_pca_features(d, seed, 'train', compressor)
        X_val,   y_val   = load_pca_features(d, seed, 'val',   compressor)
        X_test,  y_test  = load_pca_features(d, seed, 'test',  compressor)
    except FileNotFoundError as e:
        logger.error(f"File mancante: {e}")
        raise

    logger.info(
        f"Feature caricate ({compressor}): train {X_train.shape} | "
        f"val {X_val.shape} | test {X_test.shape}"
    )
    print(
        f"[{compressor} d={d} s={seed}] feature: "
        f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}",
        flush=True,
    )

    # — Tensori su device (val e test interi, per evaluation completa) ------
    X_val_t  = torch.tensor(X_val,  dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    # — Batch di training bilanciato (32 campioni = 8 per classe) -----------
    try:
        X_batch, y_batch = make_balanced_batch(
            X_train, y_train, SAMPLES_PER_CLASS, seed
        )
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Pesi classe e criterion ---------------------------------------------
    class_weights = compute_class_weights(y_train).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # — Modello e optimizer -------------------------------------------------
    input_dim = X_train.shape[1]   # = d
    model     = ClassicalMLP(input_dim=input_dim).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri totali MLP: {n_params} (d={d})")
    print(f"[{compressor} d={d} s={seed}] parametri MLP: {n_params}", flush=True)

    # — Training loop -------------------------------------------------------
    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0
    best_loss          = float('inf')
    patience_ctr       = 0
    history            = []
    os.makedirs("experiments_classical/models", exist_ok=True)

    print(
        f">>> MLP: {compressor} d={d} seed={seed} | "
        f"epochs={EPOCHS} | patience={PATIENCE} | lr={LR}",
        flush=True,
    )
    logger.info(f"Inizio training MLP epochs={EPOCHS} patience={PATIENCE}")

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            # — Un passo Adam sul batch bilanciato ---------------------------
            model.train()
            optimizer.zero_grad()
            outputs    = model(X_batch)
            train_loss_t = criterion(outputs, y_batch)
            train_loss_t.backward()
            optimizer.step()
            train_loss = train_loss_t.item()

            # — Valutazione su val e test (interi) ---------------------------
            val_loss,  val_acc,  val_macro_f1,  _            = evaluate(
                model, X_val_t,  y_val,  criterion, device
            )
            test_loss, test_acc, test_macro_f1, per_class_f1 = evaluate(
                model, X_test_t, y_test, criterion, device
            )

        except Exception:
            logger.error(f"Epoch {epoch+1}:\n{traceback.format_exc()}")
            print(
                f"[WARN] {compressor} d={d} s={seed} | Epoch {epoch+1} fallita — continuo",
                flush=True,
            )
            history.append([
                epoch+1, d, seed, compressor,
                float('nan'),
                float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), '[]',
            ])
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
            continue

        elapsed = time.time() - t0

        # Stampa: macro-F1 val e test prima della loss, poi per-class F1
        log_msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"macro-F1 val: {val_macro_f1:.4f} | macro-F1 test: {test_macro_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"per-class F1: {[f'{v:.3f}' for v in per_class_f1]} | "
            f"t={elapsed:.3f}s"
        )
        logger.info(log_msg)

        # Stampa a console ogni 10 epoche per non saturare il terminale
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{compressor} d={d} s={seed} | {log_msg}", flush=True)

        # — Early stopping su val macro-F1 ----------------------------------
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = test_macro_f1
            best_loss          = train_loss
            patience_ctr       = 0
            ckpt_path = (
                f"experiments_classical/models/"
                f"best_mlp_{compressor}_d{d}_seed{seed}.pth"
            )
            try:
                torch.save(model.state_dict(), ckpt_path)
                logger.info(
                    f"Checkpoint → {ckpt_path} "
                    f"(val F1={val_macro_f1:.4f} | test F1={test_macro_f1:.4f})"
                )
            except Exception:
                logger.error(f"Errore checkpoint:\n{traceback.format_exc()}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping a epoch {epoch+1} (patience={PATIENCE})")
                print(
                    f"[STOP] {compressor} d={d} s={seed} | "
                    f"Early stopping a epoch {epoch+1}",
                    flush=True,
                )
                history.append([
                    epoch+1, d, seed, compressor,
                    train_loss,
                    val_loss,  val_acc,  val_macro_f1,
                    test_loss, test_acc, test_macro_f1,
                    str([f'{v:.3f}' for v in per_class_f1]),
                ])
                break

        history.append([
            epoch+1, d, seed, compressor,
            train_loss,
            val_loss,  val_acc,  val_macro_f1,
            test_loss, test_acc, test_macro_f1,
            str([f'{v:.3f}' for v in per_class_f1]),
        ])

    # — Salvataggio CSV per-worker (no race condition) ----------------------
    try:
        save_worker_csv(history, d, seed, compressor)
    except Exception:
        logger.error(f"Errore CSV:\n{traceback.format_exc()}")

    logger.info(
        f"Completato. Best val macro-F1: {best_val_macro_f1:.4f} | "
        f"test macro-F1: {best_test_macro_f1:.4f}"
    )
    print(
        f"[OK] {compressor} d={d} s={seed} → "
        f"val F1: {best_val_macro_f1:.4f} | test F1: {best_test_macro_f1:.4f}",
        flush=True,
    )
    return {
        "d":                  d,
        "seed":               seed,
        "compressor":         compressor,
        "best_loss":          best_loss,
        "best_val_macro_f1":  best_val_macro_f1,
        "best_test_macro_f1": best_test_macro_f1,
    }


# ---------------------------------------------------------------------------
# Entry point — parallelizzazione Windows-safe
# ---------------------------------------------------------------------------
def main():
    os.makedirs("experiments_classical/models",  exist_ok=True)
    os.makedirs("experiments_classical/logs",    exist_ok=True)
    os.makedirs("experiments_classical/history", exist_ok=True)
    os.makedirs("experiments_classical",         exist_ok=True)

    jobs = [(d, s, c) for c in COMPRESSORS for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(f"[INFO] Compressori: {COMPRESSORS}")
    print(
        f"[INFO] ClassicalMLP | hidden={HIDDEN_DIM} | dropout={DROPOUT} | "
        f"lr={LR} | weight_decay={WEIGHT_DECAY}"
    )
    print(f"[INFO] epochs={EPOCHS} | patience={PATIENCE} | samples/class={SAMPLES_PER_CLASS}\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(train_production, d, s, c): (d, s, c)
            for d, s, c in jobs
        }
        for future in as_completed(futures):
            d, s, c = futures[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SEC)
                all_results.append(result)
                print(
                    f"[DONE] {c} d={d} s={s} → "
                    f"val F1: {result['best_val_macro_f1']:.4f} | "
                    f"test F1: {result['best_test_macro_f1']:.4f} | "
                    f"loss: {result['best_loss']:.4f}",
                    flush=True,
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] {c} d={d} s={s} → "
                    f"limite {JOB_TIMEOUT_SEC // 3600}h superato",
                    flush=True,
                )
            except Exception:
                print(
                    f"[ERROR] {c} d={d} s={s} →\n{traceback.format_exc()}",
                    flush=True,
                )

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['compressor', 'd', 'seed', 'best_loss',
                 'best_val_macro_f1', 'best_test_macro_f1']]
        df = df.sort_values(
            ["compressor", "d", "seed"], ascending=[True, False, True]
        ).reset_index(drop=True)
        df.to_csv("experiments_classical/classical_summary.csv", index=False)

        # Merge log CSV per-worker
        history_files = [
            f"experiments_classical/history/log_{c}_d{d}_s{s}.csv"
            for d, s, c in jobs
            if os.path.exists(f"experiments_classical/history/log_{c}_d{d}_s{s}.csv")
        ]
        if history_files:
            pd.concat(
                [pd.read_csv(f) for f in history_files], ignore_index=True
            ).to_csv("experiments_classical/classical_log.csv", index=False)
            print("[INFO] Log unificato → experiments_classical/classical_log.csv")

        print("\n[DONE] Training classico completato.")
        print(df.to_string(index=False))
    else:
        print(
            "\n[WARNING] Nessun risultato. "
            "Controllare experiments_classical/logs/ per i traceback."
        )


if __name__ == "__main__":
    main()