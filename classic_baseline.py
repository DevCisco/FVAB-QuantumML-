# ---------------------------------------------------------------------------
# Obbligatorio su Windows come PRIMA istruzione eseguibile del modulo.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import f1_score as sk_f1

# ---------------------------------------------------------------------------
# classical_baseline.py
# ---------------------------------------------------------------------------
# Baseline classica MLP per confronto diretto con il VQC (train_vqc_production.py).
#
# Differenze architetturali rispetto al VQC — e perché rendono il confronto valido:
#
#   VQC: backbone → PCA(d) → troncato a min(d,4) qubit → RealAmplitudes → head(4,4)
#   MLP: PCA(d) → Linear(d,32) → ReLU → Dropout(0.3) → Linear(32,4)
#
# Il MLP riceve tutte le d componenti PCA (vantaggio rispetto ai 4 qubit del VQC).
# Se il VQC compete con il MLP a d=8 o d=4, è un risultato sperimentalmente forte.
#
# Vantaggi del MLP sulla velocità:
#   - Nessun simulatore quantistico → ogni epoch dura millisecondi
#   - Backpropagation standard → nessun overhead NFT o parameter-shift
#   - EPOCHS=100 con PATIENCE=10 garantisce convergenza completa
#
# CSV output: identico a train_vqc_production.py (+ per_class_f1_val / _test)
# per confronto diretto con pd.concat o merge su ['d','seed','epoch'].
#
# Feature sorgente: artifacts/datasetresnet/features/b1/{split}_ids_dimensione{d}_seed{s}_.csv
# Prodotte da test.py (backbone → StandardScaler → PCA, fit solo su train).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Costanti globali
# ---------------------------------------------------------------------------
DIMS     = [32, 16, 8, 4]
SEEDS    = [11, 17, 29]
EPOCHS   = 100          # backprop è rapido: 100 epoche convergono in secondi
PATIENCE = 10           # più alto di VQC — backprop ha oscillazioni più frequenti
MAX_WORKERS = 4

N_CLASSES         = 4
SAMPLES_PER_CLASS = 8   # uguale al VQC per comparabilità sul training set

HIDDEN_DIM   = 32       # identico alla dimensione nascosta nell'head VQC
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4     # L2 regularization: compensa la mancanza di Dropout sul VQC

MODEL_NAME = 'mlp'      # usato nella colonna 'backbone' del CSV per distinguere dal VQC

JOB_TIMEOUT_SEC = 1 * 3600   # 1h: più che sufficiente (ogni job dura secondi/minuti)


# ---------------------------------------------------------------------------
# Logging su file — identico a train_vqc_production.py
# ---------------------------------------------------------------------------
def get_logger(d: int, seed: int) -> logging.Logger:
    os.makedirs("experiments/logs", exist_ok=True)
    name   = f"classical_d{d}_s{seed}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        f"experiments/logs/{name}.log", mode='w', encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Architettura MLP
# ---------------------------------------------------------------------------
class ClassicalMLP(nn.Module):
    """
    MLP a singolo hidden layer per classificazione 4-class su feature PCA.

    Architettura: Linear(d_in, 32) → ReLU → Dropout(0.3) → Linear(32, 4)

    Un solo hidden layer è la scelta corretta con 32 campioni di training:
    - due hidden layer overfittano quasi certamente su dataset così piccoli
    - Dropout(0.3) fornisce regolarizzazione implicita
    - weight_decay nell'ottimizzatore aggiunge L2 come ulteriore regolarizzatore

    Il modello riceve tutte le d componenti PCA (vantaggio vs VQC che ne usa max 4).
    """
    def __init__(self, d_in: int, hidden: int = HIDDEN_DIM,
                 n_classes: int = N_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Caricamento feature da CSV
# ---------------------------------------------------------------------------
def load_features_csv(d: int, seed: int, split: str) -> tuple:
    """
    Carica le feature PCA d-dim dal CSV prodotto da test.py.

    Percorso: artifacts/datasetresnet/features/b1/{split}_ids_dimensione{d}_seed{seed}_.csv
    Colonne:  feat_0, feat_1, ..., feat_{d-1}, label

    Returns:
        X (Tensor float32, shape (N, d)),
        y (ndarray int,    shape (N,))
    """
    path = (
        f"artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv"
    )
    df        = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = df['label'].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Subset bilanciato per classe (identico a train_vqc_production.py)
# ---------------------------------------------------------------------------
def get_balanced_indices(
    y: np.ndarray, samples_per_class: int, seed: int
) -> np.ndarray:
    """
    Estrae SAMPLES_PER_CLASS indici per classe, stratificati e riproducibili.

    Con replace=True se una classe ha meno campioni del richiesto (raro ma
    possibile su split di addestramento molto piccoli).
    """
    rng      = np.random.default_rng(seed)
    selected = []
    for cls in range(N_CLASSES):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Classe {cls} assente in split train. "
                f"Classi presenti: {np.unique(y).tolist()}"
            )
        replace = len(idx) < samples_per_class
        chosen  = rng.choice(idx, size=samples_per_class, replace=replace)
        selected.extend(chosen.tolist())
    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Class weights — identici a train_vqc_production.py
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=N_CLASSES).astype(float)
    weights = len(y) / (N_CLASSES * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation su feature pre-caricate — macro-F1 prima di loss (direttiva)
# ---------------------------------------------------------------------------
def evaluate_mlp(
    model:     nn.Module,
    X:         torch.Tensor,
    y_np:      np.ndarray,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple:
    """
    Valuta il MLP su un split (val o test).

    Restituisce nell'ordine: macro_f1, loss, acc, per_class_f1
    macro-F1 è il primo valore restituito: è la metrica principale in presenza
    di classi sbilanciate (8 campioni/classe nel training).

    Args:
        model     : ClassicalMLP in modalità eval.
        X         : feature Tensor (N, d), già in CPU.
        y_np      : label array int (N,).
        criterion : CrossEntropyLoss con class weights.
        device    : device di esecuzione.

    Returns:
        (macro_f1: float, loss: float, acc: float, per_class_f1: list)
    """
    model.eval()
    y_t = torch.tensor(y_np, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(X.to(device))
        loss    = criterion(outputs, y_t)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()

    acc          = float((preds == y_np).mean() * 100)
    macro_f1     = float(sk_f1(y_np, preds, average='macro',    zero_division=0))
    per_class_f1 = sk_f1(y_np, preds, average=None, zero_division=0).tolist()

    return macro_f1, loss.item(), acc, per_class_f1


# ---------------------------------------------------------------------------
# Salvataggio CSV per-worker (struttura identica a train_vqc_production.py
# + colonne val e test separati per confronto diretto)
# ---------------------------------------------------------------------------
def save_worker_csv(history: list, d: int, seed: int) -> None:
    """
    Salva il log di training in experiments/history/classical_d{d}_s{seed}.csv.

    Colonne identiche a train_vqc_production.py più per_class_f1_val e
    per_class_f1_test separati per confronto diretto.

    Il merge con il log VQC è banale:
        pd.concat([df_vqc, df_classical]).groupby(['d','seed','epoch'])
    """
    os.makedirs("experiments/history", exist_ok=True)
    path = f"experiments/history/classical_d{d}_s{seed}.csv"
    pd.DataFrame(
        history,
        columns=[
            'epoch', 'd', 'seed', 'backbone',
            'train_loss',
            'val_macro_f1', 'val_loss', 'val_acc', 'per_class_f1_val',
            'test_macro_f1', 'test_loss', 'test_acc', 'per_class_f1_test',
        ],
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Worker — un job per coppia (d, seed)
# ---------------------------------------------------------------------------
def train_classical(d: int, seed: int) -> dict:
    """
    Addestra il MLP classico per una coppia (d, seed) e restituisce le metriche.

    Struttura speculare a train_production() in train_vqc_production.py:
    stesse fasi, stessi CSV, stessa logica di early stopping e checkpoint.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed)
    logger.info(f"Avvio job d={d} seed={seed}")

    # — Caricamento feature da CSV -------------------------------------------
    # Nessun backbone, nessuna immagine: le feature PCA sono già su disco.
    # Vantaggio rispetto al VQC: il MLP riceve tutte le d componenti PCA.
    try:
        X_train_full, y_train_full = load_features_csv(d, seed, 'train')
        X_val,        y_val_np     = load_features_csv(d, seed, 'val')
        X_test,       y_test_np    = load_features_csv(d, seed, 'test')
    except FileNotFoundError:
        logger.error(
            f"CSV non trovato per d={d} seed={seed}.\n"
            f"Eseguire test.py prima di lanciare classical_baseline.py.\n"
            f"{traceback.format_exc()}"
        )
        raise

    logger.info(
        f"Feature caricate | train: {X_train_full.shape} | "
        f"val: {X_val.shape} | test: {X_test.shape}"
    )

    # — Subset bilanciato training (8 per classe = 32 campioni totali) -------
    # Identico al VQC per comparabilità: stesso numero di campioni di training.
    try:
        bal_idx     = get_balanced_indices(y_train_full.numpy() if isinstance(
                          y_train_full, torch.Tensor) else y_train_full,
                          SAMPLES_PER_CLASS, seed)
        X_train_bal = X_train_full[bal_idx].to(device)
        y_train_bal = torch.tensor(
            y_train_full[bal_idx] if isinstance(y_train_full, np.ndarray)
            else y_train_full.numpy()[bal_idx],
            dtype=torch.long, device=device
        )
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Class weights (stesso schema del VQC) --------------------------------
    y_bal_np     = y_train_bal.cpu().numpy()
    class_weights = compute_class_weights(y_bal_np).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # — Modello e ottimizzatore ----------------------------------------------
    model     = ClassicalMLP(d_in=d).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri MLP totali: {n_params}")
    print(
        f"[d={d} s={seed}] MLP: {n_params} parametri | "
        f"train bal: {len(X_train_bal)} campioni",
        flush=True,
    )

    # — Training loop --------------------------------------------------------
    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0   # snapshot al momento del best val — non usato per selezione
    best_train_loss    = float('inf')
    patience_ctr       = 0
    history            = []
    os.makedirs("experiments/models", exist_ok=True)

    print(
        f">>> MLP: d={d} seed={seed} | "
        f"epochs={EPOCHS} | patience={PATIENCE} | lr={LR} | wd={WEIGHT_DECAY}",
        flush=True,
    )

    for epoch in range(EPOCHS):
        t0 = time.time()

        # — Forward + backward (full-batch: 32 campioni) ---------------------
        # Con dataset così piccolo il full-batch è più stabile del mini-batch.
        # Non serve un DataLoader: X_train_bal è già un Tensor in memoria.
        try:
            model.train()
            optimizer.zero_grad()
            outputs    = model(X_train_bal)
            train_loss = criterion(outputs, y_train_bal)
            train_loss.backward()
            optimizer.step()
            train_loss_val = train_loss.item()

        except Exception:
            logger.error(f"Epoch {epoch+1} train fallita:\n{traceback.format_exc()}")
            history.append([
                epoch+1, d, seed, MODEL_NAME,
                float('nan'),
                float('nan'), float('nan'), float('nan'), '[]',
                float('nan'), float('nan'), float('nan'), '[]',
            ])
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
            continue

        # — Valutazione su VAL -----------------------------------------------
        # macro-F1 val: metrica di selezione modello (checkpoint).
        # Valutazione separata da test — nessun leakage.
        try:
            val_macro_f1, val_loss, val_acc, per_class_f1_val = evaluate_mlp(
                model, X_val, y_val_np, criterion, device
            )
        except Exception:
            logger.error(f"Epoch {epoch+1} eval val:\n{traceback.format_exc()}")
            val_macro_f1, val_loss, val_acc = float('nan'), float('nan'), float('nan')
            per_class_f1_val = []

        # — Valutazione su TEST ----------------------------------------------
        # Stessa procedura di val, stessi metrici, split diverso.
        # Loggato ma NON usato per checkpoint — evita information leakage.
        try:
            test_macro_f1, test_loss, test_acc, per_class_f1_test = evaluate_mlp(
                model, X_test, y_test_np, criterion, device
            )
        except Exception:
            logger.error(f"Epoch {epoch+1} eval test:\n{traceback.format_exc()}")
            test_macro_f1, test_loss, test_acc = float('nan'), float('nan'), float('nan')
            per_class_f1_test = []

        elapsed = time.time() - t0

        # — Stampa: macro-F1 prima di loss (stessa direttiva del VQC) --------
        log_msg = (
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Val  macro-F1: {val_macro_f1:.4f} | Val  Loss: {val_loss:.4f} | Val  Acc: {val_acc:.2f}% | "
            f"per-class: {[f'{v:.3f}' for v in per_class_f1_val]} | "
            f"Test macro-F1: {test_macro_f1:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
            f"per-class: {[f'{v:.3f}' for v in per_class_f1_test]} | "
            f"Train Loss: {train_loss_val:.4f} | t={elapsed:.2f}s"
        )
        logger.info(log_msg)
        print(f"d={d} s={seed} | {log_msg}", flush=True)

        # — Checkpoint su best val_macro_f1 ----------------------------------
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = test_macro_f1   # snapshot contestuale
            best_train_loss    = train_loss_val
            patience_ctr       = 0
            ckpt_path = (
                f"experiments/models/best_classical_mlp_d{d}_seed{seed}.pth"
            )
            try:
                torch.save(model.state_dict(), ckpt_path)
                logger.info(
                    f"Checkpoint → {ckpt_path} "
                    f"(val macro-F1={val_macro_f1:.4f} | "
                    f"test macro-F1={test_macro_f1:.4f})"
                )
            except Exception:
                logger.error(f"Errore checkpoint:\n{traceback.format_exc()}")
        else:
            patience_ctr += 1
            logger.info(f"No improvement: {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping a epoch {epoch+1}")
                print(
                    f"[STOP] d={d} s={seed} | "
                    f"Early stopping a epoch {epoch+1} (patience={PATIENCE}) | "
                    f"Best val macro-F1: {best_val_macro_f1:.4f}",
                    flush=True,
                )
                history.append([
                    epoch+1, d, seed, MODEL_NAME,
                    train_loss_val,
                    val_macro_f1,  val_loss,  val_acc,
                    str([f'{v:.3f}' for v in per_class_f1_val]),
                    test_macro_f1, test_loss, test_acc,
                    str([f'{v:.3f}' for v in per_class_f1_test]),
                ])
                break

        history.append([
            epoch+1, d, seed, MODEL_NAME,
            train_loss_val,
            val_macro_f1,  val_loss,  val_acc,
            str([f'{v:.3f}' for v in per_class_f1_val]),
            test_macro_f1, test_loss, test_acc,
            str([f'{v:.3f}' for v in per_class_f1_test]),
        ])

    # — Salvataggio CSV per-worker -------------------------------------------
    try:
        save_worker_csv(history, d, seed)
    except Exception:
        logger.error(f"Errore CSV:\n{traceback.format_exc()}")

    logger.info(
        f"Completato | "
        f"Best val macro-F1: {best_val_macro_f1:.4f} | "
        f"Test macro-F1 al best: {best_test_macro_f1:.4f}"
    )
    print(
        f"[OK] d={d} s={seed} → "
        f"Best val macro-F1: {best_val_macro_f1:.4f} | "
        f"Test macro-F1 al best: {best_test_macro_f1:.4f}",
        flush=True,
    )
    return {
        "d":                   d,
        "seed":                seed,
        "backbone":            MODEL_NAME,
        "best_train_loss":     best_train_loss,
        "best_val_macro_f1":   best_val_macro_f1,
        "best_test_macro_f1":  best_test_macro_f1,
    }


# ---------------------------------------------------------------------------
# Entry point — identico a train_vqc_production.py
# ---------------------------------------------------------------------------
def main():
    os.makedirs("experiments/models",  exist_ok=True)
    os.makedirs("experiments/logs",    exist_ok=True)
    os.makedirs("experiments/history", exist_ok=True)
    os.makedirs("experiments",         exist_ok=True)

    jobs = [(d, s) for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(
        f"[INFO] Modello: MLP({HIDDEN_DIM}) | "
        f"lr={LR} | wd={WEIGHT_DECAY} | dropout={DROPOUT}"
    )
    print(f"[INFO] epochs={EPOCHS} | patience={PATIENCE}")
    print(f"[INFO] Feature: PCA d-dim da artifacts/datasetresnet/features/b1/\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(train_classical, d, s): (d, s)
            for d, s in jobs
        }
        for future in as_completed(futures):
            d, s = futures[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SEC)
                all_results.append(result)
                print(
                    f"[DONE] d={d} s={s} → "
                    f"val macro-F1: {result['best_val_macro_f1']:.4f} | "
                    f"test macro-F1: {result['best_test_macro_f1']:.4f}",
                    flush=True,
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] d={d} s={s} → "
                    f"limite {JOB_TIMEOUT_SEC // 3600}h superato",
                    flush=True,
                )
            except Exception:
                print(
                    f"[ERROR] d={d} s={s} →\n{traceback.format_exc()}",
                    flush=True,
                )

    if not all_results:
        print(
            "\n[WARNING] Nessun risultato. "
            "Controllare experiments/logs/ per i traceback."
        )
        return

    # — Summary CSV ----------------------------------------------------------
    df = pd.DataFrame(all_results)
    df = df[['d', 'seed', 'backbone',
             'best_train_loss', 'best_val_macro_f1', 'best_test_macro_f1']]
    df = df.sort_values(
        ["d", "seed"], ascending=[False, True]
    ).reset_index(drop=True)
    df.to_csv("experiments/classical_summary.csv", index=False)

    # — Merge CSV per-worker in un unico log ---------------------------------
    # Identico a train_vqc_production.py: log_d{d}_s{seed}.csv → production_log.csv
    # Per confronto diretto con il VQC:
    #   pd.concat([pd.read_csv("experiments/production_log.csv"),
    #              pd.read_csv("experiments/classical_log.csv")])
    history_files = [
        f"experiments/history/classical_d{d}_s{s}.csv"
        for d, s in jobs
        if os.path.exists(f"experiments/history/classical_d{d}_s{s}.csv")
    ]
    if history_files:
        merged = pd.concat(
            [pd.read_csv(f) for f in history_files], ignore_index=True
        )
        merged.to_csv("experiments/classical_log.csv", index=False)
        print("[INFO] Log unificato → experiments/classical_log.csv")

    print("\n[DONE] Training baseline classica completato.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()