# ---------------------------------------------------------------------------
# Windows spawn: multiprocessing.freeze_support() viene chiamata come prima
# istruzione DENTRO main(), subito dopo il guard `if __name__ == "__main__"`.
# ---------------------------------------------------------------------------

import multiprocessing
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
# Nessuna dipendenza da HybridModel/ResNetCompressor a runtime — le feature
# arrivano già pronte dai CSV prodotti da test.py (B1) e b2_b3_training.py
# (B2, B3). Il circuito viene costruito direttamente da QuantumPipeline +
# DirectVQC.
# ---------------------------------------------------------------------------
from quantum_model import QuantumPipeline
from hybrid_engine import DirectVQC
from qiskit_aer.primitives import EstimatorV2

# ---------------------------------------------------------------------------
# NFT — Nakanishi-Fujii-Todo optimizer
# Rispetto ad altri Optimizer di Qiskit, NFT è più robusto e stabile per il training di VQC con data re-uploading.
# Esso ottimizza un parametro alla volta, riducendo la probabilità di divergenza 
# e migliorando la convergenza in scenari con molti parametri.
# ---------------------------------------------------------------------------
from qiskit_algorithms.optimizers import NFT


# ---------------------------------------------------------------------------
# Costanti globali
# ---------------------------------------------------------------------------
DIMS        = [32, 16, 8, 4]
SEEDS       = [11, 17, 29]
COMPRESSORS = ['B1', 'B2', 'B3']
EPOCHS      = 10
PATIENCE    = 3

# Percorsi CSV per ogni compressore — devono corrispondere esattamente a
# quelli prodotti da test.py (B1) e b2_b3_training.py (B2, B3).
COMPRESSOR_PATHS = {
    'B1': "artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv",
    'B2': "artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv",
    'B3': "artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv",
}

N_CLASSES              = 4
N_QUBITS                = N_CLASSES   # N_QUBITS = N_CLASSES: i readout VQC sono i logit
# RealAmplitudes reps per blocco di re-upload. Era 3, ora 2 — allineato alla
# raccomandazione del documento iniziale ("Entangling budget suggerito":
# al massimo due layer entanglianti per l'ansatz condiviso). Con reps=2:
# 4*(2+1)=12 pesi per blocco. Da quando i pesi sono CONDIVISI tra i cicli
# di re-upload (vedi quantum_model.py), questo è anche il numero TOTALE
# di pesi variazionali del VQC, indipendente da d_latent.
N_LAYERS                = 2
SAMPLES_PER_CLASS       = 8
SAMPLES_POOL_PER_CLASS  = 64

EVAL_BATCH_SIZE = 512
JOB_TIMEOUT_SEC = 8 * 3600

# ---------------------------------------------------------------------------
# Range di scaling per l'encoding angolare RY.
#
# Il documento iniziale specifica letteralmente [0, 2π]. Con un singolo
# RY(θ), però, l'expectation value <Z> = cos(θ) NON è iniettivo su [0,2π]:
# θ=0.5 e θ=2π-0.5 producono lo STESSO <Z>=cos(0.5)=0.8776 — due valori di
# feature distinti diventano indistinguibili per il circuito. Con [0,π],
# cos(θ) è strettamente monotona (iniettiva).
#
# Evidenza empirica: con [0,2π] la macro-F1 di test è crollata di 25-30
# punti su tutti i compressori (B1/B2/B3) in una run precedente. Default
# qui: π. Cambiare a `2 * np.pi` per testare la lettura letterale del
# documento nel contesto del nuovo circuito multi-ciclo (ogni ciclo
# applica lo STESSO ansatz condiviso — resta da verificare empiricamente
# se questo disambigua parzialmente il fold di cos(θ), non è assunto).
ENCODING_SCALE_MAX = 2 * np.pi


# ---------------------------------------------------------------------------
# Budget NFT scalato con il numero di parametri trainabili
#
# Con pesi ansatz INDIPENDENTI per ciclo di re-upload (vedi quantum_model.py
# — la variante a pesi condivisi è stata testata e scartata: verificata
# corretta a livello di conteggio parametri, ma empiricamente peggiore,
# macro-F1 sotto il livello casuale), il numero di parametri variazionali
# del VQC torna a scalare con d_latent: d=32 → più cicli → più parametri,
# d=4 → 1 ciclo → meno parametri.
#
# NFT ottimizza un parametro alla volta: con un budget fisso, le
# configurazioni con più parametri (d alti) ricevono proporzionalmente
# meno sweep completi e convergono peggio. La funzione scala il budget
# con un floor a MAX_EVALS_NFT_BASE (300), così i casi a bassa d (pochi
# parametri) non vengono penalizzati, solo le combinazioni con più cicli
# di re-upload ricevono un budget maggiore.
#
# TUNING (tentativo di ridurre il gap VQC-classico, entro i vincoli del
# mandato Team B — nessuna modifica a encoding/ansatz, solo iperparametri
# di ottimizzazione, coperti dalla clausola di flessibilità del documento):
# NFT_TARGET_SWEEPS 3→5 e EPOCHS 10→15 / PATIENCE 3→5 sopra. Aumenta il
# tempo di calcolo per le combinazioni a d alta di circa 1,7×, con un
# fattore aggiuntivo fino a 1,5× dalle epoche extra (meno in pratica,
# per via dell'early stopping). Se il tempo totale rischia di superare
# i limiti disponibili, ridurre prima EPOCHS, poi NFT_TARGET_SWEEPS.
MAX_EVALS_NFT_BASE = 300
NFT_TARGET_SWEEPS  = 3


def get_max_evals_nft(n_trainable_params: int) -> int:
    """Calcola maxfev sufficiente per almeno NFT_TARGET_SWEEPS sweep completi."""
    return max(MAX_EVALS_NFT_BASE, n_trainable_params * NFT_TARGET_SWEEPS)


# ---------------------------------------------------------------------------
# Logging per worker
# ---------------------------------------------------------------------------
def get_logger(d: int, seed: int, compressor: str) -> logging.Logger:
    os.makedirs("experiments/logs", exist_ok=True)
    name   = f"worker_{compressor}_d{d}_s{seed}"
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
# Lettura CSV — restituisce feature RAW (non scalate), come numpy array.
# Lo scaling va fittato SOLO sul train e applicato identicamente a val/test
# (vedi fit_scaler/apply_scaler) — non più uno scaling per-campione.
# ---------------------------------------------------------------------------
def load_features_from_csv(split: str, d: int, seed: int, compressor: str) -> tuple:
    """
    Legge le feature d-dim RAW (non scalate) per il compressore specificato.

    Returns:
        X (ndarray float32, shape (N, d)): feature raw.
        y (ndarray int64,   shape (N,)  ): label corrispondenti.
    """
    path = COMPRESSOR_PATHS[compressor].format(split=split, d=d, seed=seed)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV non trovato: {path}\n"
            f"Per B1: eseguire test.py\n"
            f"Per B2/B3: eseguire b2_b3_training.py"
        )

    df        = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Scaler min-max fittato SOLO su train, applicato identicamente a val/test
#
# FIX rispetto alla versione precedente: lo scaling era per-CAMPIONE (ogni
# vettore scalato sul proprio min/max) — non un vero scaler fittato sul
# training set come richiesto dal documento ("Scaling: min-max su train",
# "Applicazione scaling: stesso scaler applicato a val/test", lo stesso
# principio già usato per PCA/AE). Ora è un vero scaler globale PER-FEATURE
# (una coppia min/max per ciascuna delle d dimensioni, calcolata sul train).
# ---------------------------------------------------------------------------
def fit_scaler(u_train: np.ndarray) -> tuple:
    """Calcola min/max per-feature sul training set. Returns (min_vec, max_vec), shape (d,)."""
    return u_train.min(axis=0), u_train.max(axis=0)


def apply_scaler(u: np.ndarray, min_vec: np.ndarray, max_vec: np.ndarray,
                  scale_max: float = ENCODING_SCALE_MAX) -> np.ndarray:
    """Applica lo scaler fittato su train a un batch qualsiasi (train/val/test)."""
    range_ = max_vec - min_vec
    range_ = np.where(range_ < 1e-8, 1.0, range_)   # feature costanti → range fittizio, evita /0
    return (u - min_vec) / range_ * scale_max


def pad_features(u: np.ndarray, target_width: int) -> np.ndarray:
    """
    Zero-padding a destra fino a target_width colonne — richiesto dal
    documento quando d non è multiplo di n_qubit. Con D={32,16,8,4} e
    n_qubits=4 il padding è sempre 0 (divisione esatta), ma la funzione
    resta generica e corretta per qualunque combinazione.
    """
    current_width = u.shape[1]
    if current_width >= target_width:
        return u
    pad_width = target_width - current_width
    return np.pad(u, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)


# ---------------------------------------------------------------------------
# Pool e campionamento bilanciato — operano su tensori già scalati/paddati
# ---------------------------------------------------------------------------
def make_balanced_pool(
    u_full: torch.Tensor, y_full: np.ndarray, samples_per_class: int, seed: int
) -> tuple:
    rng = np.random.default_rng(seed)
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_full == cls)[0]
        if len(cls_idx) == 0:
            raise ValueError(f"Classe {cls} assente. Presenti: {np.unique(y_full).tolist()}")
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())
    idx_t = torch.tensor(indices, dtype=torch.long)
    return u_full[idx_t], y_full[np.array(indices)]


def sample_balanced_batch(
    u_pool: torch.Tensor, y_pool: np.ndarray, samples_per_class: int, rng
) -> tuple:
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_pool == cls)[0]
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())
    idx_t = torch.tensor(indices, dtype=torch.long)
    return u_pool[idx_t], y_pool[np.array(indices)]


# ---------------------------------------------------------------------------
# Pesi per classe — formula sklearn balanced
# ---------------------------------------------------------------------------
def compute_class_weights(all_labels: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
    weights = len(all_labels) / (N_CLASSES * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Bridge NFT ↔ PyTorch
# ---------------------------------------------------------------------------
def get_trainable_params(modules: list) -> np.ndarray:
    all_params = []
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                all_params.append(p.detach().cpu().numpy().ravel())
    if not all_params:
        raise RuntimeError("Nessun parametro trainable nei moduli forniti.")
    return np.concatenate(all_params)


def set_trainable_params(modules: list, flat_params: np.ndarray) -> None:
    flat_t = torch.from_numpy(flat_params).float()
    offset = 0
    with torch.no_grad():
        for module in modules:
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                numel = p.numel()
                p.copy_(flat_t[offset:offset + numel].reshape(p.shape))
                offset += numel


# ---------------------------------------------------------------------------
# Closure NFT — opera su feature GIÀ scalate e paddate (nessuna proiezione
# intermedia: il circuito con re-uploading consuma direttamente il vettore
# d-dim, niente più feature_selector).
# ---------------------------------------------------------------------------
def make_loss_fn(modules: list, u_batch_scaled: torch.Tensor,
                  labels_t: torch.Tensor, criterion) -> callable:
    vqc, classifier = modules

    def loss_fn(params: np.ndarray) -> float:
        set_trainable_params(modules, params)
        vqc.train()
        classifier.train()
        with torch.no_grad():
            q_out   = vqc(u_batch_scaled)
            outputs = classifier(q_out)
            loss    = criterion(outputs, labels_t)
        val = float(loss.item())
        return val if np.isfinite(val) else 1e6

    return loss_fn


def safe_result_fun(result) -> float:
    if result.fun is None:
        return float('nan')
    val = float(result.fun)
    return val if np.isfinite(val) else float('nan')


def evaluate_on_features(modules: list, u_scaled: torch.Tensor, labels_np: np.ndarray,
                          criterion, device, batch_size: int = EVAL_BATCH_SIZE) -> tuple:
    """Valuta su feature già scalate/paddate, in mini-batch."""
    vqc, classifier = modules
    vqc.eval()
    classifier.eval()

    eval_criterion = nn.CrossEntropyLoss(weight=criterion.weight, reduction='sum')

    all_preds  = []
    total_loss = 0.0
    n_samples  = len(labels_np)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end     = min(start + batch_size, n_samples)
            u_batch = u_scaled[start:end]
            y_batch = torch.tensor(labels_np[start:end], dtype=torch.long, device=device)

            q_out   = vqc(u_batch)
            outputs = classifier(q_out)
            loss    = eval_criterion(outputs, y_batch)

            total_loss += loss.item()
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())

    preds        = np.concatenate(all_preds)
    avg_loss     = total_loss / n_samples
    acc          = float((preds == labels_np).mean() * 100)
    macro_f1     = float(sk_f1(labels_np, preds, average='macro', zero_division=0))
    per_class_f1 = sk_f1(labels_np, preds, average=None, zero_division=0).tolist()
    return avg_loss, acc, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# CSV per-worker
# ---------------------------------------------------------------------------
def save_worker_csv(history: list, d: int, seed: int, compressor: str) -> None:
    os.makedirs("experiments/history", exist_ok=True)
    path = f"experiments/history/log_{compressor}_d{d}_s{seed}.csv"
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
    Addestra VQC (con data re-uploading) + classificatore per una tripla
    (d, seed, compressor). Nessun feature_selector: il circuito consuma
    direttamente il vettore d-dim scalato/paddato.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed, compressor)
    logger.info(f"Avvio job d={d} seed={seed} compressor={compressor}")

    # — Circuito quantistico + VQC + classifier -------------------------------
    # Costruito PRIMA del caricamento feature: serve n_encoding_padded per
    # il padding.
    try:
        q_pipeline = QuantumPipeline(n_qubits=N_QUBITS, d_latent=d, n_layers=N_LAYERS)
        circuit    = q_pipeline.build_circuit()

        vqc = DirectVQC(
            circuit     = circuit,
            features_pv = q_pipeline.features,
            weights_pv  = q_pipeline.weights,
            n_qubits    = N_QUBITS,
            estimator   = EstimatorV2(),
        ).to(device)

        classifier = nn.Linear(N_QUBITS, N_CLASSES).to(device)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    modules = [vqc, classifier]

    n_total = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
    if n_total == 0:
        raise RuntimeError(f"d={d} seed={seed} {compressor}: nessun parametro trainable.")

    logger.info(
        f"Circuito: {q_pipeline.n_cycles} cicli re-upload | "
        f"VQC+classifier params={n_total}"
    )
    print(
        f"[{compressor} d={d} s={seed}] cicli={q_pipeline.n_cycles} | "
        f"param totali={n_total}",
        flush=True,
    )

    # — Caricamento feature RAW + fit scaler SOLO su train -------------------
    try:
        X_train_raw, y_train = load_features_from_csv('train', d, seed, compressor)
        X_val_raw,   y_val   = load_features_from_csv('val',   d, seed, compressor)
        X_test_raw,  y_test  = load_features_from_csv('test',  d, seed, compressor)

        min_vec, max_vec = fit_scaler(X_train_raw)
        target_width     = q_pipeline.n_encoding_padded

        X_train_s = pad_features(apply_scaler(X_train_raw, min_vec, max_vec), target_width)
        X_val_s   = pad_features(apply_scaler(X_val_raw,   min_vec, max_vec), target_width)
        X_test_s  = pad_features(apply_scaler(X_test_raw,  min_vec, max_vec), target_width)

        u_train = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        u_val   = torch.tensor(X_val_s,   dtype=torch.float32, device=device)
        u_test  = torch.tensor(X_test_s,  dtype=torch.float32, device=device)

        logger.info(
            f"Feature caricate ({compressor}): train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | test {tuple(u_test.shape)} | "
            f"scale_max={ENCODING_SCALE_MAX:.4f}"
        )
        print(
            f"[{compressor} d={d} s={seed}] CSV letti: train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | test {tuple(u_test.shape)}",
            flush=True,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Pool bilanciato dal training set --------------------------------------
    try:
        u_pool, y_pool = make_balanced_pool(
            u_train, y_train, samples_per_class=SAMPLES_POOL_PER_CLASS, seed=seed
        )
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Pesi per classe e criterion --------------------------------------------
    class_weights = compute_class_weights(y_train).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # — NFT — budget scalato sul numero di parametri (ora dipendenti da d) ---
    max_evals = get_max_evals_nft(n_total)
    optimizer = NFT(maxfev=max_evals)
    epoch_rng = np.random.default_rng(seed)

    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0
    best_loss          = float('inf')
    patience_ctr       = 0
    history            = []
    os.makedirs("experiments/models", exist_ok=True)

    n_sweeps = max_evals / n_total
    print(
        f">>> NFT: {compressor} | d={d} seed={seed} | maxfev={max_evals} "
        f"({n_sweeps:.1f} sweep su {n_total} param) | patience={PATIENCE}",
        flush=True,
    )
    logger.info(f"Inizio NFT maxfev={max_evals} ({n_sweeps:.1f} sweep) patience={PATIENCE}")

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            u_batch, y_batch_np = sample_balanced_batch(
                u_pool, y_pool, SAMPLES_PER_CLASS, epoch_rng
            )
            y_batch_t = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            loss_fn = make_loss_fn(modules, u_batch, y_batch_t, criterion)
            result  = optimizer.minimize(fun=loss_fn, x0=get_trainable_params(modules))
            set_trainable_params(modules, result.x)

            val_loss,  val_acc,  val_macro_f1,  _            = evaluate_on_features(
                modules, u_val,  y_val,  criterion, device
            )
            test_loss, test_acc, test_macro_f1, per_class_f1 = evaluate_on_features(
                modules, u_test, y_test, criterion, device
            )

        except Exception:
            logger.error(f"Epoch {epoch+1}:\n{traceback.format_exc()}")
            print(f"[WARN] {compressor} d={d} s={seed} | Epoch {epoch+1} fallita — continuo", flush=True)
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

        train_loss = safe_result_fun(result)
        elapsed    = time.time() - t0

        log_msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"macro-F1 val: {val_macro_f1:.4f} | macro-F1 test: {test_macro_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"per-class F1: {[f'{v:.3f}' for v in per_class_f1]} | "
            f"t={elapsed:.1f}s"
        )
        logger.info(log_msg)
        print(f"{compressor} d={d} s={seed} | {log_msg}", flush=True)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = test_macro_f1
            best_loss          = train_loss
            patience_ctr       = 0
            ckpt_path = f"experiments/models/best_vqc_{compressor}_d{d}_seed{seed}.pth"
            try:
                torch.save(
                    {
                        'vqc':        vqc.state_dict(),
                        'classifier': classifier.state_dict(),
                        'min_vec':    min_vec,
                        'max_vec':    max_vec,
                        'd': d, 'seed': seed, 'compressor': compressor, 'epoch': epoch + 1,
                    },
                    ckpt_path,
                )
                logger.info(
                    f"Checkpoint → {ckpt_path} "
                    f"(val F1={val_macro_f1:.4f} | test F1={test_macro_f1:.4f})"
                )
            except Exception:
                logger.error(f"Errore checkpoint:\n{traceback.format_exc()}")
        else:
            patience_ctr += 1
            logger.info(f"No improvement val: {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping a epoch {epoch+1}")
                print(
                    f"[STOP] {compressor} d={d} s={seed} | Early stopping a epoch {epoch+1} "
                    f"(patience={PATIENCE})",
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

    try:
        save_worker_csv(history, d, seed, compressor)
    except Exception:
        logger.error(f"Errore CSV:\n{traceback.format_exc()}")

    if not np.isfinite(best_loss):
        best_loss = float('nan')

    logger.info(
        f"Completato. Best val macro-F1: {best_val_macro_f1:.4f} | "
        f"Test macro-F1 al miglior val: {best_test_macro_f1:.4f}"
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
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs("experiments/models",  exist_ok=True)
    os.makedirs("experiments/logs",    exist_ok=True)
    os.makedirs("experiments/history", exist_ok=True)

    jobs = [(d, s, c) for c in COMPRESSORS for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {os.cpu_count() or 1} core disponibili...")
    print(f"[INFO] NFT | maxfev_base={MAX_EVALS_NFT_BASE}, scalato con n_param | "
          f"epochs={EPOCHS} | patience={PATIENCE}")
    print(f"[INFO] Encoding: data re-uploading, scale_max={ENCODING_SCALE_MAX:.4f} "
          f"({'π' if abs(ENCODING_SCALE_MAX - np.pi) < 1e-6 else '2π' if abs(ENCODING_SCALE_MAX - 2*np.pi) < 1e-6 else '?'})")
    print(f"[INFO] Nessun feature_selector — il circuito consuma direttamente le d feature.\n")

    MAX_WORKERS = 4
    all_results = []

    with ProcessPoolExecutor(max_workers=min(len(jobs), MAX_WORKERS)) as executor:
        futures = {
            executor.submit(train_production, d, s, c): (d, s, c)
            for d, s, c in jobs
        }
        for future in as_completed(futures):
            d, s, c = futures[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SEC)
                all_results.append(result)
                best_loss_str = (
                    f"{result['best_loss']:.4f}" if np.isfinite(result['best_loss']) else "nan"
                )
                print(
                    f"[DONE] {c} d={d} s={s} → "
                    f"val F1: {result['best_val_macro_f1']:.4f} | "
                    f"test F1: {result['best_test_macro_f1']:.4f} | "
                    f"loss: {best_loss_str}",
                    flush=True,
                )
            except TimeoutError:
                print(f"[TIMEOUT] {c} d={d} s={s} → limite {JOB_TIMEOUT_SEC // 3600}h", flush=True)
            except Exception:
                print(f"[ERROR] {c} d={d} s={s} →\n{traceback.format_exc()}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['compressor', 'd', 'seed', 'best_loss',
                 'best_val_macro_f1', 'best_test_macro_f1']]
        df = df.sort_values(["compressor", "d", "seed"], ascending=[True, False, True]).reset_index(drop=True)
        df.to_csv("experiments/production_summary.csv", index=False)

        history_files = [
            f"experiments/history/log_{c}_d{d}_s{s}.csv"
            for c, d, s in [(c, d, s) for c in COMPRESSORS for d in DIMS for s in SEEDS]
            if os.path.exists(f"experiments/history/log_{c}_d{d}_s{s}.csv")
        ]
        if history_files:
            pd.concat([pd.read_csv(f) for f in history_files], ignore_index=True) \
                .to_csv("experiments/production_log.csv", index=False)
            print("[INFO] Log unificato → experiments/production_log.csv")

        print("\n[DONE] Training NFT completato.")
        print(df.to_string(index=False))
    else:
        print("\n[WARNING] Nessun risultato. Controllare experiments/logs/ per i traceback.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()