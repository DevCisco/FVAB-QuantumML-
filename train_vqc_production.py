# ---------------------------------------------------------------------------
# Obbligatorio su Windows come PRIMA istruzione eseguibile del modulo:
# impedisce che i worker figli rilancino main() ricorsivamente.
# Deve precedere qualsiasi import che usi multiprocessing internamente.
# ---------------------------------------------------------------------------
import multiprocessing
multiprocessing.freeze_support()

import logging
import os
import time
import traceback
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader, Subset

from hybrid_engine import HybridModel
from data_loader import get_data_loaders

# ---------------------------------------------------------------------------
# NFT — Nakanishi-Fujii-Todo optimizer
# ---------------------------------------------------------------------------
# NFT è progettato specificamente per VQC con gate parametrici sinusoidali
# (RY, RZ come in RealAmplitudes). Ottimizza un parametro alla volta
# sfruttando la struttura trigonometrica esatta della loss rispetto a
# ciascun gate — trova il minimo analitico per quel parametro con sole
# 2-3 valutazioni, senza stimare gradienti o il Quantum Geometric Tensor.
#
# Non richiede fidelity, SamplerV2 né ComputeUncompute — elimina
# le dipendenze da API Qiskit instabili tra versioni.
#
# Con il nuovo HybridModel (DirectVQC + classifier lineare):
#   - qweights: 16 parametri (RealAmplitudes n_layers=3, n_qubits=4)
#   - classifier: 4*4+4 = 20 parametri
#   - totale ottimizzabile: 36 parametri
#   - eval per sweep NFT: 36 parametri × ~3 eval = ~108 eval/epoca
#   - con MAX_EVALS_NFT=300: ~2-3 sweep completi per epoca
from qiskit_algorithms.optimizers import NFT


# ---------------------------------------------------------------------------
# Logging su file — ogni worker scrive su file separato
# ---------------------------------------------------------------------------
os.makedirs("experiments/logs", exist_ok=True)

def get_logger(d: int, seed: int) -> logging.Logger:
    name   = f"worker_d{d}_s{seed}"
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
# Costanti globali
# ---------------------------------------------------------------------------
DIMS          = [32, 16, 8, 4]
SEEDS         = [11, 17, 29]
BACKBONE      = 'resnet'
EPOCHS        = 5

# NFT non usa maxiter ma valutazioni totali della funzione obiettivo.
# Con 36 parametri: 1 sweep = ~108 eval → MAX_EVALS_NFT=300 ≈ ~2-3 sweep/epoca.
MAX_EVALS_NFT = 300

MAX_WORKERS   = 4

N_CLASSES         = 4
N_QUBITS          = N_CLASSES
SAMPLES_PER_CLASS = 8

JOB_TIMEOUT_SEC = 8 * 3600

# Lock per scrittura CSV condiviso — protegge da race condition con MAX_WORKERS > 1
_csv_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Estrazione label dal loader — robusto a qualsiasi wrapper dataset
# ---------------------------------------------------------------------------
def extract_labels_from_loader(loader: DataLoader) -> np.ndarray:
    all_labels = []
    with torch.no_grad():
        for _, labels in loader:
            all_labels.append(labels.cpu().numpy().ravel())
    return np.concatenate(all_labels).astype(int)


# ---------------------------------------------------------------------------
# Subset bilanciato per classe
# ---------------------------------------------------------------------------
def make_balanced_subset(
    dataset, all_labels: np.ndarray, samples_per_class: int, seed: int
) -> Subset:
    rng = np.random.default_rng(seed)
    selected_indices = []
    for cls in range(N_CLASSES):
        cls_indices = np.where(all_labels == cls)[0]
        if len(cls_indices) == 0:
            raise ValueError(
                f"Classe {cls} assente nel dataset. "
                f"Classi presenti: {np.unique(all_labels).tolist()}"
            )
        replace = len(cls_indices) < samples_per_class
        chosen  = rng.choice(cls_indices, size=samples_per_class, replace=replace)
        selected_indices.extend(chosen.tolist())
    return Subset(dataset, selected_indices)


def make_balanced_loader(
    dataset, all_labels: np.ndarray, samples_per_class: int, seed: int
) -> DataLoader:
    subset = make_balanced_subset(dataset, all_labels, samples_per_class, seed)
    return DataLoader(
        subset,
        batch_size=N_CLASSES * samples_per_class,
        shuffle=True,
    )


# ---------------------------------------------------------------------------
# Pesi per classe
# ---------------------------------------------------------------------------
def compute_class_weights(all_labels: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.squeeze().long().to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss / len(loader), 100 * correct / total


# ---------------------------------------------------------------------------
# Bridge NFT ↔ PyTorch — solo parametri trainabili
# ---------------------------------------------------------------------------
def get_trainable_params(model) -> np.ndarray:
    params = [
        p.detach().cpu().numpy().ravel()
        for p in model.parameters()
        if p.requires_grad
    ]
    if not params:
        raise RuntimeError(
            "Nessun parametro con requires_grad=True. "
            "Il backbone non è stato congelato correttamente in HybridModel."
        )
    return np.concatenate(params)


def set_trainable_params(model, flat_params: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.copy_(
                torch.tensor(
                    flat_params[offset:offset + numel], dtype=p.dtype
                ).reshape(p.shape)
            )
            offset += numel


def make_loss_fn(model, images, labels, criterion):
    """
    Closure NFT-compatibile (stessa firma: callable(params: np.ndarray) → float).
    NaN/Inf dalla simulazione → restituisce 1e6 invece di propagare.
    """
    def loss_fn(params: np.ndarray) -> float:
        set_trainable_params(model, params)
        model.train()
        with torch.no_grad():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        val = float(loss.item())
        return val if np.isfinite(val) else 1e6
    return loss_fn


def safe_result_fun(result) -> float:
    """Estrae result.fun gestendo il caso None di NFT non convergente."""
    if result.fun is None:
        return float('nan')
    val = float(result.fun)
    return val if np.isfinite(val) else float('nan')





# ---------------------------------------------------------------------------
# Scrittura CSV thread-safe
# ---------------------------------------------------------------------------
def append_to_csv(log_file: str, df: pd.DataFrame) -> None:
    with _csv_lock:
        header = not os.path.exists(log_file)
        df.to_csv(log_file, mode='a', header=header, index=False)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def train_production(d: int, seed: int, backbone: str = BACKBONE) -> dict:
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed)
    logger.info(f"Avvio worker d={d} seed={seed} backbone={backbone}")

    # — Dati -----------------------------------------------------------------
    try:
        train_loader_full, val_loader, _ = get_data_loaders(
            seed=seed, batch_size=32
        )
    except Exception:
        logger.error(f"Errore get_data_loaders:\n{traceback.format_exc()}")
        raise

    try:
        train_labels = extract_labels_from_loader(train_loader_full)
        logger.info(
            f"Label estratte: {len(train_labels)} campioni | "
            f"classi: {np.unique(train_labels).tolist()}"
        )
    except Exception:
        logger.error(f"Errore estrazione label:\n{traceback.format_exc()}")
        raise

    try:
        class_weights   = compute_class_weights(train_labels).to(device)
        criterion       = nn.CrossEntropyLoss(weight=class_weights)
        balanced_loader = make_balanced_loader(
            train_loader_full.dataset,
            all_labels=train_labels,
            samples_per_class=SAMPLES_PER_CLASS,
            seed=seed,
        )
    except Exception:
        logger.error(f"Errore preparazione dataset:\n{traceback.format_exc()}")
        raise

    # — Modello --------------------------------------------------------------
    try:
        model = HybridModel(
            {'d_latent': d, 'n_qubits': N_QUBITS, 'n_layers': 3, 'seed': seed},
            backbone_type=backbone,
        ).to(device)
    except Exception:
        logger.error(f"Errore HybridModel init:\n{traceback.format_exc()}")
        raise

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri ottimizzabili (VQC): {n_trainable}")
    print(f"[d={d} s={seed}] parametri ottimizzabili: {n_trainable}", flush=True)

    # — NFT (Nakanishi-Fujii-Todo) -------------------------------------------
    # NFT ottimizza un parametro alla volta sfruttando la struttura sinusoidale
    # esatta dei gate RY in RealAmplitudes — nessuna stima di gradiente,
    # nessuna fidelity, nessuna dipendenza da SamplerV2 o AerSimulator.
    #
    # Con 36 parametri ottimizzabili (16 qweights + 20 classifier):
    #   1 sweep = ~108 valutazioni (36 param × 3 eval/param)
    #   MAX_EVALS_NFT=300 → ~2-3 sweep completi per epoca
    optimizer = NFT(maxfev=MAX_EVALS_NFT)

    best_val_acc = 0.0
    best_loss    = float('inf')
    history      = []
    os.makedirs("experiments/models", exist_ok=True)

    print(
        f">>> Inizio NFT: {backbone.upper()} + VQC | "
        f"d={d} seed={seed} | maxfev={MAX_EVALS_NFT}",
        flush=True,
    )
    logger.info(f"Inizio training loop NFT maxfev={MAX_EVALS_NFT}")

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            images, labels_batch = next(iter(balanced_loader))
            images       = images.to(device)
            labels_batch = labels_batch.squeeze().long().to(device)

            loss_fn = make_loss_fn(model, images, labels_batch, criterion)
            x0      = get_trainable_params(model)

            # NFT.minimize: stessa firma callable (params: np.ndarray) → float
            result  = optimizer.minimize(fun=loss_fn, x0=x0)
            set_trainable_params(model, result.x)

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        except Exception:
            logger.error(f"Epoch {epoch+1} fallita:\n{traceback.format_exc()}")
            print(
                f"[WARN] d={d} seed={seed} | Epoch {epoch+1} fallita — continuo",
                flush=True,
            )
            history.append(
                [epoch + 1, d, seed, backbone, float('nan'), float('nan')]
            )
            continue

        train_loss = safe_result_fun(result)
        elapsed    = time.time() - t0

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | Loss={train_loss:.4f} | "
            f"ValAcc={val_acc:.2f}% | t={elapsed:.1f}s"
        )
        print(
            f"d={d} seed={seed} | Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_loss    = train_loss
            ckpt_path    = (
                f"experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth"
            )
            try:
                # weights_only implicito nel salvataggio (non nel caricamento):
                # torch.save non ha weights_only, il flag è solo per torch.load.
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint salvato: {ckpt_path}")
            except Exception:
                logger.error(
                    f"Errore salvataggio checkpoint:\n{traceback.format_exc()}"
                )

        history.append([epoch + 1, d, seed, backbone, train_loss, val_acc])

    # — Log CSV thread-safe --------------------------------------------------
    log_file = "experiments/production_log.csv"
    os.makedirs("experiments", exist_ok=True)
    try:
        df_h = pd.DataFrame(
            history,
            columns=['epoch', 'd', 'seed', 'backbone', 'loss', 'val_acc'],
        )
        append_to_csv(log_file, df_h)
    except Exception:
        logger.error(f"Errore scrittura CSV:\n{traceback.format_exc()}")

    logger.info(f"Completato. Best Val Acc: {best_val_acc:.2f}%")
    print(
        f"[OK] d={d} seed={seed} → Best Val Acc: {best_val_acc:.2f}%",
        flush=True,
    )
    return {
        "d": d, "seed": seed, "backbone": backbone,
        "best_loss": best_loss, "best_val_acc": best_val_acc,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("experiments/logs",   exist_ok=True)
    os.makedirs("experiments",        exist_ok=True)

    jobs = [(d, s, BACKBONE) for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} run su {MAX_WORKERS} processi paralleli...")
    print(f"[INFO] Ottimizzatore: NFT | maxfev={MAX_EVALS_NFT} per epoca")
    print(f"[INFO] Log per worker in experiments/logs/\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(train_production, d, s, backbone): (d, s, backbone)
            for d, s, backbone in jobs
        }
        for future in as_completed(futures):
            d, s, backbone = futures[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SEC)
                all_results.append(result)
                print(
                    f"[DONE] d={d} seed={s} → "
                    f"Best Val Acc: {result['best_val_acc']:.2f}%",
                    flush=True,
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] d={d} seed={s} → "
                    f"superato limite {JOB_TIMEOUT_SEC // 3600}h — job annullato",
                    flush=True,
                )
            except Exception:
                print(
                    f"[ERROR] d={d} seed={s} →\n{traceback.format_exc()}",
                    flush=True,
                )

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['d', 'seed', 'backbone', 'best_loss', 'best_val_acc']]
        df = df.sort_values(
            ["d", "seed"], ascending=[False, True]
        ).reset_index(drop=True)
        df.to_csv("experiments/production_summary.csv", index=False)
        print("\n[DONE] Training NFT completato.")
        print(df.to_string(index=False))
    else:
        print(
            "\n[WARNING] Nessun risultato: tutti i job sono falliti. "
            "Controllare experiments/logs/ per i traceback completi."
        )


if __name__ == "__main__":
    main()