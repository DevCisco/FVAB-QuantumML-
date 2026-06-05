# ---------------------------------------------------------------------------
# FIX #9: freeze_support e import come PRIME istruzioni — nessun effetto
# collaterale a livello di modulo (os.makedirs era a riga 37 nel vecchio file,
# eseguito anche nei worker spawn durante l'import).
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
from qiskit_algorithms.optimizers import NFT


# ---------------------------------------------------------------------------
# Costanti globali
# ---------------------------------------------------------------------------
DIMS          = [32, 16, 8, 4]
SEEDS         = [11, 17, 29]
BACKBONE      = 'resnet'
EPOCHS        = 10          # aumentato: early stopping gestisce la convergenza
PATIENCE      = 3           # epoche senza miglioramento macro-F1 → stop anticipato
MAX_EVALS_NFT = 300         # valutazioni NFT per epoca
MAX_WORKERS   = 4

N_CLASSES         = 4
N_QUBITS          = N_CLASSES
SAMPLES_PER_CLASS = 8

JOB_TIMEOUT_SEC = 8 * 3600


def get_logger(d: int, seed: int) -> logging.Logger:
    os.makedirs("experiments/logs", exist_ok=True)   # solo quando serve
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
# Estrazione label dal loader
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
                f"Classe {cls} assente. Presenti: {np.unique(all_labels).tolist()}"
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
# Class weights inversamente proporzionali alla frequenza — utile con 8 campioni/classe
# L'obiettivo è penalizzare di più le classi meno rappresentate, bilanciando la loss
# ---------------------------------------------------------------------------
def compute_class_weights(all_labels: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
    weights = len(all_labels) / (N_CLASSES * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation su feature pre-calcolate — macro-F1, per-class F1, accuracy
# ---------------------------------------------------------------------------
def evaluate_on_features(
    model,
    u_scaled: torch.Tensor,
    labels_np: np.ndarray,
    criterion,
    device,
) -> tuple:
    """
    Valuta il modello su feature già compresse e scalate.

    Non ri-esegue il backbone — opera su u_scaled pre-calcolato.
    Restituisce: test_loss, test_acc, macro_f1, per_class_f1.

    macro-F1 è la metrica principale: più informativo dell'accuracy
    con classi sbilanciate e dataset piccoli (8 campioni/classe).
    """
    model.eval()
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
    with torch.no_grad():
        q_out   = model.vqc(u_scaled)
        outputs = model.classifier(q_out)
        loss    = criterion(outputs, labels_t)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()

    acc          = float((preds == labels_np).mean() * 100)
    macro_f1     = float(sk_f1(labels_np, preds, average='macro',    zero_division=0))
    per_class_f1 = sk_f1(labels_np, preds, average=None, zero_division=0).tolist()

    return loss.item(), acc, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# Pre-calcolo feature compresse — PRINCIPALE OTTIMIZZAZIONE DI VELOCITÀ
#
# Il backbone ResNet18 (~11.2M param, congelato) veniva chiamato dentro
# la loss_fn di NFT → 300 volte/epoca × 5 epoche = 1500 chiamate/job.
# Pre-calcolando una volta, scende a 1 chiamata/job.
# Riduzione stimata: 80-90% del tempo per epoca.
#
# Accede a model._compress() e model.scale_features() — metodi di HybridModel.
# In Python non esistono metodi veramente privati: l'underscore è solo
# una convenzione, l'accesso diretto è legale e intenzionale qui.
# ---------------------------------------------------------------------------
def precompute_features(
    model, loader: DataLoader, device
) -> tuple:
    """
    Estrae feature per tutti i campioni del loader con una sola passata backbone.

    Returns:
        u_scaled  (Tensor): feature scalate in [0, π], shape (N, n_encoding)
        labels_np (ndarray): label corrispondenti, shape (N,)
    """
    all_u, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            u = model.scale_features(model._compress(imgs.to(device)))
            all_u.append(u.cpu())
            all_labels.append(lbls.squeeze().long().numpy())
    return (
        torch.cat(all_u, dim=0).to(device),
        np.concatenate(all_labels),
    )


# ---------------------------------------------------------------------------
#  Bridge NFT ↔ PyTorch — torch.from_numpy + copy_
# ---------------------------------------------------------------------------
def get_trainable_params(model) -> np.ndarray:
    params = [
        p.detach().cpu().numpy().ravel()
        for p in model.parameters()
        if p.requires_grad
    ]
    if not params:
        raise RuntimeError(
            "Nessun parametro trainable. "
            "Backbone non congelato correttamente in HybridModel."
        )
    return np.concatenate(params)


def set_trainable_params(model, flat_params: np.ndarray) -> None:
    # torch.from_numpy: zero-copy (condivide il buffer numpy)
    # copy_: scrive in-place, no allocazione extra
    flat_t = torch.from_numpy(flat_params).float()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.copy_(flat_t[offset:offset + numel].reshape(p.shape))
            offset += numel


def make_loss_fn_fast(
    model,
    u_scaled: torch.Tensor,
    labels_t: torch.Tensor,
    criterion,
):
    """
    Closure NFT-compatibile su feature pre-calcolate.

    Chiama solo model.vqc + model.classifier — salta completamente il backbone.
    È la differenza tra 300 forward ResNet18/epoca e 0.
    """
    def loss_fn(params: np.ndarray) -> float:
        set_trainable_params(model, params)
        model.train()
        with torch.no_grad():
            q_out   = model.vqc(u_scaled)
            outputs = model.classifier(q_out)
            loss    = criterion(outputs, labels_t)
        val = float(loss.item())
        return val if np.isfinite(val) else 1e6
    return loss_fn


def safe_result_fun(result) -> float:
    """Estrae result.fun gestendo il caso None di NFT non convergente."""
    if result.fun is None:
        return float('nan')
    val = float(result.fun)
    return val if np.isfinite(val) else float('nan')


def save_worker_csv(history: list, d: int, seed: int) -> None:
    os.makedirs("experiments/history", exist_ok=True)
    path = f"experiments/history/log_d{d}_s{seed}.csv"
    pd.DataFrame(
        history,
        columns=[
            'epoch', 'd', 'seed', 'backbone',
            'train_loss',
            'val_loss', 'val_acc', 'val_macro_f1',
            'test_loss', 'test_acc', 'test_macro_f1',
            'per_class_f1',
        ],
    ).to_csv(path, index=False)


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
    logger.info(f"Avvio job d={d} seed={seed} backbone={backbone}")

    # — Dati -----------------------------------------------------------------
    try:
        train_loader_full, val_loader, test_loader = get_data_loaders(
            seed=seed, batch_size=32
        )
    except Exception:
        logger.error(traceback.format_exc())
        raise

    try:
        train_labels = extract_labels_from_loader(train_loader_full)
        logger.info(
            f"Campioni train: {len(train_labels)} | "
            f"classi: {np.unique(train_labels).tolist()}"
        )
    except Exception:
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        raise

    # — Modello --------------------------------------------------------------
    try:
        model = HybridModel(
            {'d_latent': d, 'n_qubits': N_QUBITS, 'n_layers': 3, 'seed': seed},
            backbone_type=backbone,
        ).to(device)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri trainabili: {n_trainable}")
    print(f"[d={d} s={seed}] parametri trainabili: {n_trainable}", flush=True)

    # — Pre-calcolo feature (UNA VOLTA per job) ------------------------------
    # Il backbone viene chiamato qui e MAI più nel loop NFT.
    # Risparmio stimato: 300 forward ResNet18/epoca → 1 forward/job.
    try:
        u_train_scaled, train_labels_np = precompute_features(
            model, balanced_loader, device
        )
        u_val_scaled, val_labels_np = precompute_features(
            model, val_loader, device
        )
        u_test_scaled, test_labels_np = precompute_features(
            model, test_loader, device
        )
        train_labels_t = torch.tensor(
            train_labels_np, dtype=torch.long, device=device
        )
        logger.info(
            f"Feature pre-calcolate: "
            f"train {u_train_scaled.shape} | val {u_val_scaled.shape} | test {u_test_scaled.shape}"
        )
        print(
            f"[d={d} s={seed}] feature calcolate: "
            f"train {tuple(u_train_scaled.shape)} | val {tuple(u_val_scaled.shape)} | test {tuple(u_test_scaled.shape)}",
            flush=True,
        )
    except Exception:
        logger.error(f"Errore pre-calcolo feature:\n{traceback.format_exc()}")
        raise

    # — NFT ------------------------------------------------------------------
    optimizer = NFT(maxfev=MAX_EVALS_NFT)

    best_val_macro_f1  = 0.0   # metrica per checkpoint e early stopping
    best_test_macro_f1 = 0.0   # test F1 all'epoca in cui val era migliore (solo report)
    best_loss          = float('inf')
    patience_ctr       = 0     # contatore early stopping
    history       = []
    os.makedirs("experiments/models", exist_ok=True)

    print(
        f">>> NFT: {backbone.upper()} | d={d} seed={seed} | "
        f"maxfev={MAX_EVALS_NFT} | patience={PATIENCE}",
        flush=True,
    )
    logger.info(f"Inizio training NFT maxfev={MAX_EVALS_NFT} patience={PATIENCE}")

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            # loss_fn opera su feature pre-calcolate — backbone escluso
            loss_fn = make_loss_fn_fast(
                model, u_train_scaled, train_labels_t, criterion
            )
            result  = optimizer.minimize(fun=loss_fn, x0=get_trainable_params(model))
            set_trainable_params(model, result.x)

            # valutazione su val e test — macro-F1 prima della loss
            # per_class_f1 loggato per diagnosticare classi problematiche
            val_loss, val_acc, val_macro_f1, _ = evaluate_on_features(
                model, u_val_scaled, val_labels_np, criterion, device
            )
            test_loss, test_acc, macro_f1, per_class_f1 = evaluate_on_features(
                model, u_test_scaled, test_labels_np, criterion, device
            )

        except Exception:
            logger.error(f"Epoch {epoch+1}:\n{traceback.format_exc()}")
            print(
                f"[WARN] d={d} s={seed} | Epoch {epoch+1} fallita — continuo",
                flush=True,
            )
            history.append([
                epoch+1, d, seed, backbone,
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

        # Stampa: macro-F1 test e val prima della loss, poi per-class F1
        log_msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"macro-F1 test: {macro_f1:.4f} | macro-F1 val: {val_macro_f1:.4f} | "
            f"Test Loss: {test_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | "
            f"per-class F1: {[f'{v:.3f}' for v in per_class_f1]} | "
            f"t={elapsed:.1f}s"
        )
        logger.info(log_msg)
        print(f"d={d} s={seed} | {log_msg}", flush=True)

        # checkpoint e early stopping basati SOLO su val — nessuna decisione sul test
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = macro_f1   # test passivo: registrato, non usato per decidere
            best_loss          = train_loss
            patience_ctr       = 0
            ckpt_path = (
                f"experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth"
            )
            try:
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint → {ckpt_path} (val F1={val_macro_f1:.4f} | test F1={macro_f1:.4f})")
            except Exception:
                logger.error(f"Errore checkpoint:\n{traceback.format_exc()}")
        else:
            patience_ctr += 1
            logger.info(f"No improvement val: {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping a epoch {epoch+1}")
                print(
                    f"[STOP] d={d} s={seed} | "
                    f"Early stopping a epoch {epoch+1} (patience={PATIENCE})",
                    flush=True,
                )
                history.append([
                    epoch+1, d, seed, backbone,
                    train_loss,
                    val_loss, val_acc, val_macro_f1,
                    test_loss, test_acc, macro_f1,
                    str([f'{v:.3f}' for v in per_class_f1]),
                ])
                break

        history.append([
            epoch+1, d, seed, backbone,
            train_loss,
            val_loss, val_acc, val_macro_f1,
            test_loss, test_acc, macro_f1,
            str([f'{v:.3f}' for v in per_class_f1]),
        ])

    try:
        save_worker_csv(history, d, seed)
    except Exception:
        logger.error(f"Errore CSV:\n{traceback.format_exc()}")

    logger.info(f"Completato. Best val macro-F1: {best_val_macro_f1:.4f} | Test macro-F1 al miglior val: {best_test_macro_f1:.4f}")
    print(f"[OK] d={d} s={seed} → Val macro-F1: {best_val_macro_f1:.4f} | Test macro-F1: {best_test_macro_f1:.4f}", flush=True)
    return {
        "d":              d,
        "seed":           seed,
        "backbone":       backbone,
        "best_loss":      best_loss,
        "best_val_macro_f1":  best_val_macro_f1,
        "best_test_macro_f1": best_test_macro_f1,
    }


# ---------------------------------------------------------------------------
# Entry point — parallelizzazione Windows-safe
# ---------------------------------------------------------------------------
def main():
    os.makedirs("experiments/models",  exist_ok=True)
    os.makedirs("experiments/logs",    exist_ok=True)
    os.makedirs("experiments/history", exist_ok=True)
    os.makedirs("experiments",         exist_ok=True)

    jobs = [(d, s, BACKBONE) for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(
        f"[INFO] NFT | maxfev={MAX_EVALS_NFT} | "
        f"epochs={EPOCHS} | patience={PATIENCE}"
    )
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
                    f"[DONE] d={d} s={s} → "
                    f"val F1: {result['best_val_macro_f1']:.4f} | "
                    f"test F1 (al miglior val): {result['best_test_macro_f1']:.4f} | "
                    f"loss: {result['best_loss']:.4f}",
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

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['d', 'seed', 'backbone', 'best_loss', 'best_val_macro_f1', 'best_test_macro_f1']]
        df = df.sort_values(
            ["d", "seed"], ascending=[False, True]
        ).reset_index(drop=True)
        df.to_csv("experiments/production_summary.csv", index=False)

        # Merge CSV per-worker in un unico log finale
        history_files = [
            f"experiments/history/log_d{d}_s{s}.csv"
            for d, s, _ in jobs
            if os.path.exists(f"experiments/history/log_d{d}_s{s}.csv")
        ]
        if history_files:
            pd.concat(
                [pd.read_csv(f) for f in history_files], ignore_index=True
            ).to_csv("experiments/production_log.csv", index=False)
            print(f"[INFO] Log unificato → experiments/production_log.csv")

        print("\n[DONE] Training NFT completato.")
        print(df.to_string(index=False))
    else:
        print(
            "\n[WARNING] Nessun risultato. "
            "Controllare experiments/logs/ per i traceback."
        )


if __name__ == "__main__":
    main()