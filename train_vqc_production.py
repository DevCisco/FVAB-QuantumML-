# ---------------------------------------------------------------------------
# CRITICO — Windows spawn: deve essere la PRIMA istruzione eseguibile.
# Senza freeze_support ogni worker processo ri-lancia main() ricorsivamente.
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

from hybrid_engine import HybridModel

# ---------------------------------------------------------------------------
# NFT — Nakanishi-Fujii-Todo optimizer
# ---------------------------------------------------------------------------
# Progettato per VQC con gate parametrici sinusoidali (RY in RealAmplitudes).
# Ottimizza un parametro alla volta trovando il minimo analitico con ~3
# valutazioni, senza stimare gradienti né il Quantum Geometric Tensor.
# Non richiede fidelity né SamplerV2 — zero dipendenze da API Qiskit instabili.

from qiskit_algorithms.optimizers import NFT


# ---------------------------------------------------------------------------
# Costanti globali
# ---------------------------------------------------------------------------
DIMS     = [32, 16, 8, 4]
SEEDS    = [11, 17, 29]
BACKBONE = 'pca'
EPOCHS   = 10
PATIENCE = 3

MAX_EVALS_NFT = 300
MAX_WORKERS   = 4

N_CLASSES              = 4
N_QUBITS               = N_CLASSES
SAMPLES_PER_CLASS      = 8
SAMPLES_POOL_PER_CLASS = 64

# Dimensione batch per la valutazione su val/test set completi.
# Un singolo forward sull'intero set causa OOM con dataset grandi.
EVAL_BATCH_SIZE = 512

JOB_TIMEOUT_SEC = 8 * 3600


# ---------------------------------------------------------------------------
# Budget NFT scalato con il numero di parametri trainabili
#
# Problema: feature_selector ha d×N_QUBITS parametri — con d=32 i parametri
# totali (164) sono più del triplo di d=4 (52), ma MAX_EVALS_NFT era fisso
# a 300 per tutti. Risultato osservato: d=16 (100 param) ottiene esattamente
# 3.0 sweep completi e converge stabilmente su tutti i seed (test F1
# 0.852-0.862). d=32 (164 param) ottiene solo 1.8 sweep e fallisce
# catastroficamente per seed=11 (test F1 0.461) — con meno di 2 sweep,
# l'ordine in cui NFT visita i parametri diventa determinante per il risultato.
#
# Fix: scala il budget SOLO quando serve, con un floor a MAX_EVALS_NFT (300).
# d=4/8/16 restano IDENTICI a prima (nessuna regressione su config già
# validate). Solo d=32 sale a 492 (+64%, il minimo per garantire 3 sweep).
#
# Scelta deliberatamente conservativa per il vincolo hardware
# (i5-1135G7, 4 core fisici): niente scaling uniforme a 5 sweep per tutti
# (che avrebbe quasi triplicato il costo anche per d=8/d=16 che già
# funzionano bene) — solo il minimo indispensabile per d=32.
NFT_TARGET_SWEEPS = 3   # sweep minimi garantiti — validato empiricamente su d=16


def get_max_evals_nft(n_trainable_params: int) -> int:
    """
    Calcola maxfev sufficiente per almeno NFT_TARGET_SWEEPS sweep completi,
    senza mai scendere sotto MAX_EVALS_NFT (300) per non alterare il
    comportamento delle configurazioni d<=16 già validate.

    Args:
        n_trainable_params: somma di feature_selector + VQC + classifier.

    Returns:
        maxfev da passare a NFT().
    """
    return max(MAX_EVALS_NFT, n_trainable_params * NFT_TARGET_SWEEPS)



# ---------------------------------------------------------------------------
# Logging per worker
# ---------------------------------------------------------------------------
def get_logger(d: int, seed: int) -> logging.Logger:
    os.makedirs("experiments/logs", exist_ok=True)
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
# MODIFICA PRINCIPALE — Lettura CSV prodotti da test.py
# ---------------------------------------------------------------------------
# test.py salva: artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv
# Colonne: feat_0, feat_1, ..., feat_{d-1}, label
#
# Questa funzione sostituisce completamente precompute_features_full() e
# tutta la pipeline backbone→scaler→PCA che era eseguita a runtime.
# Le feature d-dim sono già calcolate: basta leggerle e convertirle in tensor.
# ---------------------------------------------------------------------------
def load_features_from_csv(split: str, d: int, seed: int, device) -> tuple:
    """
    Legge le feature PCA d-dim pre-calcolate da test.py.

    Returns:
        u  (Tensor float32, shape (N, d)): feature su device.
        y  (ndarray int64,  shape (N,)  ): label corrispondenti.
    """
    path = f"artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv"
    df   = pd.read_csv(path)

    feat_cols = [f"feat_{i}" for i in range(d)]
    features  = df[feat_cols].values.astype(np.float32)
    labels    = df['label'].values.astype(np.int64)

    u = torch.tensor(features, dtype=torch.float32, device=device)
    return u, labels


# ---------------------------------------------------------------------------
# Pool bilanciato — costruito direttamente dagli array numpy (no DataLoader)
# ---------------------------------------------------------------------------
def make_balanced_pool(
    u_full:            torch.Tensor,  # (N, d) — tutte le feature del train
    y_full:            np.ndarray,    # (N,)
    samples_per_class: int,
    seed:              int,
) -> tuple:
    """
    Seleziona samples_per_class campioni per classe dal training set completo.

    Returns:
        u_pool  (Tensor): shape (N_CLASSES * samples_per_class, d).
        y_pool  (ndarray): shape (N_CLASSES * samples_per_class,).
    """
    rng = np.random.default_rng(seed)
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_full == cls)[0]
        if len(cls_idx) == 0:
            raise ValueError(
                f"Classe {cls} assente nel train. "
                f"Presenti: {np.unique(y_full).tolist()}"
            )
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())

    # Indicizzazione Tensor con LongTensor (non con ndarray numpy: non garantito
    # su tutte le versioni di PyTorch).
    idx_t = torch.tensor(indices, dtype=torch.long)
    return u_full[idx_t], y_full[np.array(indices)]


# ---------------------------------------------------------------------------
# Campionamento fresco dal pool ogni epoca
# ---------------------------------------------------------------------------
def sample_balanced_batch(
    u_pool:            torch.Tensor,
    y_pool:            np.ndarray,
    samples_per_class: int,
    rng,
) -> tuple:
    """
    Campiona samples_per_class indici per classe dal pool pre-calcolato.
    Il rng esterno garantisce batch diversi ad ogni chiamata.
    """
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_pool == cls)[0]
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())

    # Stessa correzione di make_balanced_pool: LongTensor per indicizzare Tensor.
    idx_t = torch.tensor(indices, dtype=torch.long)
    return u_pool[idx_t], y_pool[np.array(indices)]


# ---------------------------------------------------------------------------
# Pesi per classe
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
        raise RuntimeError(
            "Nessun parametro trainable. "
            "Verificare requires_grad e che il backbone sia congelato."
        )
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
# Closure NFT
# ---------------------------------------------------------------------------
def make_loss_fn(
    modules:  list,
    u_batch:  torch.Tensor,  # (32, d) — batch fresco di questo epoch
    labels_t: torch.Tensor,  # (32,)
    criterion,
) -> callable:
    # Closure NFT-compatibile: callable(params: np.ndarray) → float.
    def loss_fn(params: np.ndarray) -> float:
        set_trainable_params(modules, params)
        for module in modules:
            module.train()
        with torch.no_grad():
            u_4      = modules[0](u_batch)          # feature_selector: d → 4
            u_scaled = modules[1].scale_features(u_4)
            q_out    = modules[1].vqc(u_scaled)
            outputs  = modules[1].classifier(q_out)
            loss     = criterion(outputs, labels_t)
        val = float(loss.item())
        return val if np.isfinite(val) else 1e6

    return loss_fn


def safe_result_fun(result) -> float:
    if result.fun is None:
        return float('nan')
    val = float(result.fun)
    return val if np.isfinite(val) else float('nan')


# ---------------------------------------------------------------------------
# Evaluation su feature d-dim pre-calcolate — con mini-batch loop
# ---------------------------------------------------------------------------
def evaluate_on_features(
    modules:   list,
    u_d:       torch.Tensor,  # (N, d)
    labels_np: np.ndarray,
    criterion,
    device,
    batch_size: int = EVAL_BATCH_SIZE,
) -> tuple:
    
    # Valuta su feature d-dim pre-calcolate processando in mini-batch.
    feature_selector, model = modules[0], modules[1]
    feature_selector.eval()
    model.eval()

    # Criterion separato con reduction='sum' per aggregazione corretta
    # tra batch di dimensioni diverse (ultimo batch potrebbe essere più piccolo).
    eval_criterion = nn.CrossEntropyLoss(
        weight=criterion.weight,
        reduction='sum',
    )

    all_preds  = []
    total_loss = 0.0
    n_samples  = len(labels_np)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end      = min(start + batch_size, n_samples)
            u_batch  = u_d[start:end]
            y_batch  = torch.tensor(labels_np[start:end], dtype=torch.long, device=device)

            u_4      = feature_selector(u_batch)
            u_scaled = model.scale_features(u_4)
            q_out    = model.vqc(u_scaled)
            outputs  = model.classifier(q_out)
            loss     = eval_criterion(outputs, y_batch)

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
def save_worker_csv(history: list, d: int, seed: int) -> None:
    os.makedirs("experiments/history", exist_ok=True)
    path = f"experiments/history/log_d{d}_s{seed}.csv"
    pd.DataFrame(
        history,
        columns=[
            'epoch', 'd', 'seed', 'backbone',
            'train_loss',
            'val_loss',  'val_acc',  'val_macro_f1',
            'test_loss', 'test_acc', 'test_macro_f1',
            'per_class_f1',
        ],
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Worker — un processo per coppia (d, seed)
# ---------------------------------------------------------------------------
def train_production(d: int, seed: int, backbone: str = BACKBONE) -> dict:
    """
    Addestra HybridModel + feature_selector per una coppia (d, seed).
    Le feature d-dim vengono lette dai CSV prodotti da test.py.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed)
    logger.info(f"Avvio job d={d} seed={seed} backbone={backbone}")

    # — Caricamento feature da CSV (prodotti da test.py) ---------------------
    try:
        u_train, y_train = load_features_from_csv('train', d, seed, device)
        u_val,   y_val   = load_features_from_csv('val',   d, seed, device)
        u_test,  y_test  = load_features_from_csv('test',  d, seed, device)
        logger.info(
            f"Feature caricate da CSV: "
            f"train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | "
            f"test {tuple(u_test.shape)}"
        )
        print(
            f"[d={d} s={seed}] CSV letti: "
            f"train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | "
            f"test {tuple(u_test.shape)}",
            flush=True,
        )
    except FileNotFoundError as e:
        logger.error(f"CSV non trovato: {e}\nEseguire prima test.py!")
        raise
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Pool bilanciato dal training set completo ----------------------------
    try:
        u_pool, y_pool = make_balanced_pool(
            u_train, y_train,
            samples_per_class=SAMPLES_POOL_PER_CLASS,
            seed=seed,
        )
        logger.info(f"Pool bilanciato: {tuple(u_pool.shape)}")
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Pesi per classe e criterion ------------------------------------------
    try:
        class_weights = compute_class_weights(y_train).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Modello VQC ----------------------------------------------------------
    # backbone_type=None: il backbone non viene mai invocato nel nuovo flusso.
    # I CSV contengono già le feature d-dim pre-calcolate; HybridModel serve
    # solo per model.vqc, model.classifier e model.scale_features.
    try:
        model = HybridModel(
            {'d_latent': d, 'n_qubits': N_QUBITS, 'n_layers': 3, 'seed': seed},
            backbone_type=None,
        ).to(device)
    except Exception:
        logger.error(traceback.format_exc())
        raise

    # — Feature selector: proiezione trainabile d → N_QUBITS -----------------
    feature_selector = nn.Linear(d, N_QUBITS, bias=False).to(device)
    nn.init.orthogonal_(feature_selector.weight)

    modules = [feature_selector, model]

    n_fs    = sum(p.numel() for p in feature_selector.parameters() if p.requires_grad)
    n_vqc   = sum(p.numel() for p in model.parameters()            if p.requires_grad)
    n_total = n_fs + n_vqc
    logger.info(
        f"Parametri: feature_selector={n_fs} (d={d}×{N_QUBITS}) | "
        f"VQC+classifier={n_vqc} | totale={n_total}"
    )
    print(
        f"[d={d} s={seed}] parametri: fs={n_fs} | vqc+clf={n_vqc} | tot={n_total}",
        flush=True,
    )

    # — NFT ------------------------------------------------------------------
    # Budget scalato sul numero di parametri trainabili (n_total) — vedi
    # get_max_evals_nft per la motivazione. Per d<=16 coincide con il
    # precedente valore fisso (300); solo d=32 ottiene un budget maggiore.
    max_evals = get_max_evals_nft(n_total)
    optimizer = NFT(maxfev=max_evals)
    epoch_rng = np.random.default_rng(seed)

    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0
    best_loss          = float('inf')
    patience_ctr       = 0
    history            = []
    os.makedirs("experiments/models", exist_ok=True)

    print(
        f">>> NFT: {backbone.upper()} | d={d} seed={seed} | "
        f"maxfev={max_evals} ({max_evals/n_total:.1f} sweep) | patience={PATIENCE}",
        flush=True,
    )
    logger.info(
        f"Inizio training NFT maxfev={max_evals} "
        f"({max_evals/n_total:.1f} sweep su {n_total} param) patience={PATIENCE}"
    )

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            u_batch, y_batch_np = sample_balanced_batch(
                u_pool, y_pool, SAMPLES_PER_CLASS, epoch_rng
            )
            y_batch_t = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            loss_fn = make_loss_fn(modules, u_batch, y_batch_t, criterion)
            result  = optimizer.minimize(
                fun=loss_fn, x0=get_trainable_params(modules)
            )
            set_trainable_params(modules, result.x)

            val_loss,  val_acc,  val_macro_f1,  _            = evaluate_on_features(
                modules, u_val,  y_val,  criterion, device
            )
            test_loss, test_acc, test_macro_f1, per_class_f1 = evaluate_on_features(
                modules, u_test, y_test, criterion, device
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

        log_msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"macro-F1 val: {val_macro_f1:.4f} | macro-F1 test: {test_macro_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"per-class F1: {[f'{v:.3f}' for v in per_class_f1]} | "
            f"t={elapsed:.1f}s"
        )
        logger.info(log_msg)
        print(f"d={d} s={seed} | {log_msg}", flush=True)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = test_macro_f1
            best_loss          = train_loss
            patience_ctr       = 0
            ckpt_path = f"experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth"
            try:
                torch.save(
                    {
                        'model':            model.state_dict(),
                        'feature_selector': feature_selector.state_dict(),
                        'd':                d,
                        'seed':             seed,
                        'epoch':            epoch + 1,
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
                    f"[STOP] d={d} s={seed} | "
                    f"Early stopping a epoch {epoch+1} (patience={PATIENCE})",
                    flush=True,
                )
                history.append([
                    epoch+1, d, seed, backbone,
                    train_loss,
                    val_loss,  val_acc,  val_macro_f1,
                    test_loss, test_acc, test_macro_f1,
                    str([f'{v:.3f}' for v in per_class_f1]),
                ])
                break

        history.append([
            epoch+1, d, seed, backbone,
            train_loss,
            val_loss,  val_acc,  val_macro_f1,
            test_loss, test_acc, test_macro_f1,
            str([f'{v:.3f}' for v in per_class_f1]),
        ])

    try:
        save_worker_csv(history, d, seed)
    except Exception:
        logger.error(f"Errore CSV:\n{traceback.format_exc()}")

    # best_loss resta float('inf') se val_macro_f1 non migliora mai
    # (es. tutte le epoche falliscono). float('inf') nel CSV finale causa
    # problemi di serializzazione; si normalizza a nan.
    if not np.isfinite(best_loss):
        best_loss = float('nan')

    logger.info(
        f"Completato. Best val macro-F1: {best_val_macro_f1:.4f} | "
        f"Test macro-F1 al miglior val: {best_test_macro_f1:.4f}"
    )
    print(
        f"[OK] d={d} s={seed} → "
        f"val F1: {best_val_macro_f1:.4f} | test F1: {best_test_macro_f1:.4f}",
        flush=True,
    )
    return {
        "d":                  d,
        "seed":               seed,
        "backbone":           backbone,
        "best_loss":          best_loss,
        "best_val_macro_f1":  best_val_macro_f1,
        "best_test_macro_f1": best_test_macro_f1,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    multiprocessing.freeze_support()

    os.makedirs("experiments/models",  exist_ok=True)
    os.makedirs("experiments/logs",    exist_ok=True)
    os.makedirs("experiments/history", exist_ok=True)

    jobs = [(d, s, BACKBONE) for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(
        f"[INFO] NFT | maxfev base={MAX_EVALS_NFT}, scalato a {NFT_TARGET_SWEEPS} "
        f"sweep min. per d grandi (vedi get_max_evals_nft) | "
        f"epochs={EPOCHS} | patience={PATIENCE}"
    )
    print(
        f"[INFO] feature_selector d→{N_QUBITS} | "
        f"pool={SAMPLES_POOL_PER_CLASS}/classe | batch={SAMPLES_PER_CLASS}/classe"
    )
    print(f"[INFO] Feature lette da: artifacts/sweep/B1_pca_{{split}}_d{{d}}_seed{{seed}}.csv\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Variabile di loop rinominata `bb` per evitare lo shadowing
        # della costante globale BACKBONE.
        futures = {
            executor.submit(train_production, d, s, bb): (d, s, bb)
            for d, s, bb in jobs
        }
        for future in as_completed(futures):
            d, s, bb = futures[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SEC)
                all_results.append(result)
                best_loss_str = (
                    f"{result['best_loss']:.4f}"
                    if np.isfinite(result['best_loss'])
                    else "nan"
                )
                print(
                    f"[DONE] d={d} s={s} → "
                    f"val F1: {result['best_val_macro_f1']:.4f} | "
                    f"test F1: {result['best_test_macro_f1']:.4f} | "
                    f"loss: {best_loss_str}",
                    flush=True,
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] d={d} s={s} → limite {JOB_TIMEOUT_SEC // 3600}h",
                    flush=True,
                )
            except Exception:
                print(
                    f"[ERROR] d={d} s={s} →\n{traceback.format_exc()}",
                    flush=True,
                )

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['d', 'seed', 'backbone', 'best_loss',
                 'best_val_macro_f1', 'best_test_macro_f1']]
        df = df.sort_values(
            ["d", "seed"], ascending=[False, True]
        ).reset_index(drop=True)
        df.to_csv("experiments/production_summary.csv", index=False)

        history_files = [
            f"experiments/history/log_d{d}_s{s}.csv"
            for d, s, _ in jobs
            if os.path.exists(f"experiments/history/log_d{d}_s{s}.csv")
        ]
        if history_files:
            pd.concat(
                [pd.read_csv(f) for f in history_files], ignore_index=True
            ).to_csv("experiments/production_log.csv", index=False)
            print("[INFO] Log unificato → experiments/production_log.csv")

        print("\n[DONE] Training NFT completato.")
        print(df.to_string(index=False))
    else:
        print(
            "\n[WARNING] Nessun risultato. "
            "Controllare experiments/logs/ per i traceback."
        )


if __name__ == "__main__":
    main()