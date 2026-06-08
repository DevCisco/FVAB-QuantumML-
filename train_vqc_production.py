# ---------------------------------------------------------------------------
# CRITICO — Windows spawn: deve essere la PRIMA istruzione eseguibile.
# Senza freeze_support ogni worker processo ri-lancia main() ricorsivamente.
# BUG MANCANTE IN ciccio.py: era presente in train_vqc_production.py
# originale ma omesso in ciccio.py. Reinserito qui.
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
BACKBONE = 'resnet'
EPOCHS   = 10
PATIENCE = 3

# Valutazioni NFT per epoca. Con feature_selector (d×4 param) + VQC (dipende
# da d e n_layers) + classifier (20 param), il totale parametri cresce con d.
# 300 valutazioni coprono ~2-3 sweep completi anche per d=32.
MAX_EVALS_NFT = 300
MAX_WORKERS   = 4

N_CLASSES              = 4
N_QUBITS               = N_CLASSES   # i readout VQC corrispondono ai logit
SAMPLES_PER_CLASS      = 8           # campioni per batch per epoca (32 totali)
SAMPLES_POOL_PER_CLASS = 64          # pool: 64×4=256 feature pre-calcolate;
                                     # ogni epoca campiona 8/classe da queste 256

JOB_TIMEOUT_SEC = 8 * 3600


# ---------------------------------------------------------------------------
# Logging per worker — file separato per ogni (d, seed)
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
# Estrazione label
# ---------------------------------------------------------------------------
def extract_labels_from_loader(loader: DataLoader) -> np.ndarray:
    all_labels = []
    with torch.no_grad():
        for _, labels in loader:
            all_labels.append(labels.cpu().numpy().ravel())
    return np.concatenate(all_labels).astype(int)


# ---------------------------------------------------------------------------
# Subset e loader bilanciati — per il pool di diversità
# ---------------------------------------------------------------------------
def make_balanced_subset(
    dataset, all_labels: np.ndarray, samples_per_class: int, seed: int
) -> Subset:
    rng = np.random.default_rng(seed)
    selected = []
    for cls in range(N_CLASSES):
        idx = np.where(all_labels == cls)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Classe {cls} assente. Presenti: {np.unique(all_labels).tolist()}"
            )
        replace = len(idx) < samples_per_class
        chosen  = rng.choice(idx, size=samples_per_class, replace=replace)
        selected.extend(chosen.tolist())
    return Subset(dataset, selected)


def make_balanced_loader(
    dataset, all_labels: np.ndarray, samples_per_class: int, seed: int
) -> DataLoader:
    subset = make_balanced_subset(dataset, all_labels, samples_per_class, seed)
    # shuffle=False: la randomizzazione avviene in sample_balanced_batch,
    # non nel DataLoader — il pool è fisso, il campionamento è dinamico
    return DataLoader(
        subset,
        batch_size=N_CLASSES * samples_per_class,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Pesi per classe — formula sklearn balanced
# ---------------------------------------------------------------------------
def compute_class_weights(all_labels: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
    weights = len(all_labels) / (N_CLASSES * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# FIX PROBLEMA 1 + 2 — Pre-calcolo feature d-dim COMPLETE (senza troncamento)
# ---------------------------------------------------------------------------
# PROBLEMA 1 risolto: model._compress() troncava a n_encoding = min(d, 4) = 4.
# Per d > 4 il VQC riceveva sempre le stesse 4 feature → d irrilevante.
# Questa funzione restituisce TUTTE le d componenti PCA.
# La proiezione d → 4 avviene tramite feature_selector (trainabile).
#
# PROBLEMA 2 risolto parzialmente: questa funzione viene chiamata sul pool
# (256 campioni = 64/classe), non sul batch fisso di 32. Il campionamento
# fresco ogni epoca è gestito da sample_balanced_batch (vedi sotto).
#
# Pipeline: backbone(imgs) → 512-dim → scaler → PCA → d-dim (NO truncation)
# ---------------------------------------------------------------------------
def precompute_features_full(model, loader: DataLoader, device) -> tuple:
    """
    Estrae feature d-dim complete per tutti i campioni del loader.
    Il backbone viene chiamato UNA SOLA VOLTA (ottimizzazione velocità):
    risparmio stimato 80-90% rispetto a chiamarlo dentro NFT ogni valutazione.

    Returns:
        u_full    (Tensor float32, shape (N, d)): feature complete, su device.
        labels_np (ndarray int,    shape (N,)  ): label corrispondenti.
    """
    all_u, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats_512    = model.backbone.extract_backbone(imgs.to(device)).cpu().numpy()
            feats_scaled = model.scaler.transform(feats_512)
            # .pca è l'oggetto sklearn PCA dentro il wrapper PCACompressor
            feats_d      = model.pca.pca.transform(feats_scaled)  # (B, d) — NO [:,:4]
            all_u.append(torch.tensor(feats_d, dtype=torch.float32))
            all_labels.append(lbls.squeeze().long().numpy())
    return (
        torch.cat(all_u, dim=0).to(device),
        np.concatenate(all_labels),
    )


# ---------------------------------------------------------------------------
# FIX PROBLEMA 2 — Campionamento fresco dal pool ogni epoca
# ---------------------------------------------------------------------------
# PROBLEMA 2 risolto: il vecchio codice usava 32 campioni fissi per tutte le
# epoche. NFT li memorizzava perfettamente → train_loss → 0 senza migliorare
# la generalizzazione. Ora ogni epoca riceve un batch diverso estratto da un
# pool di 256 campioni, senza rieseguire il backbone.
# ---------------------------------------------------------------------------
def sample_balanced_batch(
    u_pool:            torch.Tensor,  # (N_pool, d)
    y_pool:            np.ndarray,    # (N_pool,)
    samples_per_class: int,
    rng,                              # np.random.Generator — avanza tra le epoche
) -> tuple:
    """
    Campiona samples_per_class indici per classe dal pool pre-calcolato.
    Il rng esterno garantisce batch diversi ad ogni chiamata.

    Returns:
        u_batch  (Tensor): shape (N_CLASSES * samples_per_class, d).
        y_batch  (ndarray): shape (N_CLASSES * samples_per_class,).
    """
    indices = []
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_pool == cls)[0]
        replace = len(cls_idx) < samples_per_class
        chosen  = rng.choice(cls_idx, size=samples_per_class, replace=replace)
        indices.extend(chosen.tolist())
    idx = np.array(indices)
    return u_pool[idx], y_pool[idx]


# ---------------------------------------------------------------------------
# FIX PROBLEMA 3 — Bridge NFT ↔ PyTorch su lista di moduli
# ---------------------------------------------------------------------------
# PROBLEMA 3 risolto: il vecchio get/set_trainable_params accettava un singolo
# modello e ignorava feature_selector. NFT ottimizzava solo i 36 param
# VQC+classifier, mai i d×4 param della proiezione.
# Ora accettano una lista di moduli nell'ordine [feature_selector, model].
# L'ordine DEVE essere identico in get e set: è il contratto con NFT.
# ---------------------------------------------------------------------------
def get_trainable_params(modules: list) -> np.ndarray:
    """
    Concatena i parametri trainabili di tutti i moduli nella lista.
    Ordine: [feature_selector params, model params] = [d×4, VQC+clf].
    """
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
    """
    Scrive i parametri flat nei moduli nell'ordine di get_trainable_params.
    torch.from_numpy + copy_ : zero-copy numpy → in-place write, no allocazioni.
    """
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
# FIX "ALTRI PROBLEMI LATENTI" — Closure NFT con feature_selector
# ---------------------------------------------------------------------------
# PROBLEMA risolto: make_loss_fn_fast chiamava model.vqc su feature già
# troncate a 4-dim, bypassando feature_selector. Ora il flusso completo è:
#   u_batch (d-dim) → feature_selector → 4-dim → scale[0,π] → VQC → clf
# ---------------------------------------------------------------------------
def make_loss_fn(
    modules:          list,
    feature_selector: nn.Module,
    model,
    u_batch:          torch.Tensor,   # (32, d) — batch fresco di questo epoch
    labels_t:         torch.Tensor,   # (32,)
    criterion,
) -> callable:
    """
    Closure NFT-compatibile: callable(params: np.ndarray) → float.
    Nessun backprop — NFT stima il minimo analitico per ogni parametro.
    u_batch e labels_t sono catturati per riferimento: ogni epoch crea
    una nuova closure sul batch fresco dell'epoch corrente.
    """
    def loss_fn(params: np.ndarray) -> float:
        set_trainable_params(modules, params)
        feature_selector.train()
        model.train()
        with torch.no_grad():
            u_4     = feature_selector(u_batch)    # d → 4
            u_scaled = model.scale_features(u_4)   # → [0, π]
            q_out   = model.vqc(u_scaled)          # → (32, 4) expectation values
            outputs = model.classifier(q_out)       # → (32, 4) logits
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


# ---------------------------------------------------------------------------
# FIX "ALTRI PROBLEMI LATENTI" — Evaluation con feature_selector
# ---------------------------------------------------------------------------
def evaluate_on_features(
    feature_selector: nn.Module,
    model,
    u_d:       torch.Tensor,    # (N, d) — feature d-dim pre-calcolate
    labels_np: np.ndarray,
    criterion,
    device,
) -> tuple:
    """
    Valuta su feature d-dim pre-calcolate.
    Pipeline: feature_selector → scale_features → VQC → classifier.
    macro-F1 prima di loss e acc: metrica principale su classi sbilanciate.

    Returns: loss, acc, macro_f1, per_class_f1
    """
    feature_selector.eval()
    model.eval()
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
    with torch.no_grad():
        u_4     = feature_selector(u_d)
        u_scaled = model.scale_features(u_4)
        q_out   = model.vqc(u_scaled)
        outputs = model.classifier(q_out)
        loss    = criterion(outputs, labels_t)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()

    acc          = float((preds == labels_np).mean() * 100)
    macro_f1     = float(sk_f1(labels_np, preds, average='macro',    zero_division=0))
    per_class_f1 = sk_f1(labels_np, preds, average=None, zero_division=0).tolist()
    return loss.item(), acc, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# CSV per-worker — nessuna race condition (file separati per processo)
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

    Architettura trainabile (Fix Problema 1):
        feature_selector : Linear(d, 4, bias=False)  → d*4 param
        model.vqc.qweights : dipende da n_layers e re_upload_cycles (varia con d)
        model.classifier   : Linear(4, 4) + bias → 20 param
        Totale: d*4 + n_vqc_weights + 20   (d ORA CONTA — ogni d è diverso)

    Pipeline forward (Fix Problemi 1, 2, altri):
        imgs → backbone[frozen] → 512-dim → scaler → PCA → d-dim
             → feature_selector → 4-dim → scale[0,π] → VQC → classifier

    Fix Problema 4: checkpoint salva ENTRAMBI i dizionari (model + feature_selector).
    """
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
        class_weights = compute_class_weights(train_labels).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        # Pool di diversità: 64/classe = 256 feature — Fix Problema 2
        pool_loader = make_balanced_loader(
            train_loader_full.dataset,
            all_labels=train_labels,
            samples_per_class=SAMPLES_POOL_PER_CLASS,
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

    # — Feature selector: proiezione trainabile d → N_QUBITS (Fix Problema 1) --
    # Senza feature_selector: il circuito riceveva sempre min(d,4)=4 feature →
    # d=32 e d=4 producevano input identici al VQC.
    # Con feature_selector: NFT apprende quale combinazione delle d componenti
    # PCA è più discriminativa per il circuito quantistico.
    #
    # bias=False: le feature PCA hanno media≈0 (StandardScaler+PCA),
    #             un bias sarebbe ridondante.
    # init orthogonal: preserva le distanze inizialmente (rotazione nel
    #                  sottospazio 4-dim), lasciando a NFT di trovare la
    #                  direzione ottimale.
    feature_selector = nn.Linear(d, N_QUBITS, bias=False).to(device)
    nn.init.orthogonal_(feature_selector.weight)

    # Lista ordinata [feature_selector, model] — DEVE essere identica
    # in get_trainable_params e set_trainable_params (contratto con NFT).
    modules = [feature_selector, model]

    n_fs    = sum(p.numel() for p in feature_selector.parameters() if p.requires_grad)
    n_vqc   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = n_fs + n_vqc
    logger.info(
        f"Parametri: feature_selector={n_fs} (d={d}×4) | "
        f"VQC+classifier={n_vqc} | totale={n_total}"
    )
    print(
        f"[d={d} s={seed}] parametri: fs={n_fs} | vqc+clf={n_vqc} | tot={n_total}",
        flush=True,
    )

    # — Pre-calcolo feature d-dim complete (UNA VOLTA per job) ---------------
    # Fix Problema 1 + 2: precompute_features_full NON tronca a 4,
    # restituisce tutte le d componenti PCA.
    try:
        u_pool,  y_pool = precompute_features_full(model, pool_loader,   device)
        u_val,   y_val  = precompute_features_full(model, val_loader,    device)
        u_test,  y_test = precompute_features_full(model, test_loader,   device)
        logger.info(
            f"Feature pre-calcolate: pool {u_pool.shape} | "
            f"val {u_val.shape} | test {u_test.shape}"
        )
        print(
            f"[d={d} s={seed}] feature: "
            f"pool {tuple(u_pool.shape)} | val {tuple(u_val.shape)} | test {tuple(u_test.shape)}",
            flush=True,
        )
    except Exception:
        logger.error(f"Errore pre-calcolo feature:\n{traceback.format_exc()}")
        raise

    # — NFT ------------------------------------------------------------------
    optimizer = NFT(maxfev=MAX_EVALS_NFT)

    # RNG per campionamento fresco — avanza con ogni chiamata (Fix Problema 2)
    epoch_rng = np.random.default_rng(seed)

    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0   # snapshot passivo all'epoca del miglior val
    best_loss          = float('inf')
    patience_ctr       = 0
    history            = []
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
            # Batch fresco ogni epoca dal pool (Fix Problema 2)
            u_batch, y_batch_np = sample_balanced_batch(
                u_pool, y_pool, SAMPLES_PER_CLASS, epoch_rng
            )
            y_batch_t = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            # Closure su tutti i moduli (Fix Problema 3 + "altri problemi")
            loss_fn = make_loss_fn(
                modules, feature_selector, model, u_batch, y_batch_t, criterion
            )
            result = optimizer.minimize(
                fun=loss_fn, x0=get_trainable_params(modules)
            )
            set_trainable_params(modules, result.x)

            # Evaluation con feature_selector (Fix "altri problemi")
            val_loss,  val_acc,  val_macro_f1,  _            = evaluate_on_features(
                feature_selector, model, u_val,  y_val,  criterion, device
            )
            test_loss, test_acc, test_macro_f1, per_class_f1 = evaluate_on_features(
                feature_selector, model, u_test, y_test, criterion, device
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

        # macro-F1 prima di loss e acc (direttiva metrica)
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

        # Checkpoint e early stopping su val — nessuna decisione sul test
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1  = val_macro_f1
            best_test_macro_f1 = test_macro_f1
            best_loss          = train_loss
            patience_ctr       = 0
            ckpt_path = (
                f"experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth"
            )
            try:
                # FIX PROBLEMA 4: salva ENTRAMBI i moduli.
                # Il vecchio torch.save(model.state_dict()) escludeva
                # feature_selector → checkpoint inutilizzabile per inference.
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
    os.makedirs("experiments/models",  exist_ok=True)
    os.makedirs("experiments/logs",    exist_ok=True)
    os.makedirs("experiments/history", exist_ok=True)

    jobs = [(d, s, BACKBONE) for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(
        f"[INFO] NFT | maxfev={MAX_EVALS_NFT} | "
        f"epochs={EPOCHS} | patience={PATIENCE}"
    )
    print(
        f"[INFO] feature_selector d→{N_QUBITS} | "
        f"pool={SAMPLES_POOL_PER_CLASS}/classe | batch={SAMPLES_PER_CLASS}/classe"
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
                    f"test F1: {result['best_test_macro_f1']:.4f} | "
                    f"loss: {result['best_loss']:.4f}",
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