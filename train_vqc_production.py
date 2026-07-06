# ---------------------------------------------------------------------------
# Windows spawn: multiprocessing.freeze_support() viene chiamata come prima
# istruzione DENTRO main(), subito dopo il guard `if __name__ == "__main__"`.
# Questo soddisfa il requisito reale di Python (deve precedere qualunque
# spawn di processo figlio) — non serve essere a livello di modulo.
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
# FIX PRINCIPALE — rimossa la dipendenza da HybridModel.
#
# HybridModel.__init__ istanziava SEMPRE un ResNetCompressor completo
# (~11.2M parametri, caricamento pesi da disco) e rifittava StandardScaler+PCA
# su 10.000 campioni raw 512-dim, INDIPENDENTEMENTE dal parametro backbone_type
# passato (che nella classe non viene mai letto nel corpo del costruttore).
#
# Nel flusso attuale le feature d-dim arrivano già pronte dai CSV prodotti
# da test.py — backbone, scaler e PCA di HybridModel non servono a nulla,
# ma venivano comunque costruiti 12 volte (una per job), sprecando CPU/RAM
# e gonfiando ogni checkpoint di decine di MB di pesi ResNet mai usati.
#
# Fix: si costruiscono direttamente solo i tre componenti realmente
# necessari — QuantumPipeline (circuito), DirectVQC (layer quantistico),
# nn.Linear (classifier) — bypassando completamente ResNetCompressor.
# ---------------------------------------------------------------------------
from quantum_model import QuantumPipeline
from hybrid_engine import DirectVQC
from qiskit_aer.primitives import EstimatorV2

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
DIMS        = [32, 16, 8, 4]
SEEDS       = [11, 17, 29]
COMPRESSORS = ['B1', 'B2', 'B3']   # B1=PCA (test.py), B2=VanillaAE, B3=RegularizedAE
EPOCHS      = 5
PATIENCE    = 3

# Percorsi CSV per ogni compressore — devono corrispondere esattamente a
# quelli prodotti da test.py (B1) e b2_b3_training.py (B2, B3).
COMPRESSOR_PATHS = {
    'B1': "artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv",
    'B2': "artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv",
    'B3': "artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv",
}

MAX_EVALS_NFT = 300
MAX_WORKERS   = 4

N_CLASSES              = 4
N_QUBITS                = N_CLASSES
N_LAYERS                = 3   # RealAmplitudes reps → 4×(3+1)=16 parametri variazionali
SAMPLES_PER_CLASS       = 8
SAMPLES_POOL_PER_CLASS  = 64

# Dimensione batch per la valutazione su val/test set completi.
# Un singolo forward sull'intero set causa OOM con dataset grandi.
EVAL_BATCH_SIZE = 512

JOB_TIMEOUT_SEC = 8 * 3600

# ---------------------------------------------------------------------------
# Pre-training di feature_selector con Adam
#
# Problema risolto: NFT ottimizza un parametro alla volta (parameter shift
# analitico) e scala male in alta dimensione. feature_selector ha d×N_QUBITS
# parametri: d=32 → 128 param, d=4 → 16 param. Con maxfev=300 fisso, d=32
# riceveva ~1.8 sweep completi — insufficienti a gestire le interazioni tra
# 128 parametri interdipendenti → d=32 underperformava rispetto a d=4 (soli
# 16 param, ~5.8 sweep), invertendo l'ordine atteso.
#
# Fix — separazione degli ottimizzatori per natura del problema:
#   • Adam     → feature_selector (d*4 param, backprop, scala O(n) con d)
#   • NFT      → VQC + classifier (36 param fissi, indipendente da d)
#
# Dopo il pre-training, feature_selector viene congelato.
# Le feature proiettate (pool/val/test) vengono pre-calcolate UNA SOLA VOLTA
# → NFT nel hot loop chiama esclusivamente vqc + classifier (niente
# feature_selector né scale_features ad ogni valutazione).
# Risultato: budget NFT torna flat a 300 per tutti i d (5.77 sweep su 36 param).
PRETRAIN_STEPS = 200    # passi Adam — converge in poche decine su 256 campioni
PRETRAIN_LR    = 0.01   # LR Adam standard


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
# Lettura CSV prodotti da test.py (B1) e b2_b3_training.py (B2, B3)
# ---------------------------------------------------------------------------
# Percorsi in COMPRESSOR_PATHS — stessa struttura per tutti e tre.
# Il rilevamento delle colonne feature usa [c != 'label'] invece di
# [f"feat_{i}..."] perché B2/B3 usano 'latent_i' mentre B1 usa 'feat_i':
# entrambi vengono letti correttamente senza dipendere dal naming.
# ---------------------------------------------------------------------------
def load_features_from_csv(split: str, d: int, seed: int,
                            compressor: str, device) -> tuple:
    """
    Legge le feature d-dim pre-calcolate per il compressore specificato.

    Args:
        split:      'train' | 'val' | 'test'
        d:          dimensione latente (4/8/16/32)
        seed:       seed (11/17/29)
        compressor: 'B1' | 'B2' | 'B3'
        device:     torch.device

    Returns:
        u  (Tensor float32, shape (N, d)): feature su device.
        y  (ndarray int64,  shape (N,)  ): label corrispondenti.
    """
    path = COMPRESSOR_PATHS[compressor].format(split=split, d=d, seed=seed)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV non trovato: {path}\n"
            f"Per B1: eseguire test.py\n"
            f"Per B2/B3: eseguire b2_b3_training.py"
        )

    df        = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']   # funziona con feat_i e latent_i
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
# Scaling feature in [0, π] per i gate RY (angle encoding)
#
# Replica esatta dell'ex HybridModel.scale_features, ora standalone perché
# non dipendiamo più da HybridModel. Pura funzione matematica, nessuno stato.
# ---------------------------------------------------------------------------
def scale_features(u: torch.Tensor) -> torch.Tensor:
    u_min = u.min(dim=1, keepdim=True)[0]
    u_max = u.max(dim=1, keepdim=True)[0]
    return (u - u_min) / (u_max - u_min + 1e-8) * np.pi


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
            "Verificare requires_grad sui moduli (feature_selector, vqc, classifier)."
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
# Pre-training di feature_selector con Adam + testa classica temporanea
# ---------------------------------------------------------------------------
def pretrain_feature_selector(
    feature_selector: nn.Module,
    u_pool:           torch.Tensor,   # (N_pool, d) — pool completo pre-calcolato
    y_pool:           np.ndarray,     # (N_pool,)
    criterion,
    device,
) -> None:
    """
    Ottimizza feature_selector con Adam usando una testa lineare temporanea.

    La testa classica (temp_head) approssima il readout VQC per propagare
    un gradiente coerente attraverso feature_selector. Viene scartata
    al termine — serve solo a costruire il segnale di aggiornamento.

    Usa il pool completo (256 campioni, 64/classe) a ogni step per massimizzare
    la qualità del gradiente. Con Adam e 256 campioni, 200 passi convergono
    in pochi secondi su CPU.

    Args:
        feature_selector: nn.Linear(d, N_QUBITS, bias=False) — verrà aggiornato in-place.
        u_pool:           feature d-dim del pool bilanciato.
        y_pool:           label corrispondenti.
        criterion:        CrossEntropyLoss con class weights dal training set.
        device:           torch.device.
    """
    temp_head = nn.Linear(N_QUBITS, N_CLASSES, bias=True).to(device)
    adam_opt  = torch.optim.Adam(
        list(feature_selector.parameters()) + list(temp_head.parameters()),
        lr=PRETRAIN_LR,
        weight_decay=1e-4,
    )
    y_pool_t = torch.tensor(y_pool, dtype=torch.long, device=device)

    feature_selector.train()
    temp_head.train()

    for _ in range(PRETRAIN_STEPS):
        adam_opt.zero_grad()
        u_4     = feature_selector(u_pool)   # (N_pool, d) → (N_pool, 4)
        u_s     = scale_features(u_4)
        outputs = temp_head(u_s)
        loss    = criterion(outputs, y_pool_t)
        loss.backward()
        adam_opt.step()

    # temp_head scartata — non viene restituita né salvata


# ---------------------------------------------------------------------------
# Closure NFT — opera su feature già proiettate e scalate (4-dim)
#
# modules = [vqc, classifier] — solo 36 parametri, identici per tutti i d.
# feature_selector è già stato ottimizzato con Adam e congelato.
# u_batch_scaled è già 4-dim e già in [0,π] — nessun forward di
# feature_selector né scale_features nel hot loop NFT.
# ---------------------------------------------------------------------------
def make_loss_fn(
    modules:        list,
    u_batch_scaled: torch.Tensor,   # (32, N_QUBITS) — già proiettato e scalato
    labels_t:       torch.Tensor,   # (32,)
    criterion,
) -> callable:
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


# ---------------------------------------------------------------------------
# Evaluation — opera su feature già proiettate e scalate (4-dim)
# ---------------------------------------------------------------------------
def evaluate_on_features(
    modules:        list,
    u_4_scaled:     torch.Tensor,   # (N, N_QUBITS) — già proiettato e scalato
    labels_np:      np.ndarray,
    criterion,
    device,
    batch_size:     int = EVAL_BATCH_SIZE,
) -> tuple:
    """Valuta su feature 4-dim pre-proiettate e scalate, in mini-batch."""
    vqc, classifier = modules
    vqc.eval()
    classifier.eval()

    eval_criterion = nn.CrossEntropyLoss(
        weight=criterion.weight,
        reduction='sum',
    )

    all_preds  = []
    total_loss = 0.0
    n_samples  = len(labels_np)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end     = min(start + batch_size, n_samples)
            u_batch = u_4_scaled[start:end]
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
# Worker — un processo per coppia (d, seed)
# ---------------------------------------------------------------------------
def train_production(d: int, seed: int, compressor: str) -> dict:
    """
    Addestra feature_selector + VQC + classifier per una tripla (d, seed, compressor).
    Le feature d-dim vengono lette dal CSV del compressore specificato.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")
    logger = get_logger(d, seed, compressor)
    logger.info(f"Avvio job d={d} seed={seed} compressor={compressor}")

    # — Caricamento feature da CSV -------------------------------------------
    try:
        u_train, y_train = load_features_from_csv('train', d, seed, compressor, device)
        u_val,   y_val   = load_features_from_csv('val',   d, seed, compressor, device)
        u_test,  y_test  = load_features_from_csv('test',  d, seed, compressor, device)
        logger.info(
            f"Feature caricate ({compressor}): "
            f"train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | "
            f"test {tuple(u_test.shape)}"
        )
        print(
            f"[{compressor} d={d} s={seed}] CSV letti: "
            f"train {tuple(u_train.shape)} | "
            f"val {tuple(u_val.shape)} | "
            f"test {tuple(u_test.shape)}",
            flush=True,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
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

    # — Circuito quantistico + VQC + classifier -------------------------------
    # Costruzione diretta, senza passare da HybridModel: niente backbone
    # ResNet, niente fit PCA — solo i tre componenti realmente usati.
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

    # — Feature selector: proiezione trainabile d → N_QUBITS -----------------
    # Ottimizzato con Adam (backprop) PRIMA del training NFT, poi congelato.
    # Questo separa i due problemi per natura dell'ottimizzatore:
    #   Adam → 128 param (d=32) o 16 param (d=4), gestisce entrambi in modo
    #          equivalente grazie al gradiente esatto — d=32 atteso ≥ d=4.
    #   NFT  → sempre 36 param (VQC+classifier), indipendente da d.
    feature_selector = nn.Linear(d, N_QUBITS, bias=False).to(device)
    nn.init.orthogonal_(feature_selector.weight)

    n_fs = sum(p.numel() for p in feature_selector.parameters())
    n_vc = sum(
        p.numel()
        for module in (vqc, classifier)
        for p in module.parameters()
        if p.requires_grad
    )
    logger.info(
        f"Parametri: feature_selector={n_fs} (d={d}×{N_QUBITS}, Adam) | "
        f"VQC+classifier={n_vc} (NFT)"
    )
    print(
        f"[d={d} s={seed}] param: fs={n_fs} (Adam) | vqc+clf={n_vc} (NFT)",
        flush=True,
    )

    # — Pre-training feature_selector con Adam --------------------------------
    try:
        pretrain_feature_selector(feature_selector, u_pool, y_pool, criterion, device)
        logger.info(f"Pre-training feature_selector completato ({PRETRAIN_STEPS} step Adam)")
        print(f"[d={d} s={seed}] pre-training Adam completato", flush=True)
    except Exception:
        logger.error(f"Errore pre-training:\n{traceback.format_exc()}")
        raise

    # Congela feature_selector — NFT non lo toccherà più
    for p in feature_selector.parameters():
        p.requires_grad = False

    # — Pre-proiezione di pool, val, test (UNA SOLA VOLTA) -------------------
    # feature_selector è congelato: le feature 4-dim scalate sono deterministiche.
    # Il hot loop NFT chiama solo vqc + classifier — zero overhead di proiezione.
    with torch.no_grad():
        u_pool_4s = scale_features(feature_selector(u_pool))  # (N_pool, 4)
        u_val_4s  = scale_features(feature_selector(u_val))   # (N_val,  4)
        u_test_4s = scale_features(feature_selector(u_test))  # (N_test, 4)
    logger.info(
        f"Feature pre-proiettate: pool {tuple(u_pool_4s.shape)} | "
        f"val {tuple(u_val_4s.shape)} | test {tuple(u_test_4s.shape)}"
    )

    # modules NFT: solo VQC + classifier (36 parametri, identici per tutti i d)
    modules_nft = [vqc, classifier]

    if n_vc == 0:
        raise RuntimeError(
            f"d={d} seed={seed}: nessun parametro trainable per NFT "
            f"(vqc+classifier={n_vc}). Verificare requires_grad."
        )

    # — NFT — budget flat 300 per tutti i d (5.77 sweep su 36 param) ---------
    # Non serve più scalare il budget con d: feature_selector è già ottimizzato
    # e congelato, NFT opera sempre sugli stessi 36 parametri.
    optimizer = NFT(maxfev=MAX_EVALS_NFT)
    epoch_rng = np.random.default_rng(seed)

    best_val_macro_f1  = 0.0
    best_test_macro_f1 = 0.0
    best_loss          = float('inf')
    patience_ctr       = 0
    history            = []
    os.makedirs("experiments/models", exist_ok=True)

    n_sweeps = MAX_EVALS_NFT / n_vc
    print(
        f">>> NFT [{compressor}]: d={d} seed={seed} | "
        f"maxfev={MAX_EVALS_NFT} ({n_sweeps:.1f} sweep su {n_vc} param) | patience={PATIENCE}",
        flush=True,
    )
    logger.info(
        f"Inizio NFT maxfev={MAX_EVALS_NFT} "
        f"({n_sweeps:.1f} sweep su {n_vc} param) patience={PATIENCE}"
    )

    for epoch in range(EPOCHS):
        t0 = time.time()

        try:
            # Campiona batch dal pool già pre-proiettato e scalato (4-dim)
            u_batch_4s, y_batch_np = sample_balanced_batch(
                u_pool_4s, y_pool, SAMPLES_PER_CLASS, epoch_rng
            )
            y_batch_t = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            loss_fn = make_loss_fn(modules_nft, u_batch_4s, y_batch_t, criterion)
            result  = optimizer.minimize(
                fun=loss_fn, x0=get_trainable_params(modules_nft)
            )
            set_trainable_params(modules_nft, result.x)

            val_loss,  val_acc,  val_macro_f1,  _            = evaluate_on_features(
                modules_nft, u_val_4s,  y_val,  criterion, device
            )
            test_loss, test_acc, test_macro_f1, per_class_f1 = evaluate_on_features(
                modules_nft, u_test_4s, y_test, criterion, device
            )

        except Exception:
            logger.error(f"Epoch {epoch+1}:\n{traceback.format_exc()}")
            print(
                f"[WARN] d={d} s={seed} | Epoch {epoch+1} fallita — continuo",
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
            ckpt_path = f"experiments/models/best_vqc_{compressor}_d{d}_seed{seed}.pth"
            try:
                # Salva solo i tre moduli realmente trainabili.
                # Niente più pesi ResNet18 frozen inclusi per errore nel
                # checkpoint (~44MB sprecati per file con il vecchio
                # model.state_dict() di HybridModel).
                torch.save(
                    {
                        'feature_selector': feature_selector.state_dict(),  # congelato, ottimizzato Adam
                        'vqc':              vqc.state_dict(),
                        'classifier':       classifier.state_dict(),
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
        "compressor":         compressor,
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

    jobs = [(d, s, c) for c in COMPRESSORS for d in DIMS for s in SEEDS]

    print(f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli...")
    print(f"[INFO] Compressori: {COMPRESSORS} | Dimensioni: {DIMS} | Seed: {SEEDS}")
    print(
        f"[INFO] Fase 1 — Adam pre-training feature_selector: {PRETRAIN_STEPS} step, lr={PRETRAIN_LR}"
    )
    print(
        f"[INFO] Fase 2 — NFT su VQC+classifier: maxfev={MAX_EVALS_NFT} (36 param fissi, "
        f"{MAX_EVALS_NFT/36:.1f} sweep) | epochs={EPOCHS} | patience={PATIENCE}"
    )
    print(
        f"[INFO] feature_selector d→{N_QUBITS} | "
        f"pool={SAMPLES_POOL_PER_CLASS}/classe | batch={SAMPLES_PER_CLASS}/classe\n"
    )

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
                best_loss_str = (
                    f"{result['best_loss']:.4f}"
                    if np.isfinite(result['best_loss'])
                    else "nan"
                )
                print(
                    f"[DONE] {c} d={d} s={s} → "
                    f"val F1: {result['best_val_macro_f1']:.4f} | "
                    f"test F1: {result['best_test_macro_f1']:.4f} | "
                    f"loss: {best_loss_str}",
                    flush=True,
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] {c} d={d} s={s} → limite {JOB_TIMEOUT_SEC // 3600}h",
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
        df.to_csv("experiments/production_summary.csv", index=False)

        history_files = [
            f"experiments/history/log_{c}_d{d}_s{s}.csv"
            for c, d, s in [(c, d, s) for c in COMPRESSORS for d in DIMS for s in SEEDS]
            if os.path.exists(f"experiments/history/log_{c}_d{d}_s{s}.csv")
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