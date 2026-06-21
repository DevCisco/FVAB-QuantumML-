# Pipeline Ibrida Classico-Quantistica — OCTMNIST

## Indice

1. [Panoramica](#panoramica)
2. [Requisiti](#requisiti)
3. [Dipendenze](#dipendenze)
4. [Ordine di esecuzione](#ordine-di-esecuzione)
   - [1. `generate_fixed_splits.py`](#1-generate_fixed_splitspy--prerequisito)
   - [2. `test.py`](#2-testpy)
   - [3. `save_raw_features.py`](#3-save_raw_featurespy--opzionale)
   - [4. `week_ae_training.py`](#4-week_ae_trainingpy)
   - [5. `train_vqc_production.py`](#5-train_vqc_productionpy--pipeline-principale)
   - [6. `classical_baseline.py`](#6-classical_baselinepy--baseline-di-confronto)
   - [7. `week4_sweep_production.py`](#7-week4_sweep_productionpy)
   - [8. `week8_robustness.py`](#8-week8_robustnesspy)
   - [9. `execute_week_7.py`](#9-execute_week_7py)
5. [Script non toccati in questa sessione](#script-non-toccati-in-questa-sessione)
6. [Mappa rapida script → output](#mappa-rapida-script--output)
7. [Troubleshooting](#troubleshooting)
8. [Esecuzione rapida di debug](#esecuzione-rapida-di-debug)

---

## Panoramica

Pipeline ibrida classico-quantistica per la classificazione di **OCTMNIST** (4 classi, immagini OCT della retina).

Un backbone **ResNet18** (pesi ufficiali MedMNIST) estrae feature a 512 dimensioni, una **PCA** le riduce a `d` componenti, e un **circuito quantistico variazionale** (VQC, 4 qubit, simulatore Aer statevector) le classifica. L'ottimizzazione dei pesi variazionali usa **NFT** (Nakanishi-Fujii-Todo), non più QN-SPSA.

Il protocollo sperimentale è stato corretto durante questa sessione per eliminare due problemi di validità riscontrati nella versione iniziale:

1. **Test set fisso e mai visto dal backbone** — prima il test set veniva rimescolato per ogni seed da un pool unico (train+val+test concatenati), causando sia varianza spuria tra seed (~11% di range) sia data leakage (~89% del test era già stato visto dal ResNet durante il suo addestramento). Ora il test è fisso (canonical val+test MedMNIST) per tutti i seed, e il seed controlla solo la suddivisione train/val.

2. **Budget NFT scalato con il numero di parametri trainabili** — con `d=32` il circuito aveva troppi pochi sweep di ottimizzazione (1.8×) rispetto a `d=4` (5.8×), causando fallimenti di convergenza imprevedibili. Il budget ora si adatta automaticamente.

---

## Requisiti

| Componente    | Dettaglio                                                                 |
|:--------------|:--------------------------------------------------------------------------|
| Python        | 3.8–3.11                                                                  |
| CPU           | Testato su Intel i5-1135G7 (4 core fisici, 8 thread) — nessuna GPU richiesta |
| RAM           | 8 GB sufficienti con la pipeline corrente (vedi nota su rimozione backbone ridondante) |
| Spazio disco  | Qualche GB per pesi, dataset e checkpoint                                 |

---

## Dipendenze

```text
torch, torchvision
numpy, pandas
scikit-learn
qiskit, qiskit-aer, qiskit-algorithms   # qiskit-algorithms deve esporre NFT
medmnist
requests
```

### Installazione (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install numpy pandas scikit-learn medmnist requests qiskit qiskit-aer qiskit-algorithms
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Ordine di esecuzione

| # | Script | Note |
|:-:|:-------|:-----|
| 1 | `generate_fixed_splits.py` | ⚠️ Prerequisito — vedi nota sotto |
| 2 | `test.py` | |
| 3 | `save_raw_features.py` | Opzionale — vedi nota sotto |
| 4 | `week_ae_training.py` | Necessario solo per B2/B3 |
| 5 | `train_vqc_production.py` | VQC con NFT — **pipeline principale** |
| 6 | `classical_baseline.py` | Baseline MLP per confronto |
| 7 | `week4_sweep_production.py` | StabilizedVQC, backpropagation classica |
| 8 | `week8_robustness.py` | Audit robustezza con rumore gaussiano |
| 9 | `execute_week_7.py` | Sweep few-shot, frazioni di training set |

---

### 1. `generate_fixed_splits.py` — prerequisito

> ⚠️ **Non incluso tra i file consegnati in questa sessione.**

`data_loader.py` richiede tre CSV nella cartella `dataset_splits/`:

| File | Descrizione |
|:-----|:------------|
| `train_ids_{seed}.csv` | Per ciascun seed in `[11, 17, 29]` |
| `val_ids_{seed}.csv` | Per ciascun seed in `[11, 17, 29]` |
| `test_ids_fixed.csv` | **Uno solo, condiviso da tutti i seed** |

**Composizione attesa** (vedi `data_loader.py` per i dettagli):

- `test_ids_fixed.csv` → canonical val + canonical test MedMNIST (~11.832 immagini), mai viste dal backbone ResNet durante il suo addestramento.
- `train_ids_{seed}.csv` / `val_ids_{seed}.csv` → suddivisione seed-dipendente del canonical train MedMNIST (~87.729 + ~9.748 immagini).

Se questo script non esiste ancora nel progetto, va scritto prima di proseguire — `get_data_loaders()` solleva `FileNotFoundError` esplicito se i CSV mancano.

---

### 2. `test.py`

Estrae embedding ResNet18 (512-dim) e li riduce con PCA per ogni combinazione `(d, seed)` con `d ∈ {32, 16, 8, 4}` e `seed ∈ {11, 17, 29}`.

**Output:**

| File | Descrizione |
|:-----|:------------|
| `artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz` | Feature raw 512-dim (chiavi `features`, `labels`), per `split ∈ {train, val, test}`. Necessario solo per `week_ae_training.py` (B2/B3). |
| `artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv` | Feature PCA `d`-dim, colonne `feat_0..feat_{d-1}`, `label`. Input principale per `train_vqc_production.py`, `classical_baseline.py`, `execute_week_7.py`. |

> **⚠️ Gap noto:** la versione corrente di `test.py` **non chiama `pca.save_model(...)`**, quindi `artifacts/sweep/B1_pca_d{d}.pkl` non viene generato. `week8_robustness.py` lo richiede per la valutazione B1 (`joblib.load(...)`).
>
> **Fix:** aggiungere `pca.save_model(f"artifacts/sweep/B1_pca_d{d}.pkl")` dopo il fit nella funzione `main()` di `test.py`.

---

### 3. `save_raw_features.py` — opzionale

Script standalone che salva **solo** lo split `train` delle feature raw 512-dim, per tutte le combinazioni `(d, seed)`:

```text
artifacts/resnet/features/res_raw_d_{d}_train_s{seed}.npz
```

È ridondante rispetto a `test.py` (che salva già train+val+test), utile solo se si vuole rigenerare o aggiornare le sole feature di training senza rieseguire l'intera pipeline PCA.

---

### 4. `week_ae_training.py`

Addestra **B2 (VanillaAE)** e **B3 (RegularizedAE)** — i due compressori alternativi alla PCA, definiti in `unsupervised_models.py`. Legge le feature raw 512-dim prodotte da `test.py` / `save_raw_features.py`.

**Parametri:**

| Parametro     | Valore         |
|:--------------|:---------------|
| `DIMS`        | [32, 16, 8, 4] |
| `SEEDS`       | [11, 17, 29]   |
| `EPOCHS`      | 30             |
| `BATCH_SIZE`  | 256            |
| `LR`          | 1e-3 (Adam)    |

**Architetture** (`unsupervised_models.py`):

- **B2 — VanillaAE:** `512 → 128 → d_latent → 128 → 512`, loss = MSE
- **B3 — RegularizedAE:** `512 → d_latent (sigmoid) → 512` (shallow), loss = MSE + `1e-3 · mean(|z|)` (penalità L1 esplicita via `model.sparsity_loss(z)`)

**Output:** `artifacts/sweep/B2_d{d}_s{seed}.pt`, `artifacts/sweep/B3_d{d}_s{seed}.pt`, `artifacts/ae_training_report.csv`

---

### 5. `train_vqc_production.py` — pipeline principale

Addestra il circuito VQC con **NFT** per ogni combinazione `(d, seed)`, leggendo le feature PCA direttamente dai CSV di `test.py` (**nessuna dipendenza da ResNet18/PCA a runtime** — vedi nota architetturale sotto).

#### Architettura trainabile

```text
feature_selector = nn.Linear(d, N_QUBITS, bias=False)   # proiezione d → 4, init ortogonale
        ↓
scale_features([0, π])                                   # angle encoding
        ↓
DirectVQC (QuantumPipeline: 4 qubit, RealAmplitudes reps=3, entanglement lineare)
        ↓
classifier = nn.Linear(N_QUBITS, N_CLASSES)
```

> **📝 Nota architetturale:** nelle versioni precedenti il modello passava da `HybridModel`, che istanziava sempre un `ResNetCompressor` completo (~11.2M parametri) e rifittava una PCA su 10.000 campioni raw — **anche se quei componenti non venivano mai usati**, dato che le feature arrivano già pronte dai CSV. Questo è stato rimosso: il circuito viene ora costruito direttamente da `QuantumPipeline` + `DirectVQC` (importati da `quantum_model.py` e `hybrid_engine.py`), eliminando dodici istanziazioni inutili di ResNet18 per run e checkpoint gonfiati di decine di MB di pesi mai usati.

> **Perché `feature_selector` esiste:** il circuito ha sempre 4 qubit, quindi userebbe sempre e solo le prime 4 delle `d` componenti PCA per costruzione. `feature_selector` è una proiezione lineare trainabile che apprende quale combinazione delle `d` componenti è più discriminativa, rendendo `d` effettivamente rilevante per il risultato (prima dell'introduzione di questo layer, `d=32` e `d=4` producevano risultati quasi identici).

#### Budget NFT scalato

```python
NFT_TARGET_SWEEPS = 3
maxfev = max(300, n_trainable_params * NFT_TARGET_SWEEPS)
```

| d  | Parametri trainabili | maxfev | Sweep effettivi |
|:--:|:--------------------:|:------:|:---------------:|
| 4  | 52                   | 300    | 5.8×            |
| 8  | 68                   | 300    | 4.4×            |
| 16 | 100                  | 300    | 3.0×            |
| 32 | 164                  | 492    | 3.0×            |

Solo `d=32` riceve un budget maggiore del floor di 300 — le altre dimensioni restano invariate per non alterare configurazioni già validate.

#### Parametri globali

| Parametro                        | Valore                                      |
|:---------------------------------|:--------------------------------------------|
| `EPOCHS`                         | 10                                          |
| `PATIENCE` (early stopping su val macro-F1) | 3                                |
| `SAMPLES_PER_CLASS` (batch per epoca) | 8                                      |
| `SAMPLES_POOL_PER_CLASS` (pool di diversità) | 64                              |
| `MAX_WORKERS`                    | 4                                           |
| `N_LAYERS` (RealAmplitudes reps) | 3 → 16 parametri variazionali               |
| `EVAL_BATCH_SIZE`                | 512                                         |
| `JOB_TIMEOUT_SEC`                | 28800 (8h)                                  |

#### Interpretazione val vs test

> **⚠️ Importante:** poiché train e val provengono dal canonical train MedMNIST (le stesse immagini su cui il backbone ResNet è stato addestrato), il **val macro-F1 è strutturalmente ottimistico** (tipicamente 0.94–0.99) — non è un'anomalia, riflette che il backbone "conosce" quelle immagini.
>
> Il **test macro-F1** (tipicamente 0.79–0.85) è l'unica metrica realmente indicativa della generalizzazione, perché il test set non è mai stato visto dal backbone. Il CSV finale riporta entrambi separatamente per ogni epoca.

**Output:**

| File | Descrizione |
|:-----|:------------|
| `experiments/logs/worker_d{d}_s{seed}.log` | Log per worker |
| `experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth` | Checkpoint con `feature_selector`, `vqc`, `classifier` separati (nessun peso ResNet incluso) |
| `experiments/history/log_d{d}_s{seed}.csv` | Log per-worker, poi unificato in `production_log.csv` |
| `experiments/production_log.csv` | Colonne: `epoch, d, seed, backbone, train_loss, val_loss, val_acc, val_macro_f1, test_loss, test_acc, test_macro_f1, per_class_f1` |
| `experiments/production_summary.csv` | Colonne: `d, seed, backbone, best_loss, best_val_macro_f1, best_test_macro_f1` |

---

### 6. `classical_baseline.py` — baseline di confronto

MLP classico addestrato sulle stesse feature PCA (CSV di `test.py`), con lo stesso schema di output del VQC per un confronto diretto.

| Parametro      | Valore |
|:---------------|:-------|
| Architettura   | `Linear(d, 32) → ReLU → Dropout(0.3) → Linear(32, 4)` |
| Optimizer      | Adam, `lr=0.01`, `weight_decay=1e-4` |
| `EPOCHS`       | 100    |
| `PATIENCE`     | 15     |
| `SAMPLES_PER_CLASS` | 8 (stesso batch del VQC) |

> **Nota:** a differenza del VQC, l'MLP riceve **tutte** le `d` componenti PCA (nessuna proiezione a 4 dimensioni) — un vantaggio strutturale per il classico, utile a interpretare in modo equo il confronto.

**Output:** `experiments_classical/classical_summary.csv`, `experiments_classical/classical_log.csv`, `experiments_classical/models/`, `experiments_classical/logs/`

---

### 7. `week4_sweep_production.py`

Addestra `StabilizedVQC` con Adam (backpropagation classica, nessun simulatore quantistico nel loop di training) su feature PCA pre-calcolate. Parallelizzato con `ProcessPoolExecutor`, un worker per dimensione `d`.

**Output:** `artifacts/checkpoints/vqc_d{d}_best.pth`, `artifacts/week4_vqc_results.csv`

---

### 8. `week8_robustness.py`

Audit di robustezza al rumore gaussiano (`std=0.1`, applicato **prima** della normalizzazione ImageNet) per i tre compressori B1/B2/B3, su tutte le 36 combinazioni `(d, seed, compressore)`.

| Compressore | Modalità di valutazione |
|:------------|:------------------------|
| **B1 (PCA)** | Valutato su `train`, `val`, `test` da feature pre-calcolate (`res_d{d}_{split}.npz`) — non usa il backbone live |
| **B2/B3 (AE)** | Valutati solo su `test`, con backbone ResNet18 live + rumore applicato a runtime sulle immagini |

Parallelizzato con `ProcessPoolExecutor`.

**Output:** `artifacts/week8_robustness_report.csv` (colonne: `compression, dim, seed, split, accuracy, macro_f1`)

---

### 9. `execute_week_7.py`

Sweep few-shot: confronta **Logistic Regression** e **VQC** (`vqc_fewshot_engine.train_vqc`) su frazioni decrescenti del training set.

| Parametro    | Valore |
|:-------------|:-------|
| `FRACTIONS`  | [0.25, 0.10, 0.05] |
| `DIMS`       | [8, 4] |
| `SEEDS`      | [11]   |

> **Nota:** legge le feature da `artifacts/sweep/B1_pca_{split}_d{d}_seed{s}.csv` via `pandas.read_csv` (non `np.load` — i file sono CSV, non `.npz`).

**Output:** `artifacts/fewshot_final_results.csv`

---

## Script non toccati in questa sessione

`run_production_suite.py`, `run_final_benchmarks.py`, `master_sweep_team_b.py` (se presenti nel progetto) non sono stati rivisti durante questa sessione di lavoro.

> ⚠️ Probabilmente referenziano ancora `HybridModel` / QN-SPSA / percorsi obsoleti — verificarne la compatibilità con l'architettura corrente prima dell'uso, oppure considerare `classical_baseline.py` come sostituto aggiornato per il confronto con classificatori classici.

---

## Mappa rapida script → output

| Script | Output principali |
|:-------|:------------------|
| `test.py` | `artifacts/resnet/features/res_raw_*.npz`, `artifacts/sweep/B1_pca_*.csv` |
| `save_raw_features.py` | `artifacts/resnet/features/res_raw_d_{d}_train_s{seed}.npz` |
| `week_ae_training.py` | `artifacts/sweep/B2_*.pt`, `B3_*.pt`, `artifacts/ae_training_report.csv` |
| `train_vqc_production.py` | `experiments/models/`, `experiments/logs/`, `experiments/production_log.csv`, `experiments/production_summary.csv` |
| `classical_baseline.py` | `experiments_classical/classical_summary.csv`, `experiments_classical/classical_log.csv` |
| `week4_sweep_production.py` | `artifacts/checkpoints/vqc_d{d}_best.pth`, `artifacts/week4_vqc_results.csv` |
| `week8_robustness.py` | `artifacts/week8_robustness_report.csv` |
| `execute_week_7.py` | `artifacts/fewshot_final_results.csv` |

---

## Troubleshooting

### `FileNotFoundError` su `dataset_splits/test_ids_fixed.csv`

Eseguire prima `generate_fixed_splits.py` (vedi [sezione 1](#1-generate_fixed_splitspy--prerequisito)). I vecchi `test_ids_{seed}.csv` (uno per seed) non sono più compatibili con il protocollo corretto — serve un singolo `test_ids_fixed.csv` condiviso.

### `KeyError: 'seed'` durante la costruzione del modello

Verificare che la config passata contenga sempre `'seed'`. Dipende dalla versione di `hybrid_engine.py` installata — alcune versioni di `HybridModel.__init__` lo richiedono esplicitamente.

### `mat1 and mat2 shapes cannot be multiplied`

Verificare la shape reale dei file `.npz`/CSV coinvolti prima di assumere la pipeline di trasformazione. Le feature in `res_raw_*.npz` sono sempre 512-dim (pre-PCA); quelle in `B1_pca_*.csv` sono `d`-dim (post-PCA).

### `EstimatorV2.__init__()` / `SamplerV2.__init__()` — unexpected keyword argument `'backend'`

Sintomo di API Qiskit cambiate tra versioni. La pipeline corrente con NFT non richiede `fidelity`, `SamplerV2` né `ComputeUncompute` — se questi errori compaiono, probabilmente si sta eseguendo un percorso di codice legacy basato su QN-SPSA (ormai sostituito).

### Loss bloccata vicino a `ln(4) ≈ 1.386`

Tipico segnale di batch troppo piccolo o budget NFT insufficiente per completare anche un solo sweep. Vedi [tabella scaling NFT](#budget-nft-scalato) sopra.

### `artifacts/sweep/B1_pca_d{d}.pkl` mancante

Gap noto di `test.py` — vedi [sezione 2](#2-testpy).

### Windows + multiprocessing

Tutti gli script con `ProcessPoolExecutor` includono `multiprocessing.freeze_support()` (chiamata come prima istruzione dentro `main()`, dopo il guard `if __name__ == "__main__":`) e vanno eseguiti direttamente da terminale (`python script.py`), mai importati come modulo.

### Carico CPU/RAM eccessivo

Se si osservano tempi di esecuzione anomali (ore invece di minuti) o uso di RAM elevato per job leggeri, verificare che `train_vqc_production.py` non stia passando da `HybridModel` (che istanzia un ResNet18 completo inutilmente). La versione corretta costruisce `QuantumPipeline` + `DirectVQC` direttamente, senza dipendenze dal backbone.

---

## Esecuzione rapida di debug

```python
from train_vqc_production import train_production
train_production(d=4, seed=11, backbone='pca')
```

Per ridurre il carico durante test rapidi, modificare le costanti globali in `train_vqc_production.py`:

| Costante       | Valore ridotto consigliato |
|:---------------|:---------------------------|
| `MAX_WORKERS`  | 1                          |
| `EPOCHS`       | 2–3                        |
| `MAX_EVALS_NFT`| 50–100                     |
