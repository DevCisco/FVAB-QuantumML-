# Pipeline Ibrida Classico-Quantistica — OCTMNIST (Team B)

## Indice

1. [Panoramica](#panoramica)
2. [Requisiti](#requisiti)
3. [Dipendenze](#dipendenze)
4. [Ordine di esecuzione](#ordine-di-esecuzione)
   - [1. `generate_splits.py`](#1-generate_splitspy--prerequisito)
   - [2. `test.py`](#2-testpy)
   - [3. `b2_b3_training.py`](#3-b2_b3_trainingpy)
   - [4. `train_vqc_production.py`](#4-train_vqc_productionpy--pipeline-principale)
   - [5. `classical_baseline.py`](#5-classical_baselinepy--baseline-di-confronto)
   - [6. `master_sweep_team_b.py`](#6-master_sweep_team_bpy--ablation-no-quantum--confronto-compressori)
   - [7. `week8_evaluation.py`](#7-week8_evaluationpy)
   - [8. `execute_week_7.py`](#8-execute_week_7py)
   - [9. `week4_sweep_production.py`](#9-week4_sweep_productionpy--indipendente)
5. [Deliverable finale](#deliverable-finale)
6. [Stato dei risultati](#stato-dei-risultati)
7. [Mappa rapida script → output](#mappa-rapida-script--output)
8. [Troubleshooting](#troubleshooting)
9. [Esecuzione rapida di debug](#esecuzione-rapida-di-debug)

---

## Panoramica

Pipeline ibrida classico-quantistica per la classificazione di **OCTMNIST** (4 classi, immagini OCT della retina) — contributo del **Team B (Gruppo 22)**, mandato: **esplorazione della compressione degli embedding**.

Un backbone **ResNet18** (pesi ufficiali MedMNIST, congelato) estrae feature a 512 dimensioni, compresse a `d` componenti da tre compressori confrontabili tra loro:

| Compressore | Metodo | Script che lo produce |
|:-----------:|:-------|:-----------------------|
| **B1** | PCA (sklearn) | `test.py` |
| **B2** | VanillaAE (`512→128→d→128→512`, MSE) | `b2_b3_training.py` |
| **B3** | RegularizedAE (`512→d(sigmoid)→512`, MSE + L1) | `b2_b3_training.py` |

Le feature compresse alimentano un **circuito quantistico variazionale** (VQC, 4 qubit, simulatore Aer statevector) con **data re-uploading multi-ciclo**: quando `d > n_qubit`, le feature vengono caricate a blocchi sugli stessi 4 qubit, alternando blocchi di encoding angolare (Ry) e blocchi variazionali (RealAmplitudes, pesi indipendenti per ciclo). L'ottimizzazione dei pesi variazionali usa **NFT** (Nakanishi-Fujii-Todo), non Adam.

### Percorso di conformità (sintesi)

L'architettura attuale è il punto di arrivo di un percorso di verifica rispetto al documento di progetto:

1. **Test set fisso e mai visto dal backbone** — canonical val+test MedMNIST, condiviso da tutti i seed; il seed controlla solo la suddivisione train/val.
2. **Data re-uploading genuino** — sostituisce una prima implementazione che troncava le feature a `min(d, n_qubit)`, rendendo `d` di fatto irrilevante. Ora tutte le `d` componenti entrano nel circuito, a blocchi.
3. **Pesi ansatz indipendenti per ciclo** — una variante a pesi condivisi tra cicli (schema Pérez-Salinas et al.) è stata testata, verificata tecnicamente corretta ma empiricamente peggiore, e scartata.
4. **Scaler fittato solo su train** — sostituisce un primo scaling per-campione, non conforme al principio "fit solo su train" richiesto dal documento.
5. **NFT al posto di Adam** — deviazione esplicitamente giustificata dalla clausola di flessibilità sugli iperparametri del documento, con evidenza di convergenza documentata.
6. **Budget di entangling interpretato per singolo blocco ansatz, non sull'intero circuito** — decisione finale, unica lettura coerente col re-uploading obbligatorio.

---

## Requisiti

| Componente    | Dettaglio                                                                 |
|:--------------|:--------------------------------------------------------------------------|
| Python        | 3.8–3.11                                                                  |
| CPU           | Testato su Intel i5-1135G7 (4 core fisici, 8 thread) — nessuna GPU richiesta |
| RAM           | 8 GB sufficienti                                                          |
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
| 1 | `generate_splits.py` | ⚠️ Prerequisito — vedi nota sotto |
| 2 | `test.py` | Produce le feature B1 (PCA) |
| 3 | `b2_b3_training.py` | Produce le feature B2/B3 (AE) |
| 4 | `train_vqc_production.py` | VQC con re-uploading + NFT — **pipeline principale** |
| 5 | `classical_baseline.py` | Baseline MLP few-shot (batch fisso, stesso regime dati del VQC) |
| 6 | `master_sweep_team_b.py` | Ablation no-quantum + screening LR/MLP/RBF-SVM |
| 7 | `week8_evaluation.py` | Valutazione pulita multi-split, tutti i compressori |
| 8 | `execute_week_7.py` | Sweep few-shot (LR + VQC) |
| 9 | `week4_sweep_production.py` | StabilizedVQC, backpropagation classica (indipendente) |

---

### 1. `generate_splits.py` — prerequisito

Genera i tre CSV richiesti da `data_loader.py` in `dataset_splits/`:

| File | Descrizione |
|:-----|:------------|
| `train_ids_{seed}.csv` | Per ciascun seed in `[11, 17, 29]` |
| `val_ids_{seed}.csv` | Per ciascun seed in `[11, 17, 29]` |
| `test_ids_fixed.csv` | **Uno solo, condiviso da tutti i seed** |

**Algoritmo**: carica i tre split canonici OCTMNIST senza transform (solo dimensioni). `test_ids_fixed.csv` = indici `[n_train, n_train+n_val+n_test)` (canonical val+test, ~11.832 immagini). `train_ids_{seed}.csv`/`val_ids_{seed}.csv` = permutazione deterministica (`torch.Generator().manual_seed(seed)`) del canonical train, split 90/10 (~87.729/~9.748).

> **⚠️ Dipendenza critica con `data_loader.py`**: gli indici presuppongono lo stesso ordine di concatenazione (`ConcatDataset([train, val, test])`). Se cambia in uno dei due file, va cambiato identicamente nell'altro.

Include verifica esplicita di non-overlap tra `test_ids_fixed` e ciascun `train_ids_{seed}`/`val_ids_{seed}` — solleva `RuntimeError` se rileva sovrapposizioni.

---

### 2. `test.py`

Estrae embedding ResNet18 (512-dim) e li riduce con PCA per ogni `(d, seed)`, `d ∈ {32,16,8,4}`, `seed ∈ {11,17,29}`.

**Output:**
- `artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz` — feature raw 512-dim, necessario solo per `b2_b3_training.py`.
- `artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv` — feature PCA `d`-dim, colonne `feat_0..feat_{d-1}, label`.

---

### 3. `b2_b3_training.py`

Addestra **B2 (VanillaAE)** e **B3 (RegularizedAE)**.

| Parametro | Valore |
|:---|:---|
| `DIMS` | [32,16,8,4] |
| `SEEDS` | [11,17,29] |
| `EPOCHS` | 30 |
| `BATCH_SIZE` | 256 |
| `LR` | 1e-3 (Adam) |

**Output:**
- `artifacts/sweep/B2_d{d}_s{seed}.pt`, `B3_d{d}_s{seed}.pt` — pesi del modello (richiesti da `week8_evaluation.py`).
- `artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv`, `B3_pca_{split}_d{d}_seed{seed}.csv` — feature latenti (colonne `latent_0..latent_{d-1}, label`).
- `artifacts/ae_training_report.csv`.

---

### 4. `train_vqc_production.py` — pipeline principale

Addestra il VQC per ogni combinazione `(d, seed, compressore)` — 36 job totali. Legge le feature direttamente dai CSV di `test.py`/`b2_b3_training.py`: nessuna dipendenza da ResNet18/PCA a runtime.

#### Architettura

```text
u (d-dim, dal compressore)
        ↓
fit_scaler (SOLO su train) + apply_scaler → [0, 2π] + pad_features
        ↓
DirectVQC — QuantumPipeline con data re-uploading:
    n_cicli = ⌈d / 4⌉
    ogni ciclo: RY su 4 feature del blocco corrente
              → ansatz RealAmplitudes (reps=2, indipendente per ciclo)
        ↓
classifier = nn.Linear(4, 4)
```

Nessuna dipendenza da `HybridModel` o da un `feature_selector`: il circuito consuma direttamente il vettore `d`-dim scalato/paddato.

#### Parametri della run riportata nel paper

| Parametro | Valore |
|:---|:---|
| `COMPRESSORS` | `['B1', 'B2', 'B3']` |
| `DIMS` | `[32, 16, 8, 4]` |
| `SEEDS` | `[11, 17, 29]` |
| `EPOCHS` | 10 |
| `PATIENCE` | 3 |
| `NFT_TARGET_SWEEPS` | 3 |
| `MAX_EVALS_NFT_BASE` | 300 |
| `N_LAYERS` | 2 |
| `ENCODING_SCALE_MAX` | `π` |
| `SAMPLES_PER_CLASS` | 8 |
| `SAMPLES_POOL_PER_CLASS` | 64 |
| `MAX_WORKERS` | 4 |

> **📝 Tuning valutato ma MAI eseguito**: nel codice è documentato (commenti in cima al file) un tentativo di tuning per ridurre il gap VQC-classico — `EPOCHS=15`, `PATIENCE=5`, `NFT_TARGET_SWEEPS=5`, stimato in ~1.5-2.5× il tempo di calcolo, concentrato sui job a `d` alta. **Non è stato eseguito**: i valori correnti nel file (10/3/3) sono quelli della run effettivamente riportata nel paper. Se in futuro si vuole provare il tuning, i valori da modificare sono chiaramente commentati nel file — ricordarsi di rieseguire anche `week8_evaluation.py` dopo, perché i checkpoint cambiano.

#### Scelte chiave e motivazione

| Scelta | Valore | Perché |
|:---|:---|:---|
| `ENCODING_SCALE_MAX` | `2π` — decisione finale | `RY(θ)` applicato a `\|0>` traccia un percorso sul meridiano della sfera di Bloch: a `θ=0` sei al polo nord (`\|0>`), a `θ=π` al polo sud (`\|1>`), a `θ=2π` sei tornato al polo nord. Visivamente, `2π` corrisponde a "un giro completo" — è la scelta intuitiva se si pensa alla rotazione come un cerchio da percorrere per intero, senza fare il conto specifico su cosa succede alla misura di `⟨Z⟩` lungo il percorso.. |
| Pesi ansatz **indipendenti** per ciclo — decisione finale | — | Variante a pesi condivisi testata e scartata: tecnicamente corretta ma empiricamente sotto il livello casuale. |
| `N_LAYERS = 2` | RealAmplitudes reps, per singolo blocco | Allineato al budget di entangling suggerito dal documento, interpretato per-blocco (unica lettura compatibile col re-uploading). |
| `get_max_evals_nft` | scala con `n_param` | I parametri crescono con `d` (più cicli). Budget NFT scalato di conseguenza, floor a 300. |
| Scaler fit-on-train | `fit_scaler`/`apply_scaler` | Vero scaler min-max per-feature, fittato solo su train — stesso principio di PCA/AE. |

#### Interpretazione val vs test

Train e val provengono dal canonical train MedMNIST (immagini viste dal backbone durante il pre-training): il **val macro-F1 è strutturalmente ottimistico**. Il **test macro-F1** (mai visto dal backbone) è l'unica metrica indicativa della generalizzazione reale. Nella run finale, il gap medio val-test è **0.032** su 36 combinazioni — molto più contenuto rispetto a configurazioni preliminari non conformi (gap ~0.13-0.15), confermando l'assenza di leakage.

**Output:**
- `experiments/logs/worker_{compressor}_d{d}_s{seed}.log`
- `experiments/models/best_vqc_{compressor}_d{d}_seed{seed}.pth` — checkpoint dict con `vqc`, `classifier`, `min_vec`, `max_vec`.
- `experiments/history/log_{compressor}_d{d}_s{seed}.csv` → unificato in `experiments/production_log.csv`
- `experiments/production_summary.csv` — colonne: `compressor, d, seed, best_loss, best_val_macro_f1, best_test_macro_f1`

---

### 5. `classical_baseline.py` — baseline di confronto

MLP a un hidden layer (32 unità, ReLU, dropout 0.3), Adam (lr=0.01, weight_decay=1e-4), fino a 100 epoche con early stopping (patience 15) — su B1/B2/B3.

> **📝 Regime dati few-shot, non full-train**: a differenza della no-quantum ablation (che usa l'intero training set), questo script estrae **un singolo batch bilanciato di 32 campioni** (8 per classe, `SAMPLES_PER_CLASS=8`) **una sola volta prima del training loop**, e lo riusa identico per tutte le epoche — stesso regime dati del VQC, ma senza il ricampionamento per-epoca che il VQC usa. Questa differenza è risultata rilevante nella Discussione del paper (Sezione 5.6): su B3, l'MLP fatica a fittare persino il proprio batch fisso (loss training vicina a `ln(4)=1.386` in 7/12 combinazioni).

**Output:** `experiments_classical/classical_summary.csv`, `classical_log.csv`.

---

### 6. `master_sweep_team_b.py` — ablation no-quantum + confronto compressori

Screening a tre vie (Logistic Regression, MLP a un hidden layer, RBF-SVM) su tutte le 36 combinazioni, con **selezione formale del comparatore primario** (solo validation, media su tutte le combinazioni e sui 3 seed) come richiesto dal documento.

> **⚠️ Due output distinti, non intercambiabili**: il documento fissa esplicitamente "la logistic regression coincide con la no-quantum ablation obbligatoria" — questo è **sempre e solo LR**, indipendentemente da quale modello vince lo screening. Un bug precedente sostituiva `team_b_comparison.csv` con i dati del modello selezionato (in una run, MLP) — **corretto**: ora `team_b_comparison.csv`/`team_b_summary.csv` sono sempre LR; l'esito dello screening (nella run corrente: **MLP**, val macro-F1 media 0.9845) è riportato separatamente in `team_b_selected_comparator.csv`.

**Metriche**: macro-F1, macro-AUROC (OvR), balanced accuracy, **ECE** (15 bin uniformi, norma L1, confidenza top-1).

**Output:**
- `artifacts/sweep/team_b_screening.csv` — dettaglio completo, 3 modelli × 36 combinazioni × val/test (108 righe)
- `artifacts/sweep/team_b_comparison.csv` — solo LR (no-quantum ablation obbligatoria)
- `artifacts/sweep/team_b_summary.csv` — media/std LR per (compressore, d)
- `artifacts/sweep/team_b_selected_comparator.csv` — esito dello screening (comparatore selezionato)

> **Risultato**: a `d` alta i tre compressori sono quasi equivalenti (differenza ≤0.012); a `d=4`, B3 crolla a macro-F1 0.7426 contro 0.8645 di B1 — un divario di oltre 12 punti assente alle dimensioni alte. B3 ha comunque l'ECE nettamente migliore su ogni dimensione (calibrazione).

---

### 7. `week8_evaluation.py`

Valuta il VQC **specifico per compressore** (checkpoint di `train_vqc_production.py`) su tutti e tre gli split (train/val/test), su feature pulite — nessuna perturbazione, nessun backbone live. Non richiede modifiche di codice quando cambia `train_vqc_production.py`: importa `N_QUBITS`, `N_CLASSES`, `N_LAYERS`, `COMPRESSOR_PATHS`, `apply_scaler`, `pad_features` in diretta — i valori aggiornati si propagano automaticamente. Va solo **rieseguito** dopo ogni run di `train_vqc_production.py` che cambi i checkpoint.

**Output:** `artifacts/week8_evaluation_report.csv` (colonne: `dim, seed, compression, split, accuracy, macro_f1`)

> **✅ Verificato**: tutti i 36 valori val/test coincidono esattamente (differenza <1e-6) con `production_summary.csv` — stesso VQC, stesse feature, due percorsi di codice indipendenti.

---

### 8. `execute_week_7.py`

Sweep few-shot: LogisticRegression + VQC su frazioni decrescenti del training set (25%, 10%, 5%), per B1/B2/B3, `D=[8,4]`, seed=`[11]` (seed=23 nel documento è un refuso confermato dal docente).

Riusa **direttamente** i building block di `train_vqc_production.py` (`QuantumPipeline`, `DirectVQC`, scaler fit-on-train, NFT, budget scalato) — stessa architettura del benchmark principale, non un modulo esterno separato. Il few-shot varia solo il training set; validation e test restano i set fissi standard.

**Output:** `artifacts/fewshot_final_results.csv` (colonne: `compressor, d, seed, fraction, macro_f1_lr, macro_f1_vqc`)

> **Risultato**: LR resta piatta per B1/B2 anche al 5% dei dati (variazione ≤0.005), mentre B3 degrada sostanzialmente (-0.126/-0.151) — conferma indipendente della fragilità di B3 a bassa disponibilità dati. Il VQC non mostra un andamento monotono con la frazione (4/6 combinazioni migliorano quando i dati calano) — con un solo seed, riportato come osservazione non conclusiva, plausibilmente dominata dalla variabilità di NFT già documentata.

---

### 9. `week4_sweep_production.py` — indipendente

Addestra `StabilizedVQC` con Adam (backpropagation diretta, nessun re-uploading) su feature **B1** soltanto. Indipendente dal resto della pipeline — il suo checkpoint non è consumato da nessun altro script. Eseguire solo se richiesto come deliverable a sé stante.

**Output:** `artifacts/checkpoints/vqc_d{d}_seed{seed}_best.pth`, `artifacts/week4_vqc_results.csv`.

---

## Deliverable finale

`short_paper_team_b.tex` / `.pdf` — short paper in formato scientifico (LaTeX, due colonne, 7 pagine), 7 sezioni (Abstract, Introduzione, Stato dell'arte, Sistema proposto, Risultati sperimentali, Discussione, Conclusioni). Tutti i dati sono reali, nessun placeholder residuo. Include il chiarimento esplicito sulla proprietà di annidamento della PCA (Sezione 5.5) — d=32 contiene sempre almeno l'informazione di d=4; il pattern opposto osservato col VQC riguarda l'estraibilità con l'ottimizzatore disponibile, non l'informazione presente nei dati.

---

## Stato dei risultati

| Artifact | Stato |
|:---|:---|
| `production_summary.csv` | ✅ Finale — run con EPOCHS=10/PATIENCE=3/NFT_TARGET_SWEEPS=3 |
| `week8_evaluation_report.csv` | ✅ Finale — verificato identico a production_summary.csv |
| `team_b_comparison.csv` / `team_b_summary.csv` | ✅ Finale — dopo il fix (sempre LR) |
| `team_b_screening.csv` / `team_b_selected_comparator.csv` | ✅ Finale |
| `classical_summary.csv` | ✅ Finale — B1/B2/B3 |
| `fewshot_final_results.csv` | ✅ Finale — B1/B2/B3, D=[8,4], seed=11 |
| `short_paper_team_b.tex`/`.pdf` | ✅ Finale — tutti i dati sopra integrati, zero placeholder |

Nessuna run pendente. Il tuning documentato in `train_vqc_production.py` (commenti) non è stato eseguito ed è opzionale per lavoro futuro.

---

## Mappa rapida script → output

| Script | Output principali |
|:-------|:------------------|
| `test.py` | `artifacts/resnet/features/res_raw_*.npz`, `artifacts/sweep/B1_pca_*.csv` |
| `save_raw_features.py` | `artifacts/resnet/features/res_raw_d_{d}_train_s{seed}.npz` |
| `b2_b3_training.py` | `artifacts/sweep/B2_*.pt`, `B3_*.pt`, `B2_pca_*.csv`, `B3_pca_*.csv` |
| `train_vqc_production.py` | `experiments/models/`, `experiments/production_summary.csv` |
| `classical_baseline.py` | `experiments_classical/classical_summary.csv` |
| `master_sweep_team_b.py` | `artifacts/sweep/team_b_screening.csv`, `team_b_comparison.csv`, `team_b_summary.csv`, `team_b_selected_comparator.csv` |
| `week8_evaluation.py` | `artifacts/week8_evaluation_report.csv` |
| `execute_week_7.py` | `artifacts/fewshot_final_results.csv` |
| `week4_sweep_production.py` | `artifacts/checkpoints/`, `artifacts/week4_vqc_results.csv` |

---

## Troubleshooting

**`FileNotFoundError` su `dataset_splits/test_ids_fixed.csv`**
Eseguire prima `generate_splits.py`. Serve un singolo `test_ids_fixed.csv` condiviso da tutti i seed.

**`mat1 and mat2 shapes cannot be multiplied`**
Verificare la shape reale dei file coinvolti: `res_raw_*.npz` sono sempre 512-dim; `B{1,2,3}_pca_*.csv` sono `d`-dim.

**`EstimatorV2`/`SamplerV2` — unexpected keyword `'backend'`**
Sintomo di un percorso di codice legacy basato su QN-SPSA. La pipeline corrente con NFT non richiede `fidelity`, `SamplerV2` né `ComputeUncompute`.

**Checkpoint B2/B3 (`.pt`) mancante per `week8_evaluation.py`**
Eseguire prima `b2_b3_training.py` — salva sia le feature CSV sia i pesi del modello.

**Macro-F1 del VQC vicina al caso casuale (~0.25)**
Verificare `ENCODING_SCALE_MAX = 2* np.pi` (non `np.pi`) e che i pesi ansatz siano indipendenti per ciclo (non condivisi) in `quantum_model.py` — entrambe le alternative sono state testate ed empiricamente scartate in via definitiva.

**`team_b_comparison.csv` sembra contenere dati di un modello diverso da LogisticRegression**
Verificare di usare la versione corrente di `master_sweep_team_b.py` — un bug precedente scriveva lì il comparatore selezionato dallo screening (es. MLP) invece della LR obbligatoria. Corretto: ora sono file separati (vedi Sezione 7).

**Windows + multiprocessing**
Tutti gli script con `ProcessPoolExecutor` chiamano `multiprocessing.freeze_support()` (prima istruzione dentro `main()`).

---

## Esecuzione rapida di debug

```python
from train_vqc_production import train_production
train_production(d=4, seed=11, compressor='B1')
```

Per ridurre il carico durante test rapidi, in `train_vqc_production.py`:

| Costante | Valore ridotto |
|:---|:---|
| `MAX_WORKERS` | 1 |
| `EPOCHS` | 2–3 |
| `MAX_EVALS_NFT_BASE` | 50–100 |

---

*Fine documento.*
