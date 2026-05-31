fvabquantum — README
=====================

Panoramica
---------
Questo repository contiene script per l'addestramento e la valutazione di modelli ibridi classico-quantistici (VQC) su OCTMNIST. La pipeline principale estrae embedding da un backbone ResNet, riduce la dimensionalità (PCA / autoencoder) e addestra un VQC usando un ottimizzatore QN‑SPSA che lavora su simulatore statevector (Qiskit Aer).

Questo README descrive i prerequisiti, l'ordine consigliato per l'esecuzione degli script, i file prodotti e alcuni suggerimenti pratici per eseguire gli esperimenti su macchine locali (Windows).

Requisiti minimali
------------------
- Python 3.8+ (consigliato 3.8–3.11)
- Spazio disco: almeno qualche GB per i pesi e i dataset temporanei.
- Memoria: la simulazione statevector (Aer) è onerosa; per esperimenti completi servono decine di GB di RAM se si aumenta il numero di qubit o la dimensione del batch.

Dipendenze (raccomandate)
-------------------------
- torch, torchvision
- numpy, pandas
- scikit-learn
- qiskit, qiskit-aer, qiskit-algorithms
- medmnist
- requests

Installazione rapida (esempio Windows)
-------------------------------------
1. Crea e attiva un virtualenv:

   python -m venv .venv
   .venv\Scripts\activate

2. Aggiorna pip e installa le dipendenze base (adatta la riga di torch al tuo hardware seguendo le istruzioni ufficiali):

   python -m pip install -U pip
   python -m pip install numpy pandas scikit-learn medmnist requests qiskit qiskit-aer qiskit-algorithms
   # Installazione PyTorch (CPU example):
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Nota: per GPU installa la versione di PyTorch compatibile con la tua versione di CUDA.

Esecuzione: flusso consigliato e file prodotti
---------------------------------------------
Segui quest'ordine per riprodurre gli esperimenti e generare gli artefatti usati negli script:

1) Generare gli split del dataset

   python generate_splits.py

   Output: in `dataset_splits/` troverai i file
     - train_ids_11.csv, val_ids_11.csv, test_ids_11.csv
     - train_ids_17.csv, ... (per ogni seed definito)

   Questi file sono necessari a `data_loader.get_data_loaders()`.

2) Estrarre embedding ResNet e generare feature PCA (script: `test.py`)

   python test.py

   Output principali (cartelle create automaticamente):
     - artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz
         (feature raw 512-dim + labels) — usato da `HybridModel` e da pipeline di compressione
     - artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv
         (feature PCA d-dim in CSV, colonne feat_0..feat_{d-1}, label)

3) Addestramento VQC di produzione (NFT)

   python train_vqc_production.py

   Cosa fa: esegue un insieme di run (per i valori `DIMS` e `SEEDS` definiti nello script), lancia più worker in parallelo, costruisce `HybridModel` (ResNet → PCA → VQC) e ottimizza i pesi variazionali con NFT (Nakanishi-Fujii-Todo).

   Output principali:
     - experiments/logs/worker_d{d}_s{seed}.log
         (log per ogni worker con traceback e info dettagliate)
     - experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth
         (checkpoint pesi del modello VQC + classifier salvati con `torch.save(model.state_dict(), ...)`)
     - experiments/production_log.csv
         (file CSV append-only con la storia degli epoch per ogni worker: epoch, d, seed, backbone, loss, val_acc)
     - experiments/production_summary.csv
         (tabella riassuntiva con i migliori risultati per run)

   Note utili:
     - Lo script usa di default `device = torch.device("cpu")` e il simulatore Aer in modalità `statevector` tramite `AerSimulator(method='statevector')`.
     - Se si hanno risorse limitate: ridurre `MAX_ITER_QNSPSA` o `EPOCHS` in `train_vqc_production.py`, o abbassare `MAX_WORKERS`.

4) Suite e benchmark finali

   - `run_production_suite.py` — lancia `train_production` in modo sequenziale (utile su macchine con poche risorse o per debug).
   - `run_final_benchmarks.py` — addestra e valuta tre classificatori classici (LogisticRegression, MLP, SVM) usando la cache di feature e salva `artifacts/final_benchmarks.csv`.

5) Workflow Team B (sweep completo)

   - `master_sweep_team_b.py` esegue l'intero esperimento Team B: addestra compressori (B1, B2, B3), addestra VQC, valuta e salva `artifacts/sweep/team_b_final_results.csv`.

Script e file principali (mappa rapida)
-------------------------------------
- `generate_splits.py` → crea `dataset_splits/*.csv` (indici train/val/test per ogni seed).
- `test.py` → estrae embedding dalla backbone ResNet e salva `artifacts/resnet/features/*.npz` e `artifacts/sweep/B1_pca_*.csv`.
- `train_vqc_production.py` → addestra VQC con QN‑SPSA; genera `experiments/models/`, `experiments/logs/`, `experiments/production_log.csv`, `experiments/production_summary.csv`.
- `run_production_suite.py` → helper per lanciare più run in sequenza.
- `run_final_benchmarks.py` → esegue i classificatori classici e salva `artifacts/final_benchmarks.csv`.
- `master_sweep_team_b.py` → pipeline completa per i compressori e valutazioni Team B (salva in `artifacts/sweep/`).
- `pca_res_compressors.py` → implementazioni di `PCACompressor` e `ResNetCompressor` (gestisce anche il download dei pesi se mancanti).
- `hybrid_engine.py`, `quantum_model.py` → definizione della pipeline ibrida, del circuito VQC e del layer quantistico (DirectVQC).

Descrizione degli artefatti prodotti (pattern)
----------------------------------------------
- `dataset_splits/train_ids_{seed}.csv` (e `val_`, `test_`): indici di campione per seed.
- `artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz`: array compressi con chiavi `features` (N×512) e `labels`.
- `artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv`: CSV con colonne `feat_0..feat_{d-1}`, `label`.
- `experiments/logs/worker_d{d}_s{seed}.log`: log testo per ogni worker del training.
- `experiments/models/best_vqc_{backbone}_d{d}_seed{seed}.pth`: stato del modello (`state_dict`).
- `experiments/production_log.csv`: file append-only con la cronologia degli epoch.
- `experiments/production_summary.csv`: tabella riassuntiva finale.
- `artifacts/final_benchmarks.csv`: risultati aggregati dei classificatori classici.
- `artifacts/sweep/team_b_final_results.csv`: output del pipeline Team B (sweep dei compressori e VQC).

Consigli pratici e troubleshooting
---------------------------------
- Errore FileNotFoundError per `dataset_splits/...`: assicurati di aver eseguito `generate_splits.py` prima di chiamare `get_data_loaders()`.
- Pesi ResNet mancanti: `ResNetCompressor` tenta di scaricarli automaticamente da Zenodo; puoi posizionare manualmente i file `.pth` nella cartella `weights/` (es. `weights/octmnist_resnet18_224.pth`).
- Memoria / tempo: la simulazione statevector è la parte più costosa. Per test rapidi riduci `DIMS` (es. solo `d=4`) e `SEEDS` o abbassa `MAX_ITER_QNSPSA` e `EPOCHS` in `train_vqc_production.py`.
- Windows + multiprocessing: gli script che usano `ProcessPoolExecutor` includono `multiprocessing.freeze_support()` e protezione `if __name__ == '__main__'` per evitare relanci ricorsivi — esegui gli script direttamente (non importandoli) da terminale.

Modifiche rapida-perf e debugging
---------------------------------
- Per eseguire un singolo run interattivo (es. d=4 seed=11) puoi chiamare direttamente la funzione contenuta in `train_vqc_production.py` (o modificare `DIMS` / `SEEDS` nello script). Esempio rapido (da prompt Python):

    from train_vqc_production import train_production
    train_production(d=4, seed=11, backbone='resnet')

- Per diminuire il carico: in `train_vqc_production.py` impostare `MAX_WORKERS = 1` e `MAX_ITER_QNSPSA = 10`.

Fine del documento.
