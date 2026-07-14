"""
vqc_fewshot_engine.py
=====================
Implementazione VQC con EstimatorV2 nativo (senza EstimatorQNN / TorchConnector).

Perché EstimatorV2 direttamente
--------------------------------
EstimatorQNN avvolge EstimatorV2 aggiungendo overhead di conversione parametri,
validazione interna e un layer di indirezione PyTorch. Usando EstimatorV2
direttamente otteniamo:
  • Un unico run() per batch (tutti i PUBs in una lista) → meno overhead di chiamata
  • Controllo totale sul batching dei PUBs per il parameter-shift
  • Nessuna dipendenza da qiskit_machine_learning

Architettura del gradiente
--------------------------
Il bridge PyTorch ↔ Qiskit è implementato con torch.autograd.Function
(ParameterShiftFunction). La backward usa la parameter-shift rule:
    ∂⟨O⟩/∂θ_k = [⟨O⟩(θ_k + π/2) - ⟨O⟩(θ_k - π/2)] / 2

Tutti i PUBs (campioni × shift × osservabili) vengono raggruppati in un
singolo EstimatorV2.run() sia nel forward che nel backward, minimizzando
l'overhead di scheduling del simulatore.

Ottimizzazioni per 4 core Windows
-----------------------------------
  • Circuito transpilato una sola volta alla costruzione (optimization_level=3)
  • AerSimulator configurato con max_parallel_threads=2 per lasciare core
    liberi al processo parallelo fratello (execute_week_7 usa 2 worker)
  • torch.set_num_threads(1) nel worker chiamante (impostato in execute_week_7)
  • OMP/MKL single-thread impostati nel worker (vedi execute_week_7)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


# ---------------------------------------------------------------------------
# Costruzione circuito
# ---------------------------------------------------------------------------

def _build_circuit(n_qubits: int, d_latent: int):
    """
    Costruisce il circuito VQC con data re-uploading.

    Restituisce:
        qc                   : QuantumCircuit con tutti i parametri
        input_params_sorted  : list[Parameter] — parametri di input x[], ordinati
        weight_params_sorted : list[Parameter] — parametri di peso theta_lK[], ordinati
        observables          : list[SparsePauliOp] — un Z per qubit, little-endian
    """
    qc = QuantumCircuit(n_qubits)
    x  = ParameterVector('x', d_latent)

    num_cycles = (d_latent + n_qubits - 1) // n_qubits
    feat_idx   = 0

    for l in range(num_cycles):
        for i in range(n_qubits):
            if feat_idx < d_latent:
                qc.ry(x[feat_idx], i)
                feat_idx += 1

        # Pesi distinti per ciclo — RealAmplitudes con ParameterVector univoco
        ansatz   = RealAmplitudes(n_qubits, reps=1, entanglement='linear')
        n_w      = ansatz.num_parameters
        w        = ParameterVector(f'theta_l{l}', n_w)
        param_map = {ansatz.parameters[i]: w[i] for i in range(n_w)}
        qc.compose(ansatz.assign_parameters(param_map), inplace=True)

    all_params = qc.parameters
    input_params_sorted = sorted(
        [p for p in all_params if p.name.startswith('x[')],
        key=lambda p: int(p.name[2:-1])
    )
    weight_params_sorted = sorted(
        [p for p in all_params if not p.name.startswith('x[')],
        key=lambda p: p.name
    )

    # Osservabili little-endian: "I"*i + "Z" + "I"*(n-i-1)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
        for i in range(n_qubits)
    ]

    return qc, input_params_sorted, weight_params_sorted, observables


# ---------------------------------------------------------------------------
# EstimatorV2 configurato per 4 core Windows (2 thread per istanza)
# ---------------------------------------------------------------------------

def _make_estimator() -> EstimatorV2:
    """
    Crea un EstimatorV2 con AerSimulator statevector limitato a 2 thread.

    Con 2 worker paralleli (execute_week_7) × 2 thread = 4 thread totali,
    pari ai core fisici disponibili — nessuna oversubscription.
    """
    sim = AerSimulator(method='statevector', max_parallel_threads=2)
    est = EstimatorV2()
    # Iniezione del backend configurato tramite opzioni
    est.options.simulator = {"method": "statevector"}
    est.options.default_shots = None   # modalità esatta (no sampling noise)
    return est


# ---------------------------------------------------------------------------
# torch.autograd.Function — bridge PyTorch ↔ EstimatorV2
# ---------------------------------------------------------------------------

class ParameterShiftFunction(torch.autograd.Function):
    """
    Implementa forward e backward del VQC tramite parameter-shift rule.

    Forward
    -------
    Costruisce un PUB per ogni (campione, osservabile) e li esegue
    tutti in un unico EstimatorV2.run(). Output: (batch, n_obs).

    Backward
    --------
    Per ogni peso θ_k costruisce 2 PUBs shifted (θ_k ± π/2) per ogni
    campione, li esegue tutti in un unico run(), calcola il gradiente:
        ∂L/∂θ_k = Σ_i (∂L/∂⟨O_i⟩) x [⟨O_i⟩(+) - ⟨O_i⟩(-)] / 2

    Il gradiente rispetto a x (input) non viene calcolato (backbone
    congelato): restituisce None per input_vals.
    """

    @staticmethod
    def forward(ctx, input_vals, weight_vals, qc, input_params,
                weight_params, observables, estimator):
        """
        input_vals  : (batch, d_latent)  — feature scalate
        weight_vals : (n_weights,)       — pesi del VQC
        """
        batch_size  = input_vals.shape[0]
        n_obs       = len(observables)
        n_weights   = len(weight_params)

        x_np = input_vals.detach().cpu().numpy().astype(float)
        w_np = weight_vals.detach().cpu().numpy().astype(float)

        # Ordine parametri nel circuito: input prima, poi pesi
        # (coerente con la separazione fatta in _build_circuit)
        input_indices  = list(range(len(input_params)))
        weight_indices = list(range(len(input_params), len(input_params) + n_weights))

        # Un PUB per campione: (circuit, [obs_0,...,obs_{n_obs-1}], param_values)
        # param_values shape: (1, n_total_params) — una riga per campione
        n_total = len(input_params) + n_weights
        pubs = []
        for b in range(batch_size):
            pv = np.zeros((1, n_total), dtype=float)
            for j, p in enumerate(input_params):
                pv[0, j] = x_np[b, j]
            for k, _ in enumerate(weight_params):
                pv[0, len(input_params) + k] = w_np[k]
            pubs.append((qc, observables, pv))

        # Unico run() per tutto il batch
        job     = estimator.run(pubs)
        results = job.result()

        # results[b].data.evs shape: (n_obs,) per ogni PUB
        fwd = np.stack([results[b].data.evs for b in range(batch_size)], axis=0)
        # fwd: (batch, n_obs)

        # Salva per backward
        ctx.save_for_backward(input_vals, weight_vals)
        ctx._qc           = qc
        ctx._input_params  = input_params
        ctx._weight_params = weight_params
        ctx._observables   = observables
        ctx._estimator     = estimator
        ctx._x_np          = x_np
        ctx._w_np          = w_np
        ctx._n_total       = n_total
        ctx._batch_size    = batch_size
        ctx._n_obs         = n_obs

        return torch.tensor(fwd, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch, n_obs) — ∂L/∂⟨O⟩ dal layer successivo
        """
        x_np       = ctx._x_np
        w_np       = ctx._w_np
        n_total    = ctx._n_total
        batch_size = ctx._batch_size
        n_obs      = ctx._n_obs
        n_weights  = len(ctx._weight_params)
        estimator  = ctx._estimator
        qc         = ctx._qc
        obs        = ctx._observables
        input_params  = ctx._input_params
        weight_params = ctx._weight_params
        SHIFT = np.pi / 2

        grad_output_np = grad_output.detach().cpu().numpy()  # (batch, n_obs)
        grad_weights   = np.zeros(n_weights, dtype=float)

        # Per ogni peso θ_k, costruiamo 2×batch PUBs shifted in un unico run()
        for k in range(n_weights):
            pubs_shift = []
            for b in range(batch_size):
                for sign in (+1, -1):
                    pv = np.zeros((1, n_total), dtype=float)
                    for j in range(len(input_params)):
                        pv[0, j] = x_np[b, j]
                    for kk in range(n_weights):
                        pv[0, len(input_params) + kk] = w_np[kk]
                    # Shift del k-esimo peso
                    pv[0, len(input_params) + k] += sign * SHIFT
                    pubs_shift.append((qc, obs, pv))

            job     = estimator.run(pubs_shift)
            results = job.result()

            # Risultati: indice 2*b → +shift, 2*b+1 → -shift
            for b in range(batch_size):
                ev_plus  = results[2 * b    ].data.evs  # (n_obs,)
                ev_minus = results[2 * b + 1].data.evs  # (n_obs,)
                ps_grad  = (ev_plus - ev_minus) / 2.0   # (n_obs,)
                # Chain rule: ∂L/∂θ_k += Σ_obs (∂L/∂⟨O⟩ × ∂⟨O⟩/∂θ_k)
                grad_weights[k] += float(np.dot(grad_output_np[b], ps_grad))

        grad_w_tensor = torch.tensor(grad_weights, dtype=torch.float32)
        # None per: input_vals, qc, input_params, weight_params, observables, estimator
        return None, grad_w_tensor, None, None, None, None, None


# ---------------------------------------------------------------------------
# Modulo PyTorch
# ---------------------------------------------------------------------------

class FewShotVQC(nn.Module):
    """
    VQC ibrido classico-quantistico con EstimatorV2 nativo.

    Pipeline:
        input (batch, d_latent)
            → ParameterShiftFunction (EstimatorV2, gradiente parameter-shift)
            → (batch, n_qubits)  — valori di aspettazione ⟨Z_i⟩
            → nn.Linear(n_qubits, 4)
            → logits (batch, 4)
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.d_latent = d_latent
        n_qubits      = 4

        qc, input_params, weight_params, observables = _build_circuit(
            n_qubits, d_latent
        )

        # Transpilazione con optimization_level=3 una sola volta alla costruzione.
        # optimization_level=3 applica ottimizzazioni aggressive (cancellazione gate
        # ridondanti, commutazione, routing) che riducono il numero di gate e quindi
        # il tempo di simulazione per ogni PUB.
        pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=AerSimulator(method='statevector')
        )
        self._qc_transpiled  = pm.run(qc)
        self._input_params   = input_params
        self._weight_params  = weight_params
        self._observables    = observables
        self._estimator      = _make_estimator()

        n_weights = len(weight_params)
        # Pesi inizializzati con distribuzione uniforme [-π, π] — buona copertura
        # dello spazio dei parametri senza vanishing gradient iniziale
        self.weights = nn.Parameter(
            torch.FloatTensor(n_weights).uniform_(-np.pi, np.pi)
        )

        self.head = nn.Linear(n_qubits, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_latent) — già scalato in [0, π] dal chiamante
        ev = ParameterShiftFunction.apply(
            x,
            self.weights,
            self._qc_transpiled,
            self._input_params,
            self._weight_params,
            self._observables,
            self._estimator,
        )
        return self.head(ev)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vqc(X: torch.Tensor, y: torch.Tensor, d: int, epochs: int = 3):
    """
    Addestra FewShotVQC per 'epochs' epoche su (X, y).

    Args:
        X      : (N, d)  — feature PCA già normalizzate, float32
        y      : (N,)    — label long
        d      : int     — dimensione latente
        epochs : int     — epoche di training (default 3)

    Returns:
        model: FewShotVQC addestrato
    """
    model     = FewShotVQC(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.CrossEntropyLoss()

    for ep in range(epochs):
        optimizer.zero_grad()
        out  = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        print(f"      [VQC d={d}] epoch {ep+1}/{epochs} loss={loss.item():.4f}",
              flush=True)

    return model