import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

# FIX BUG 5: il resto del progetto usa EstimatorV2 (già corretto in hybrid_engine
# e stabilized_vqc_model). Usare la versione legacy 'Estimator' introduce
# inconsistenza nei risultati di simulazione tra moduli.
from qiskit_aer.primitives import EstimatorV2 as Estimator


class FewShotVQC(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        n_qubits = 4

        qc = QuantumCircuit(n_qubits)
        x = ParameterVector('x', d_latent)

        num_cycles = (d_latent + n_qubits - 1) // n_qubits

        # FIX BUG 3: nella versione originale 'ansatz = RealAmplitudes(...)' veniva
        # ricreato ad ogni ciclo con nomi di parametri identici. Qiskit deduplica i
        # parametri per nome: tutti i layer variazionali condividevano gli stessi
        # pesi → gradi di libertà ridotti e gradiente sbagliato.
        # Ogni ciclo ha ora il proprio ParameterVector distinto 'theta_l{l}'.
        weight_params_all = []
        feat_idx = 0
        for l in range(num_cycles):
            for i in range(n_qubits):
                if feat_idx < d_latent:
                    qc.ry(x[feat_idx], i)
                    feat_idx += 1

            ansatz_template = RealAmplitudes(n_qubits, reps=1, entanglement='linear')
            n_w = ansatz_template.num_parameters
            w = ParameterVector(f'theta_l{l}', n_w)
            weight_params_all.extend(list(w))
            param_map = {ansatz_template.parameters[i]: w[i] for i in range(n_w)}
            qc.compose(ansatz_template.assign_parameters(param_map), inplace=True)

        # FIX BUG 2: 'qc.parameters[d_latent:]' opera su un ParameterView (set
        # non ordinato): l'ordine è arbitrario e non deterministico tra esecuzioni.
        # Separiamo esplicitamente input params (prefisso 'x[') da weight params
        # (prefisso 'theta_l') e ordiniamo ciascun gruppo numericamente.
        all_params = qc.parameters
        input_params_sorted = sorted(
            [p for p in all_params if p.name.startswith('x[')],
            key=lambda p: int(p.name[2:-1])
        )
        weight_params_sorted = sorted(
            [p for p in all_params if not p.name.startswith('x[')],
            key=lambda p: p.name
        )

        # Osservabili little-endian (coerente con il resto del progetto)
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
               for i in range(n_qubits)]

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params_sorted,
            weight_params=weight_params_sorted,
            observables=obs,
            estimator=Estimator()
        )
        self.qnn_layer = TorchConnector(qnn)

        # FIX BUG 4: nn.Linear(1, 4) causava RuntimeError al primo forward.
        # EstimatorQNN con n_qubits=4 osservabili produce output (batch, 4),
        # non (batch, 1). Il layer deve essere Linear(n_qubits, 4).
        self.head = nn.Linear(n_qubits, 4)

    def forward(self, x):
        return self.head(self.qnn_layer(x))


def train_vqc(X, y, d, epochs=10):
    model = FewShotVQC(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    return model