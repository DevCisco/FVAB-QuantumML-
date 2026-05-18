import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import real_amplitudes
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

from qiskit_aer.primitives import EstimatorV2 as Estimator


class FewShotVQC(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        n_qubits = 4

        qc = QuantumCircuit(n_qubits)
        x = ParameterVector('x', d_latent)

        num_cycles = (d_latent + n_qubits - 1) // n_qubits
        
        weight_params_all = []
        feat_idx = 0
        for l in range(num_cycles):
            for i in range(n_qubits):
                if feat_idx < d_latent:
                    qc.ry(x[feat_idx], i)
                    feat_idx += 1

            ansatz_template = real_amplitudes(n_qubits, reps=1, entanglement='linear')
            n_w = ansatz_template.num_parameters
            w = ParameterVector(f'theta_l{l}', n_w)
            weight_params_all.extend(list(w))
            param_map = {ansatz_template.parameters[i]: w[i] for i in range(n_w)}
            qc.compose(ansatz_template.assign_parameters(param_map), inplace=True)

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