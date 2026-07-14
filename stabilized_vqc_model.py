import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from quantum_ansatz import create_vqc_circuit

class StabilizedVQC(nn.Module):
    def __init__(self, n_qubits, d_latent):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_latent = d_latent

        # 1. Creazione Circuito
        circuit, input_params = create_vqc_circuit(n_qubits, d_latent, reps=1)

        all_params = circuit.parameters

        input_params_sorted = sorted(
            [p for p in all_params if p.name.startswith("x[")],
            key=lambda p: int(p.name[2:-1])
        )
        weight_params_sorted = sorted(
            [p for p in all_params if not p.name.startswith("x[")],
            key=lambda p: p.name
        )

        # 2. Osservabili: un Z per qubit (little-endian, coerente con hybrid_engine)
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
               for i in range(n_qubits)]

        # 3. Definizione QNN
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=circuit,
            input_params=input_params_sorted,
            weight_params=weight_params_sorted,
            observables=obs,
            estimator=estimator
        )

        # 4. Connector PyTorch
        self.qnn_layer = TorchConnector(qnn)

        self.post_processing = nn.Linear(n_qubits, 4)

    def forward(self, x):
        x = self.qnn_layer(x)          # → (batch, n_qubits)
        return self.post_processing(x)  # → (batch, 4)