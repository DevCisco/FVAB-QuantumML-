from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import real_amplitudes
def create_vqc_circuit(n_qubits, d_latent, reps=1):
    circuit = QuantumCircuit(n_qubits)
    x = ParameterVector('x', d_latent)

    num_layers = (d_latent + n_qubits - 1) // n_qubits

    param_idx = 0

    for l in range(num_layers):
        # Layer di Encoding (RY)
        for i in range(n_qubits):
            if param_idx < d_latent:
                circuit.ry(x[param_idx], i)
                param_idx += 1

        # FIX A: la stima originale "2 * n_qubits * reps" è sbagliata per reps > 1.
        # RealAmplitudes con entanglement='linear' ha esattamente
        # n_qubits * (reps + 1) parametri. Usiamo l'attributo reale del circuito
        # per non dipendere da nessuna formula manuale.
        ansatz_template = real_amplitudes(n_qubits, reps=reps, entanglement='linear',
                                         insert_barriers=True)
        num_ansatz_params = ansatz_template.num_parameters  # valore esatto, non stimato

        ansatz_params = ParameterVector(f'theta_l{l}', num_ansatz_params)

        param_dict = {
            ansatz_template.parameters[i]: ansatz_params[i]
            for i in range(num_ansatz_params)
        }
        ansatz_bound = ansatz_template.assign_parameters(param_dict)
        ansatz_decomposed = ansatz_bound.decompose()
        circuit.compose(ansatz_decomposed, inplace=True)

    return circuit, x