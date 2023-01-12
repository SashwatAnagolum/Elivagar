import qiskit
import numpy as np

from qiskit import execute
from qiskit.compiler import transpile

def run_qiskit_circ(circuit, dev, num_meas_qubits, num_shots=1024, transpile_circ=False, basis_gates=None, coupling_map=None, mode='exp'):
    if transpile_circ:
        circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map)
        
    outputs = execute(circuit, backend=dev, shots=num_shots).result().get_counts()
    
    if mode is 'exp':
        qubit_probs = np.zeros((len(meas_qubits), 2))

        for key in outputs.keys():
            for q in range(len(outputs[key])):
                qubit_probs[q, int(key[q])] += outputs[key]

        qubit_probs = qubit_probs[::-1, :] / shots_total
        ret_val = qubit_probs[:, 0] - qubit_probs[:, 1]
    elif mode is 'probs':
        ret_val = np.zeros(2 ** num_meas_qubits)
        
        for key in outputs.keys():
            key_bin = int(key[::-1], 2)
            
            ret_val[key_bin] = outputs[key]
            
        ret_val /= num_shots
    
    return ret_val
    