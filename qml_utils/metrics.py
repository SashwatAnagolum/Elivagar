import numpy as np
import pennylane as qml

# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, RuntimeOptions
from create_gate_circs import create_gate_circ, create_qiskit_circ

from elivagar.inference.qiskit_circs import run_qiskit_circ

def compute_reduced_similarity(circ, params, data, single_qubit=False):
    """
    Compute the similarity between the reduced density matrices obtained from a circuit's outputs.
    """
    num_data = len(data)
    traces = []
    circ_fids = np.zeros((num_data, num_data))
    
    if not single_qubit:
        circ_dms = circ(data, params)

        for i in range(num_data):
            traces.append(np.trace(circ_dms[i]))

        for s1 in range(num_data):
            for s2 in range(s1 + 1, num_data):
                trace_1 = traces[s1]
                trace_2 = traces[s2]
                fid_trace = np.trace(np.matmul(circ_dms[s1], circ_dms[s2]))

                curr_score = ((fid_trace) ** 2 / (trace_1 * trace_2)).real

                circ_fids[s1, s2] = curr_score
                circ_fids[s2, s1] = curr_score  

        circ_fids += np.eye(num_data)
    else:
        num_meas_qubits = len(circ)
        circ_dms = [circ[i](data, params) for i in range(num_meas_qubits)]
        all_fids = np.zeros((num_meas_qubits, num_data, num_data))
        
        for i in range(num_meas_qubits):
            traces = []
            
            for j in range(num_data):
                traces.append(np.trace(circ_dms[i][j]))

            for s1 in range(num_data):
                for s2 in range(s1 + 1, num_data):
                    trace_1 = traces[s1]
                    trace_2 = traces[s2]
                    fid_trace = np.trace(np.matmul(circ_dms[i][s1], circ_dms[i][s2]))

                    curr_score = ((fid_trace) ** 2 / (trace_1 * trace_2)).real

                    all_fids[i][s1, s2] = curr_score
                    all_fids[i][s2, s1] = curr_score  

            all_fids[i] += np.eye(num_data) 
            
        circ_fids = np.mean(all_fids, 0)
            
    return circ_fids


def convert_to_cdc(circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Convert a given circuit to a randomly generated Clifford replica.
    """
    clifford_gates = [['h', 's', 'z', 'x', 'y'], ['cx']]
    
    sdc_gates = []
    sdc_gate_params = gate_params
    sdc_inputs_bounds = [0 for i in range(len(circ_gates) + 1)]
    sdc_weights_bounds = [0 for i in range(len(circ_gates) + 1)]

    for i in range(len(circ_gates)):
        sdc_gates.append(np.random.choice(clifford_gates[len(gate_params[i]) - 1]))
    
    return sdc_gates, sdc_gate_params, sdc_inputs_bounds, sdc_weights_bounds


def compute_noise_metric(circ_gates, gate_params, inputs_bounds, weights_bounds, num_qubits, noiseless_dev, noisy_dev,
                         num_cdcs=32, num_shots=1024, qiskit=False, coupling_map=None, basis_gates=None):
    """
    Compute the Clifford noise resilience for a circuit.
    """
    circ_list = [convert_to_cdc(circ_gates, gate_params, inputs_bounds, weights_bounds) for i in range(num_cdcs)]
    
    bitstrings = [str(bin(i))[2:] for i in range(2 ** num_qubits)]
    bitstrings = ['0' * (num_qubits - len(i)) + i for i in bitstrings]
    
    meas_qubits = [i for i in range(num_qubits)]

    circ_noisy_dist = np.zeros((num_cdcs, 2 ** num_qubits))
    circ_noiseless_dist = np.zeros((num_cdcs, 2 ** num_qubits))
    
    if qiskit:        
        cdc_circ_list = []
        
        for i in range(num_cdcs):
            circ = create_qiskit_circ(*circ_list[i], meas_qubits, num_qubits)([], [])
            
            circ_noiseless_dist[i] = run_qiskit_circ(circ, noiseless_dev, num_qubits, num_shots, mode='probs')
            circ_noisy_dist[i] = run_qiskit_circ(circ, noisy_dev, num_qubits, num_shots, mode='probs', transpile_circ=True, basis_gates=basis_gates, coupling_map=coupling_map)
    else:
        for i in range(num_cdcs):
            noiseless_circ = create_gate_circ(noiseless_dev, *circ_list[i], meas_qubits, 'probs')
            noisy_circ = create_gate_circ(noisy_dev, *circ_list[i], meas_qubits, 'probs')

            noiseless_results = noiseless_circ([], [], shots=num_shots)
            noisy_results = noisy_circ([], [], shots=num_shots)

            circ_noisy_dist[i] = noisy_results
            circ_noiseless_dist[i] = noiseless_results
                
    tvds = 0.5 * np.sum(np.abs(circ_noiseless_dist - circ_noisy_dist), 1)
        
    return np.mean(tvds)