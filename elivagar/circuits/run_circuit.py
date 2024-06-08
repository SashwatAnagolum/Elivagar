import numpy as np
import time

from .create_circuit import create_braket_gate_circ


def run_braket_circuit(circ_gates, gate_targets, inputs_indices,
                       weights_indices, sel_qubits, circ_data,
                       circ_params, device, num_shots=2048, verbatim=True,
                       qubit_mapping=True, simulator=False):
    """
    Run a Braket circuit, either on a real device or a simulator.
    """
    circ_num_qubits = len(sel_qubits) 
    
    if qubit_mapping:
        mapped_qubits = sel_qubits
    else:
        mapped_qubits = [i for i in range(len(sel_qubits))]
    
    circ = create_braket_gate_circ(
        circ_gates, gate_targets, inputs_indices,
        weights_indices, circ_num_qubits, mapped_qubits,
        verbatim_box=verbatim
    )(circ_data, circ_params)
            
    circ.probability(target=[mapped_qubits[i] for i in range(circ_num_qubits)])
        
    if not simulator:
        try:
            task = device.run(circ, shots=num_shots, disable_qubit_rewiring=qubit_mapping)
            
            start = time.time()

            while task.state() not in ['COMPLETED', 'FAILED']:
                if (time.time() - start) > 5:
                    print(task.state())

                    start = time.time()

            if task.state() == 'COMPLETED':
                probs = task.result().result_types[0].value
            else:
                print(task.state())
                probs = np.zeros(2 ** circ_num_qubits)
        except Exception as e:
            print(e)
            probs = np.zeros(2 ** circ_num_qubits)
    else:      
        for qubit in range(circ_num_qubits):
            circ.i(qubit)
        
        probs = device.run(circ, shots=num_shots).result().result_types[0].value
        
    return probs
