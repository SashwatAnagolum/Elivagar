import numpy as np
import os
import pennylane as qml

from qiskit.providers.aer import AerSimulator
from braket.devices import LocalSimulator 

from elivagar.utils.braket_devices import get_braket_device
from elivagar.utils.create_noise_models import get_real_backend_dev, noisy_dev_from_backend, get_noise_model
from elivagar.circuits.create_circuit import create_gate_circ, create_qiskit_circ, create_braket_gate_circ
from elivagar.circuits.arbitrary import get_circ_params
from elivagar.utils.datasets import load_dataset
from elivagar.inference.noise_model import run_qiskit_circ
from elivagar.circuits.run_circuit import run_braket_circuit


def convert_to_cdc(circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Convert a given circuit to a randomly generated Clifford replica.
    """
    clifford_gates = [['h', 's', 'z', 'x', 'y'], ['cx']]
    
    cdc_gates = []
    cdc_gate_params = gate_params
    cdc_inputs_bounds = [0 for i in range(len(circ_gates) + 1)]
    cdc_weights_bounds = [0 for i in range(len(circ_gates) + 1)]

    for i in range(len(circ_gates)):
        cdc_gates.append(np.random.choice(clifford_gates[len(gate_params[i]) - 1]))
    
    return cdc_gates, cdc_gate_params, cdc_inputs_bounds, cdc_weights_bounds


def compute_noise_metric(circ_gates, gate_params, inputs_bounds, weights_bounds, num_qubits, noiseless_dev, noisy_dev,
                         qubit_mapping, num_cdcs=32, num_shots=1024, qiskit=False, coupling_map=None, basis_gates=None,
                         use_real_backend=False):
    """
    Compute the Clifford noise resilience for a circuit.
    """
    circ_list = [convert_to_cdc(circ_gates, gate_params, inputs_bounds, weights_bounds) for i in range(num_cdcs)] 
    meas_qubits = [i for i in range(num_qubits)]

    circ_noisy_dist = np.zeros((num_cdcs, 2 ** num_qubits))
    circ_noiseless_dist = np.zeros((num_cdcs, 2 ** num_qubits))
    
    if qiskit:        
        for i in range(num_cdcs):
            circ = create_qiskit_circ(*circ_list[i], meas_qubits, num_qubits)([], [])
            
            circ_noiseless_dist[i] = run_qiskit_circ(
                circ, noiseless_dev, num_qubits, num_shots, mode='probs'
            )
            
            circ_noisy_dist[i] = run_qiskit_circ(
                circ, noisy_dev, num_qubits, num_shots, mode='probs',
                transpile_circ=True, basis_gates=basis_gates,
                coupling_map=coupling_map, qubit_mapping=qubit_mapping
            )
    else:
        if use_real_backend:
            for i in range(num_cdcs):
                noiseless_results = run_braket_circuit(
                    *circ_list[i], qubit_mapping, [], [],
                    noiseless_dev, num_shots, False, False, True
                )

                noisy_results = run_braket_circuit(
                    *circ_list[i], qubit_mapping, [], [],
                    noisy_dev, num_shots, False, False, True
                )

                circ_noisy_dist[i] = noisy_results
                circ_noiseless_dist[i] = noiseless_results
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


def compute_clifford_nr_for_circuit(circ_dir, noisy_dev, noiseless_dev, device_name, num_qubits, num_cdcs, num_shots, compute_actual_fidelity=False, num_trial_params=128,
                                    x_train=None, use_qiskit=False, basis_gates=None, coupling_map=None, use_qubit_mapping=True, index_mapping=None, use_real_backend=False):
    """
    Compute Clifford Noise Robustness for a circuit using the noiseless and noisy devices passed in. num_cdc Clifford decoys will be used
    to compute the Clifford Noise Robustness.
    """
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    
    if use_qubit_mapping:
        qubit_mapping = np.genfromtxt(os.path.join(circ_dir, 'qubit_mapping.txt'))
    else:
        qubit_mapping = [i for i in range(num_qubits)]
        
    if index_mapping:
        qubit_mapping = [index_mapping[i] for i in qubit_mapping]
    
    noise_metric_dir = os.path.join(circ_dir, f'cnr_{num_cdcs}')        

    if not os.path.exists(noise_metric_dir):
        os.mkdir(noise_metric_dir)

    device_noise_metric_dir = os.path.join(noise_metric_dir, device_name)

    if not os.path.exists(device_noise_metric_dir):
        os.mkdir(device_noise_metric_dir)
        
    clifford_fid = 1 - compute_noise_metric(
        circ_gates, gate_params, inputs_bounds, weights_bounds, num_qubits,
        noiseless_dev, noisy_dev, qubit_mapping, num_cdcs=num_cdcs,
        num_shots=num_shots, qiskit=use_qiskit, coupling_map=coupling_map,
        basis_gates=basis_gates, use_real_backend=use_real_backend
    )
    
    np.savetxt(device_noise_metric_dir + '/cnr.txt', [clifford_fid])
    
    if compute_actual_fidelity:
        params = np.random.sample((num_trial_params, weights_bounds[-1])) * 2 * np.pi

        if x_train is not None:
            batch_data = x_train[np.random.choice(len(x_train), num_trial_params, False)]
        else:
            batch_data = np.random.sample((num_trial_params, inputs_bounds[-1])) * np.pi
        
        if not use_qiskit:
            meas_qubits = [i for i in range(num_qubits)]

            noisy_circ = create_batched_gate_circ(noisy_dev, circ_gates, gate_params, inputs_bounds, weights_bounds, meas_qubits, 'probs') 
            noiseless_circ = create_batched_gate_circ(noiseless_dev, circ_gates, gate_params, inputs_bounds, weights_bounds, meas_qubits, 'probs') 

            noiseless_res_raw = np.array(noiseless_circ(batch_data, params, shots=num_shots))
            noisy_res_raw = np.array(noisy_circ(batch_data, params, shots=num_shots))
            actual_fid = 1 - np.mean(np.sum(0.5 * np.abs(noiseless_res_raw - noisy_res_raw), 1))

            np.savetxt(device_noise_metric_dir + '/actual_fidelity.txt', [actual_fid])  
        else:
            circ_creator = create_qiskit_circ(circ_gates, gate_params, inputs_bounds,
                                                    weights_bounds, [i for i in range(num_qubits)], num_qubits)
            
            actual_tvds = np.zeros(num_trial_params)
            
            for i in range(num_trial_params):
                curr_circ = circ_creator(batch_data[i], params[i])
                
                probs_noiseless = run_qiskit_circ(curr_circ, noiseless_dev, num_qubits, mode='probs')
                probs_noisy = run_qiskit_circ(curr_circ, noisy_dev, num_qubits, mode='probs', transpile_circ=True, basis_gates=basis_gates, coupling_map=coupling_map)
                
                actual_tvds[i] = 0.5 * np.sum(np.abs(probs_noiseless - probs_noisy))
                
            actual_fid = 1 - np.mean(actual_tvds)
    
        return clifford_fid, actual_fid
    else:
        return clifford_fid
    
                                                                             
def compute_clifford_nr_for_circuits(circ_dir, num_circs, device_name,
    num_qubits, num_cdcs, num_shots, compute_actual_fidelity=False,
    num_trial_params=128, dataset=None, embed_type=None, data_reps=None,
    noise_model=None, use_qiskit=False, basis_gates=None, coupling_map=None,
    use_qubit_mapping=False, save=False, index_mapping=None,
    circ_index_offset=0, dataset_file_extension='txt', use_real_backend=False):
    """
    Compute the Clifford Noise Robustness for a group of circuits stored in the same directory.
    """
    if dataset is not None:
        x_train, _, _, __ = load_dataset(dataset, embed_type, data_reps, dataset_file_extension)
    else:
        x_train = None
        
    if use_qiskit:
        noiseless_dev = AerSimulator()
        noise_model = None
        basis_gates = None
        coupling_map = None

        if noise_model is None and ('ibm' in device_name):
            noise_model, basis_gates, coupling_map = get_noise_model(device_name)
            
        noisy_dev = AerSimulator(noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map)
        # noisy_dev = AerSimulator(noise_model=noise_model, basis_gates=['cx', 'rz', 'x', 's', 'id', 'h', 'z', 'y'], coupling_map=coupling_map)
    else:
        if use_real_backend:
            noisy_dev = get_braket_device(device_name)
            noiseless_dev = LocalSimulator()
        else:
            raise ValueError('Cannot set both use_real_backend and use_qiskit to False!')

    noise_metric_scores = []
    
    for i in range(circ_index_offset, circ_index_offset + num_circs):
        curr_circ_dir = os.path.join(circ_dir, f'circ_{i + 1}')
        
        noise_metric_scores.append(
            compute_clifford_nr_for_circuit(
                curr_circ_dir, noisy_dev, noiseless_dev, device_name, num_qubits, num_cdcs,
                num_shots, compute_actual_fidelity, num_trial_params, x_train,
                use_qiskit, basis_gates, coupling_map, use_qubit_mapping, index_mapping,
                use_real_backend
            )
        )
        
        print(i, noise_metric_scores[-1])
     
    if save:
        if compute_actual_fidelity:
            np.savetxt(os.path.join(circ_dir, f'cnr_scores_{device_name}_{num_cdcs}.txt'), [i[1] for i in noise_metric_scores])
            np.savetxt(os.path.join(circ_dir, f'fid_scores_{device_name}_{num_trial_params}.txt'), [i[0] for i in noise_metric_scores])   
        else:
            np.savetxt(os.path.join(circ_dir,  f'cnr_scores_{device_name}_{num_cdcs}.txt'), noise_metric_scores)


    return noise_metric_scores