import argparse
import os
import numpy as np
import pennylane as qml

from elivagar.metric_computation.compute_clifford_nr import compute_clifford_nr_for_circuits
from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams
from elivagar.utils.create_noise_models import convert_braket_properties_to_qiskit_noise_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset the circuits will be used with')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_circs', default=2500, type=int, help='number of circuits to perform inference for')
    parser.add_argument('--circ_index_offset', default=0, type=int, help='circuit index offset')
    parser.add_argument('--num_cdcs', default=32, type=int, help='number of clifford replicas to use')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, type=int, help='the number of times to re-encode the data')
    parser.add_argument('--device_name', default=None, help='the name of the device to use a noise model of')
    parser.add_argument('--num_shots', default=1024, type=int, help='the number of shots executed to compute expectation values')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_trial_params', default=128, type=int, help='the number of parameters to use to compute actual fidelity')
    parser.add_argument('--compute_actual_fidelity', action='store_true', help='whether to compute noiseless outputs as a sanity check')
    parser.add_argument('--use_qubit_mapping', action='store_true', help='whether to use qubit mapping or not')
    parser.add_argument('--save_cnr_scores', action='store_true', help='whether to store the computed CNR scores or not')
    parser.add_argument('--device_properties_file', type=str, default=None, help='the path to the pickled device properties file')
    parser.add_argument('--dataset_file_extension', default='txt', type=str, help='extension for the dataset files')
    parser.add_argument('--symmetric', action='store_true', help='whether the device edges are symmetric or not.')
    
    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    if args.device_name is None:
        raise ValueError('Device name cannot be None, please enter a valid device name.')
    
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
    
    if args.device_properties_file is not None:
        noise_model, index_mapping, basis_gates, coupling_map = convert_braket_properties_to_qiskit_noise_model(
            args.device_properties_file, args.symmetric
        )
    else:
        noise_model = None
        index_mapping = None
    
    compute_clifford_nr_for_circuits(
        args.circs_dir, args.num_circs, args.device_name, args.num_qubits,
        args.num_cdcs, args.num_shots, args.compute_actual_fidelity, args.num_trial_params,
        args.dataset, args.encoding_type, args.num_data_reps,
        noise_model, True, None, None, args.use_qubit_mapping, args.save_cnr_scores,  
        index_mapping, args.circ_index_offset, args.dataset_file_extension
    )
    
if __name__ == '__main__':
    main()
