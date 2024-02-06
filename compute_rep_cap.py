import argparse
import os
import numpy as np
import pennylane as qml

from elivagar.metric_computation.compute_rep_cap import compute_rep_cap_for_circuits
from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', type=int, default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure in each circuit')
    parser.add_argument('--num_circs', type=int, default=2500, help='number of circuits to perform inference for')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', type=int, default=None, help='the number of times to re-encode the data')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--save_matrices', action='store_true', help='whether to save matrices or not')
    parser.add_argument('--dataset_file_extension', default='txt', type=str, help='extension for the dataset files')
    parser.add_argument('--num_param_samples', type=int, default=32,
                        help='number of parameter vectors to average over')

    parser.add_argument('--num_samples_per_class', type=int, default=16,
                        help='number of samples to use per class in the dataset')
    
    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
    
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
        
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']   

    num_classes = curr_dataset_hyperparams['num_classes']
    
    compute_rep_cap_for_circuits(args.circs_dir, args.num_circs, args.circ_prefix, args.num_qubits, 
                                 args.num_meas_qubits, args.dataset, num_classes,
                                 args.num_samples_per_class,
                                 args.num_param_samples, args.encoding_type, 
                                 args.num_data_reps, args.save_matrices,
                                 args.dataset_file_extension
                                )
    
if __name__ == '__main__':
    main()
