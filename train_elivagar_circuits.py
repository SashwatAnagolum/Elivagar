import argparse
import os
import numpy as np
import pennylane as qml

from elivagar.training.train_circuits_from_predictors import train_elivagar_circuits
from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure in each circuit')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs to train the generated circuits for')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')
    parser.add_argument('--device_name', default=None, help='the device noise model the CNR scores were computed using')
    parser.add_argument('--num_runs_per_circ', default=5, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default='angle', help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, help='the number of times to re-encode the data')
    parser.add_argument('--noise_importance', type=float, default=0.5, help='the relative importance of circuit CNR scores')
    parser.add_argument('--num_data_for_rep_cap', default=None, type=int, help='number of data points used to compute representational capacity')
    parser.add_argument('--num_params_for_rep_cap', default=None, type=int, help='number of parameter vectors used to compute representational capacity')
    parser.add_argument('--num_cdcs', default=None, type=int, help='number of cdcs used to compute CNR scores')
    parser.add_argument('--num_circs', default=None, type=int, help='number of circuits evaluated')
    parser.add_argument('--num_candidates_per_circ', default=100, type=int,help='number of candidate circuits to evaluate for every circuit trained')
    parser.add_argument('--save_dir', default=None, help='folder to save the generated circuits, trained models, and training results in')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')    

    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    if args.num_data_for_rep_cap is None:
        raise ValueError('Please enter the number of data used to compute representational capacity scores.')
        
    if args.num_params_for_rep_cap is None:
        raise ValueError('Please enter the number of parameter vectors used to compute representational capacity scores.')

    if args.num_cdcs is None:
        raise ValueError('Please enter the number of clifford replicas used to compute CNR scores.')
        
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
        
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']   
    
    train_elivagar_circuits(args.circs_dir, args.dataset, args.encoding_type, 
                            args.num_data_reps, args.device_name, args.num_epochs, args.batch_size,
                            args.num_qubits, args.num_meas_qubits,
                            args.num_data_for_rep_cap, args.num_params_for_rep_cap,
                            args.num_cdcs, args.num_candidates_per_circ,
                            args.num_circs, args.num_runs_per_circ, args.noise_importance,
                            args.save_dir, args.learning_rate)
    
if __name__ == '__main__':
    main()
