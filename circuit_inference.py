import argparse
import os
import numpy as np
import torch
import qiskit.providers.aer.noise as noise

from qiskit import IBMQ

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams
from elivagar.utils.datasets import TorchDataset
from elivagar.utils.create_noise_models import get_noise_model
from elivagar.training.train_circ_np import TQMseLoss
from elivagar.inference.noise_model import run_noisy_inference_for_tq_circuits_qiskit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_circs', default=25, help='number of circuits to perform inference for')
    parser.add_argument('--num_runs_per_circ', default=5, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, help='the number of times to re-encode the data')
    parser.add_argument('--device_name', default=None, help='the name of the device to use a noise model of')
    parser.add_argument('--num_shots', default=1024, help='the number of shots executed to compute expectation values')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--human_design', action='store_true', help='whether the circuit is human designed or not')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_test_samples', default=None, help='the number of test samples to evaluate circuits on')
    parser.add_argument('--compute_noiseless', action='store_true', help='whether to compute noiseless outputs as a sanity check')
    parser.add_argument('--use_qubit_mapping', default=False, help='whether to use qubit mapping or not')
    
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
        
    if args.human_design:
        args.num_circs = 1
        
    num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']    
    meas_qubits = [i for i in range(num_meas_qubits)]
    
    noise_model, basis_gates, coupling_map = get_noise_model(args.device_name)

    run_noisy_inference_for_tq_circuits_qiskit(args.circs_dir, args.circ_prefix, args.num_circs, args.num_runs_per_circ,
                                               args.num_qubits, meas_qubits, args.device_name, noise_model, basis_gates,
                                               coupling_map, args.dataset, args.encoding_type, args.num_data_reps,
                                               args.num_test_samples, args.human_design, args.compute_noiseless,
                                               args.use_qubit_mapping, args.num_shots)    
    
if __name__ == '__main__':
    main()
