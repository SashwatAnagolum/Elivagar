import argparse
import os
import numpy as np
import torch
import qiskit.providers.aer.noise as noise
import time

from qiskit import IBMQ

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams
from elivagar.utils.datasets import TorchDataset
from elivagar.utils.create_noise_models import get_noise_model, convert_braket_properties_to_qiskit_noise_model
from elivagar.training.train_circ_np import TQMseLoss
from elivagar.inference.noise_model import run_noisy_inference_for_tq_circuits_qiskit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_circs', default=25, type=int, help='number of circuits to perform inference for')
    parser.add_argument('--num_runs_per_circ', default=5, type=int, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, type=int, help='the number of times to re-encode the data')
    parser.add_argument('--device_name', default=None, help='the name of the device to use a noise model of')
    parser.add_argument('--num_shots', default=1024, type=int, help='the number of shots executed to compute expectation values')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--human_design', action='store_true', help='whether the circuit is human designed or not')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_test_samples', default=None, type=int, help='the number of test samples to evaluate circuits on')
    parser.add_argument('--compute_noiseless', action='store_true', help='whether to compute noiseless outputs as a sanity check')
    parser.add_argument('--use_qubit_mapping', action='store_true', help='whether to use qubit mapping or not')
    parser.add_argument('--device_properties_file', default=None, help='path to the device properties file')
    parser.add_argument('--qubit_mapping_filename', default='qubit_mapping.txt', help='path to the qubit mapping file')
    parser.add_argument('--transpiler_opt_level', default=0, type=int, help='optimization level for the transpiler')
    parser.add_argument('--file_suffix', default='', help='file suffixes for the saved files')
    parser.add_argument('--results_save_dir', default='', help='folder to save inference results in')
    parser.add_argument('--pick_best', action='store_true', help='pick the best training runs for the circuit or not')
    parser.add_argument('--use_quantumnat', action='store_true', help='use quantumnat or not')
    parser.add_argument('--quantumnat_trained_dir', default='', help='the folder where the quantumnat-trained models are stored')
    
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
    
    start = time.time()
    
    if args.device_properties_file is None:
        noise_model, basis_gates, coupling_map = get_noise_model(args.device_name)
        index_map = None
    else:
        noise_model, index_map, basis_gates, coupling_map = convert_braket_properties_to_qiskit_noise_model(
            args.device_properties_file, args.device_name == 'rigetti_aspen_m_3'
        )
        
    run_noisy_inference_for_tq_circuits_qiskit(
        args.circs_dir, args.circ_prefix, args.num_circs, args.num_runs_per_circ,
        args.num_qubits, meas_qubits, args.device_name, noise_model, basis_gates,
        coupling_map, args.dataset, args.encoding_type, args.num_data_reps,
        args.num_test_samples, args.human_design, args.compute_noiseless,
        args.use_qubit_mapping, args.num_shots, args.qubit_mapping_filename,
        args.transpiler_opt_level, args.file_suffix, index_map, args.results_save_dir,
        args.pick_best, args.use_quantumnat, args.quantumnat_trained_dir
    )    
    
    print('Time taken for', args.dataset, time.time() - start)
    
if __name__ == '__main__':
    main()
