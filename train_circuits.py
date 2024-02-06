import argparse
import os
import numpy as np
import pennylane as qml
import torch
import time

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams, gatesets
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.utils.datasets import TorchDataset
from elivagar.training.train_circ_np import TQMseLoss, TQCeLoss
from elivagar.circuits.arbitrary import generate_true_random_gate_circ

def train_circuits(circ_dir, circ_prefix, num_circs, num_qubits, num_epochs, dataset_name,
                   num_data_reps, num_meas_qubits, batch_size, num_runs,
                   learning_rate, encoding_type, contains_multiple, use_quantumnat,
                   noise_strength, save_dir, human_design, use_qtn_vqc, tt_input_size=None,
                   tt_ranks=None, tt_output_size=None, dataset_file_extension='txt',
                   loss_type='mse', use_gpu=False):
    if use_gpu:
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    if loss_type == 'mse':
        loss_fn = TQMseLoss(num_meas_qubits, num_qubits, device)
    elif loss_type == 'nll':
        loss_fn = TQCeLoss(num_meas_qubits, num_qubits, device)
    
    if encoding_type == 'amp':
        encoding_type = 'angle'
    
    if human_design and encoding_type == 'angle':
        encoding_type = 'angle_layer'
    
    train_data = TorchDataset(
        dataset_name, encoding_type, num_data_reps,
        reshape_labels=True, file_type=dataset_file_extension
    )

    test_data = TorchDataset(
        dataset_name, encoding_type, num_data_reps, False,
        reshape_labels=True, file_type=dataset_file_extension
    )

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    if contains_multiple:
        for i in range(num_circs):
            print(f'Circuit {i + 1}')
            curr_circ_dir = os.path.join(circ_dir, f'{circ_prefix}_{i + 1}')
            curr_save_dir = save_dir.format(f'{circ_prefix}_{i + 1}')
            
            train_tq_circ_and_save_results(
                curr_circ_dir, train_data_loader, test_data_loader,
                num_runs, num_epochs, num_qubits,
                num_meas_qubits, loss_fn, learning_rate=learning_rate,
                quantize=use_quantumnat, noise_strength=noise_strength,
                results_save_dir=curr_save_dir, use_qtn_vqc=use_qtn_vqc,
                tt_input_size=tt_input_size,
                tt_ranks=tt_ranks, tt_output_size=tt_output_size,
                device=device
            )    
    else:
        train_tq_circ_and_save_results(
            circ_dir, train_data_loader, test_data_loader,
            num_runs, num_epochs, num_qubits,
            num_meas_qubits, loss_fn, learning_rate=learning_rate,
            quantize=use_quantumnat, noise_strength=noise_strength,
            results_save_dir=save_dir, use_qtn_vqc=use_qtn_vqc,
            tt_input_size=tt_input_size,
            tt_ranks=tt_ranks, tt_output_size=tt_output_size,
            device=device
        )            

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure in each circuit')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs to train the generated circuits for')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')
    parser.add_argument('--contains_multiple', action='store_true', help='whether the folder contains multiple circuits or not')
    parser.add_argument('--num_runs_per_circ', default=5, type=int, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default='angle', help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, type=int, help='the number of times to re-encode the data')
    parser.add_argument('--num_circs', default=None, type=int, help='number of circuits evaluated')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')    
    parser.add_argument('--human_design', action='store_true', help='whether the circuit is a human designed circuit or not')    
    parser.add_argument('--use_quantumnat', action='store_true', help='whether to use noise injection during training or not.')
    parser.add_argument('--noise_strength', type=float, default=0.05, help='noise strength for quantumNAT.')
    parser.add_argument('--use_qtn_vqc', action='store_true', help='whether to use QTN-VQC for preprocessing or not.')
    parser.add_argument('--tt_ranks', type=int, nargs='+', default=None, help='ranks of the TT layer used in the model.')
    parser.add_argument('--tt_input_size', type=int, nargs='+', default=None, help='input size for the TT layer.')
    parser.add_argument('--tt_output_size', type=int, nargs='+', default=None, help='output size for the TT layer.')
    parser.add_argument('--save_dir', default=None, help='the folder to store the training results in')
    parser.add_argument('--dataset_file_extension', default='txt', type=str, help='extension for the dataset files')
    parser.add_argument('--loss', type=str, default='mse', help='the type of loss function to use for training.')
    parser.add_argument('--use_gpu', action='store_true', help='whether to use a GPU or not.')

    args = parser.parse_args()
    
    start = time.time()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    if args.use_qtn_vqc and not (args.tt_ranks and args.tt_input_size and args.tt_output_size):
        raise ValueError('Must provide values for tt_ranks, tt_input_size, and tt_output_size to use qtn_vqc!')
        
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
        
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']   
    
    if args.encoding_type == 'amp':
        args.num_data_reps = int(np.ceil((2 ** args.num_qubits) / (args.num_embeds / args.num_data_reps)))
    
    if args.tt_ranks:
        args.tt_ranks = [int(t) for t in args.tt_ranks]
        
    if args.tt_input_size:
        args.tt_input_size = [int(t) for t in args.tt_input_size]
        
    if args.tt_output_size:
        args.tt_output_size = [int(t) for t in args.tt_output_size]
    
    train_circuits(
        args.circs_dir, args.circ_prefix, args.num_circs, args.num_qubits,
        args.num_epochs, args.dataset, args.num_data_reps,
        args.num_meas_qubits, args.batch_size, args.num_runs_per_circ,
        args.learning_rate, args.encoding_type, args.contains_multiple,
        args.use_quantumnat, args.noise_strength,
        args.save_dir, args.human_design, args.use_qtn_vqc,
        args.tt_input_size, args.tt_ranks, args.tt_output_size,
        args.dataset_file_extension, args.loss, args.use_gpu
    )
    
    print('Training for', args.dataset, ":", time.time() - start)
    
if __name__ == '__main__':
    main()
