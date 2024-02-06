import argparse
import os
import numpy as np
import torch

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams, gatesets
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.utils.datasets import TorchDataset
from elivagar.training.train_circ_np import TQMseLoss
from elivagar.circuits.human_design import convert_human_design_circ_to_gate_circ


def train_human_designed_circuit(save_dir, num_qubits, num_epochs, dataset_name,
                                 encoding_type, num_data_reps, num_meas_qubits,
                                 batch_size, num_runs, learning_rate):
    loss_fn = TQMseLoss(num_meas_qubits, num_qubits)
    
    if encoding_type == 'angle':
        encoding_type = 'angle_layer'
        
    if encoding_type == 'amp':
        encoding_type = 'angle'
    
    print(batch_size, encoding_type, dataset_name)
    
    train_data = TorchDataset(dataset_name, encoding_type, num_data_reps, reshape_labels=True)
    test_data = TorchDataset(dataset_name, encoding_type, num_data_reps, False, reshape_labels=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train_tq_circ_and_save_results(save_dir, train_data_loader, test_data_loader,
                                       num_runs, num_epochs, num_qubits,
                                       num_meas_qubits, loss_fn, learning_rate=learning_rate)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuit on')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuit')
    parser.add_argument('--num_embed_layers', default=None, type=int, help='number of embedding layers in the generated circuit')
    parser.add_argument('--num_var_layers', default=None, type=int, help='number of variational layers in the generated circuit')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs to train the generated circuit for')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')
    parser.add_argument('--num_runs_per_circ', default=5, type=int, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--var_layer_type', default='basic', help='the layer type to use for the variational layers')
    parser.add_argument('--num_data_reps', default=None, type=int, help='the number of times to re-encode the data')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--save_dir', default='./',
                        help='folder to save the generated circuit, trained model, and training results in')

    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.encoding_type is None:
        raise ValueError('Encoding type must be specified! Please choose one of (angle / iqp / amp)')
    
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
    
    if args.num_embed_layers is None:
        args.num_embed_layers = curr_dataset_hyperparams['num_embed_layers'][args.encoding_type]
        
    if args.num_var_layers is None:
        args.num_var_layers = curr_dataset_hyperparams['num_var_layers']
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
    
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = convert_human_design_circ_to_gate_circ(
        args.num_qubits, args.encoding_type, args.var_layer_type,
        args.num_embed_layers, args.num_var_layers
    )
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    np.savetxt(os.path.join(args.save_dir, 'gates.txt'), circ_gates, fmt="%s")
    np.savetxt(os.path.join(args.save_dir, 'gate_params.txt'), gate_params, fmt="%s")
    np.savetxt(os.path.join(args.save_dir, 'inputs_bounds.txt'), inputs_bounds)
    np.savetxt(os.path.join(args.save_dir, 'weights_bounds.txt'), weights_bounds)
    
    train_human_designed_circuit(
        args.save_dir, args.num_qubits, args.num_epochs,
        args.dataset, args.encoding_type, args.num_data_reps,
        args.num_meas_qubits, args.batch_size, args.num_runs_per_circ,
        args.learning_rate
    )
    

if __name__ == '__main__':
    main()
