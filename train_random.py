import argparse
import os
import numpy as np
import torch

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams, gatesets
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.utils.datasets import TorchDataset
from elivagar.training.train_circ_np import TQMseLoss
from elivagar.circuits.arbitrary import generate_true_random_gate_circ


def generate_random_circuits(save_dir, num_circs, num_qubits, num_embeds, num_params, gateset=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(num_circs):
        curr_circ_dir = os.path.join(save_dir, f'circ_{i + 1}')

        if not os.path.exists(curr_circ_dir):
            os.mkdir(curr_circ_dir)

        if gateset is None or len(gateset[0]) > 0:
            ent_prob = np.random.sample()
        elif len(gateset[1]) == 0:
            ent_prob = 0
        else:
            ent_prob = 1

        cxz = np.random.sample()
        pauli = np.random.sample()

        circ_gates, gate_params, inputs_bounds, weights_bounds = generate_true_random_gate_circ(num_qubits,
            num_embeds, num_params, ent_prob, cxz * ent_prob, pauli * (1 - cxz) * ent_prob, gateset)

        np.savetxt(curr_circ_dir + '/gates.txt', circ_gates, fmt="%s")
        np.savetxt(curr_circ_dir + '/gate_params.txt', gate_params, fmt="%s")
        np.savetxt(curr_circ_dir + '/inputs_bounds.txt', inputs_bounds)
        np.savetxt(curr_circ_dir + '/weights_bounds.txt', weights_bounds)

        
def train_random_circuits(save_dir, num_circs, num_qubits, num_embeds, num_params, num_epochs, dataset_name,
                          num_data_reps, num_meas_qubits, batch_size, num_runs):
    loss_fn = TQMseLoss(num_meas_qubits, num_qubits)

    print(dataset_name, num_data_reps)
    
    train_data = TorchDataset(dataset_name, 'angle', num_data_reps, reshape_labels=True)
    test_data = TorchDataset(dataset_name, 'angle', num_data_reps, False, reshape_labels=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    for i in range(num_circs):
        print(f'circuit {i + 1}')
        curr_circ_dir = os.path.join(save_dir, f'circ_{i + 1}')

        train_tq_circ_and_save_results(curr_circ_dir, train_data_loader, test_data_loader,
                                       num_runs, num_epochs, num_qubits,
                                       num_meas_qubits, loss_fn)    
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_embeds', default=None, help='number of embedding gates in the generated circuits')
    parser.add_argument('--num_params', default=None, help='number of parameters in the generated circuits')
    parser.add_argument('--num_epochs', default=50, help='number of epochs to train the generated circuits for')
    parser.add_argument('--batch_size', default=32, help='batch size for training')
    parser.add_argument('--num_circs', default=25, help='number of random circuits to generate')
    parser.add_argument('--num_runs_per_circ', default=5, help='number of training runs for each circuit')
    parser.add_argument('--gateset', default=None, help='which gateset to use (rzx_xx / rxyz / all)')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, help='the number of times to re-encode the data')
    parser.add_argument('--save_dir', default='./',
                        help='folder to save the generated circuits, trained models, and training results in')

    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
    
    if args.num_embeds is None:
        args.num_embeds = curr_dataset_hyperparams['num_embeds']
        
    if args.num_params is None:
        args.num_params = curr_dataset_hyperparams['num_params']
        
    if args.gateset is not None:
        args.gateset = gatesets[args.gateset]
        
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
        
    num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']
            
    generate_random_circuits(args.save_dir, args.num_circs, args.num_qubits, args.num_embeds, args.num_params, args.gateset)
    
    train_random_circuits(args.save_dir, args.num_circs, args.num_qubits, args.num_embeds, args.num_params,
                          args.num_epochs, args.dataset, args.num_data_reps, num_meas_qubits,
                          args.batch_size, args.num_runs_per_circ)
    
if __name__ == '__main__':
    main()
