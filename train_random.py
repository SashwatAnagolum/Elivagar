import argparse
import os
import numpy as np

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams

from qml_utils.train_tq_circ_and_save import train_tq_circ_and_save_results
from qml_utils.datasets import TorchDataset
from qml_utils.train_circ_np import TQMseLoss
from qml_utils.create_gate_circs import generate_true_random_gate_circ

def generate_random_circuits(save_dir, num_circs, num_qubits, num_embeds, num_params, gateset=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(num_circs):
        curr_circ_dir = os.path.join(save_dir, f'circ_{i + 1}')

        if not os.path.exists(curr_circ_dir):
            os.mkdir(curr_circ_dir)

        if gateset is None or len(gateset[0]) > 0:
            ent_prob = np.random.sample()
        else:
            ent_prob = 1

        cxz = np.random.sample()
        pauli = np.random.sample()

        circ_gates, gate_params, inputs_bounds, weights_bounds = generate_true_random_gate_circ(num_qubits,
            num_embeds, num_params, ent_prob=ent_prob, cxz_prob=cxz * ent_prob,
            pauli_prob=pauli * (1 - cxz) * ent_prob)

        np.savetxt(curr_circ_dir + '/gates.txt', circ_gates, fmt="%s")
        np.savetxt(curr_circ_dir + '/gate_params.txt', gate_params, fmt="%s")
        np.savetxt(curr_circ_dir + '/inputs_bounds.txt', inputs_bounds)
        np.savetxt(curr_circ_dir + '/weights_bounds.txt', weights_bounds)

        
def train_random_circuits(save_dir, num_circs, num_qubits, num_embeds, num_params, num_epochs, dataset):
    
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_embeds', default=None, help='number of embedding gates in the generated circuits')
    parser.add_argument('--num_params', default=None, help='number of parameters in the generated circuits')
    parser.add_argument('--num_epochs', default=50, help='number of epochs to train the generated circuits for')
    parser.add_argument('--num_circs', default=25, help='number of random circuits to generate')
    parser.add_argument('--save_dir', default='./',
                        help='folder to save the generated circuits, trained models, and training results in')

    args = parser.parse_args()
    
    options = dict()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
        
    if args.num_qubits is None:
        args.num_qubits = dataset_circuit_hyperparams[args.dataset]['num_qubits']
    
    if args.num_embeds is None:
        args.num_embeds = dataset_circuit_hyperparams[args.dataset]['num_embeds']
        
    if args.num_params is None:
        args.num_params = dataset_circuit_hyperparams[args.dataset]['num_params']
            
    generate_random_circuits(args.save_dir, args.num_circs, args.num_qubits, args.num_embeds, args.num_params)
    
    train_random_circuits(args.save_dir, args.num_circs, args.num_qubits, args.num_embeds, args.num_params,
                          args.num_epochs, args.dataset)
    
if __name__ == '__main__':
    main()