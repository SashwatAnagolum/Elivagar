import argparse
import os
import numpy as np
import torch

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams, gatesets
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.utils.datasets import TorchDataset
from elivagar.training.train_circ_np import TQMseLoss, TQCeLoss
from elivagar.circuits.arbitrary import generate_true_random_gate_circ


def add_angle_embedding(circ_gates, gate_targets, inputs_indices, weights_indices, num_qubits, num_embeds):
    layer_num = -1
    num_embeds_added = 0
    embed_gates = ['rx', 'ry', 'rz']
    
    extra_gates = []
    extra_gate_targets = []
    
    while num_embeds_added < num_embeds:
        if not num_embeds_added % num_qubits:
            layer_num += 1
            layer_num %= 3
            
        extra_gates.append(embed_gates[layer_num])
        extra_gate_targets.append([num_embeds_added % num_qubits])
        
        num_embeds_added += 1
        
    circ_gates = extra_gates + circ_gates
    gate_targets = extra_gate_targets + gate_targets
    inputs_indices = [i for i in range(num_embeds + 1)] + [num_embeds for i in range(len(circ_gates))]
    weights_indices = [0 for i in range(num_embeds)] + weights_indices

    return circ_gates, gate_targets, inputs_indices, weights_indices
    
    
def add_iqp_embedding(circ_gates, gate_targets, inputs_indices, weights_indices, num_qubits, num_embeds):
    layer_num = -1
    num_iqp_layers = int(np.ceil(num_embeds / num_qubits))
    num_iqp_rzz_gates = (num_qubits * (num_qubits - 1)) // 2
    
    extra_gates = []
    extra_gate_targets = []
    extra_inputs_indices = [0]
    
    for i in range(num_iqp_layers):
        extra_gates += ['h' for j in range(num_qubits)]
        extra_gate_targets += [[j] for j in range(num_qubits)]
        extra_inputs_indices += [extra_inputs_indices[-1] for j in range(num_qubits)]
        
        extra_gates += ['rz' for i in range(num_qubits)]
        extra_gate_targets += [[i] for i in range(num_qubits)]
        extra_inputs_indices += [extra_inputs_indices[-1] + i + 1 for i in range(num_qubits)]

        extra_gates += ['rzz' for i in range(num_iqp_rzz_gates)]
        extra_gate_targets += [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        extra_inputs_indices += [extra_inputs_indices[-1] + i + 1 for i in range(num_iqp_rzz_gates)]
        
    circ_gates = extra_gates + circ_gates
    gate_targets = extra_gate_targets + gate_targets
    inputs_indices = extra_inputs_indices + [extra_inputs_indices[-1] for i in range(len(circ_gates))]
    weights_indices = [0 for i in range(len(extra_gates))] + weights_indices

    return circ_gates, gate_targets, inputs_indices, weights_indices    


def add_amp_embedding(circ_gates, gate_targets, inputs_indices, weights_indices, num_qubits, num_embeds):
    return (
        ['amp_enc'] + circ_gates, [[0]] + gate_targets, 
        [0, 2 ** num_qubits] + [2 ** num_qubits for i in range(len(circ_gates))],
        [0] + weights_indices
    )
    
    
def generate_random_circuits(save_dir, num_circs, num_qubits, num_embeds, num_params,
                             encoding_type, gateset=None, gate_param_nums=None,
                             fixed_embedding=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_circs):
        curr_circ_dir = os.path.join(save_dir, f'circ_{i + 1}')

        if not os.path.exists(curr_circ_dir):
            os.mkdir(curr_circ_dir)

        if gateset is None:
            ent_prob = 0.7 * np.random.sample()
        elif len(gateset[1]) == 0:
            ent_prob = 0
        elif len(gateset[0]) == 0:
            ent_prob = 1
        else:
            ent_prob = 0.7 * np.random.sample()

        cxz = np.random.sample()
        pauli = np.random.sample()

        if fixed_embedding:
            circ_gates, gate_params, inputs_bounds, weights_bounds = generate_true_random_gate_circ(
                num_qubits, 0, num_params, ent_prob,
                cxz * ent_prob, pauli * (1 - cxz) * ent_prob, gateset,
                gateset_param_nums=gate_param_nums
            )

            if encoding_type == 'angle':
                circ_gates, gate_params, inputs_bounds, weights_bounds = add_angle_embedding(
                    circ_gates, gate_params, inputs_bounds, weights_bounds,
                    num_qubits, num_embeds
                )
            elif encoding_type == 'iqp':
                circ_gates, gate_params, inputs_bounds, weights_bounds = add_iqp_embedding(
                    circ_gates, gate_params, inputs_bounds, weights_bounds,
                    num_qubits, num_embeds
                )
            elif encoding_type == 'amp':
                circ_gates, gate_params, inputs_bounds, weights_bounds = add_amp_embedding(
                    circ_gates, gate_params, inputs_bounds, weights_bounds,
                    num_qubits, num_embeds
                )
            else:
                raise ValueError('Encoding type not supported!')
        else:
            circ_gates, gate_params, inputs_bounds, weights_bounds = generate_true_random_gate_circ(
                num_qubits, num_embeds, num_params, ent_prob,
                cxz * ent_prob, pauli * (1 - cxz) * ent_prob, gateset,
                gateset_param_nums=gate_param_nums
            )            
            
        np.savetxt(os.path.join(curr_circ_dir, 'gates.txt'), circ_gates, fmt="%s")
        np.savetxt(os.path.join(curr_circ_dir, 'gate_params.txt'), gate_params, fmt="%s")
        np.savetxt(os.path.join(curr_circ_dir, 'inputs_bounds.txt'), inputs_bounds)
        np.savetxt(os.path.join(curr_circ_dir, 'weights_bounds.txt'), weights_bounds)

        
def train_random_circuits(save_dir, num_circs, num_qubits, num_epochs, dataset_name,
                          num_data_reps, num_meas_qubits, batch_size, num_runs,
                          learning_rate, encoding_type, file_type, use_classification_loss):
    if use_classification_loss:
        loss_fn = TQCeLoss(num_meas_qubits, num_qubits)
    else:
        loss_fn = TQMseLoss(num_meas_qubits, num_qubits)
    
    if encoding_type == 'amp':
        encoding_type = 'angle'
    
    train_data = TorchDataset(
        dataset_name, encoding_type, num_data_reps,
        True, True, file_type
    )

    test_data = TorchDataset(
        dataset_name, encoding_type, num_data_reps,
        False, True, file_type
    )

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    for i in range(num_circs):
        print(f'\n\n\nTraining Circuit {i + 1}\n\n\n')
        curr_circ_dir = os.path.join(save_dir, f'circ_{i + 1}')

        train_tq_circ_and_save_results(
            curr_circ_dir, train_data_loader, test_data_loader,
            num_runs, num_epochs, num_qubits,
            num_meas_qubits, loss_fn, learning_rate=learning_rate,
        )    
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--num_qubits', type=int, default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_embeds', type=int, default=None, help='number of embedding gates in the generated circuits')
    parser.add_argument('--num_params', type=int, default=None, help='number of parameters in the generated circuits')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train the generated circuits for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--num_circs', type=int, default=25, help='number of random circuits to generate')
    parser.add_argument('--num_runs_per_circ', type=int, default=5, help='number of training runs for each circuit')
    parser.add_argument('--gateset', default=None, help='which gateset to use (rzx_xx / rxyz / all)')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', type=int, default=None, help='the number of times to re-encode the data')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--file_type', type=str, default='txt', help='file extension for the dataset files')
    parser.add_argument('--fixed_embedding', action='store_true', help='use a fixed embedding or not')
    parser.add_argument('--use_classification_loss', action='store_true', help='whether to use the classification loss or not')
    parser.add_argument('--save_dir', type=str, default='./',
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
        gateset, gate_param_nums = gatesets[args.gateset]
    else:
        gateset = None
        gate_param_nums = None
    
    if args.num_data_reps is None:
        args.num_data_reps = curr_dataset_hyperparams['num_data_reps']
    
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']
            
    generate_random_circuits(
        args.save_dir, args.num_circs, args.num_qubits, args.num_embeds,
        args.num_params, args.encoding_type, gateset, gate_param_nums,
        args.fixed_embedding
    )
    
    if args.encoding_type == 'amp':
        num_data_reps = int(np.ceil((2 ** args.num_qubits) / (args.num_embeds / args.num_data_reps)))
    
    train_random_circuits(
        args.save_dir, args.num_circs, args.num_qubits,
        args.num_epochs, args.dataset, args.num_data_reps,
        args.num_meas_qubits, args.batch_size, args.num_runs_per_circ,
        args.learning_rate, args.encoding_type, args.file_type,
        args.use_classification_loss
    )
    
if __name__ == '__main__':
    main()
