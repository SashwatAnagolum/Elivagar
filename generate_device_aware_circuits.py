import argparse
import os
import numpy as np
import pickle as pkl

from qiskit import IBMQ

from elivagar.utils.dataset_circuit_hyperparams import dataset_circuit_hyperparams
from elivagar.circuits.device_aware import generate_device_aware_gate_circ

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default=None, help='which dataset the circuits will be used with')
    parser.add_argument('--num_qubits', type=int, default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_circs', type=int, default=2500, help='the number of circuits to generate')
    parser.add_argument('--num_embeds', type=int, default=None, help='the number of embedding gates in each circuit')
    parser.add_argument('--num_params', type=int, default=None, help='the number of parameters in each circuit')
    parser.add_argument('--device_name', type=str, default=None, help='the name of the IBM device the circuits will be run on')
    parser.add_argument('--param_focus', type=float, default=2, help='the amount of preference given to parametrized gates over nonparametrized ones')
    parser.add_argument('--save_dir', type=str, default='./', help='the folder to save the generated circuits in')
    parser.add_argument('--add_rotations', action='store_true', help='whether to add RX / RY / RZ gates to the basis gate list or not')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='the number of qubits to be measured')
    parser.add_argument('--num_trial_mappings', type=int, default=100, help='the number of trial qubit mappings to consider')
    parser.add_argument('--temp', type=float, default=0.1, help='the temperature of the softmax distributions to sample from')
    parser.add_argument('--braket_device_properties_path', type=str, default=None, help='the path to the brakte device properties pkl file')
    
    args = parser.parse_args()
    
    if args.target_dataset is None and (args.num_embeds is None or args.num_params is None or args.num_qubits is None):
        raise ValueError('Either provide the target dataset name or the number of qubits, parameters / embeds in each circuit.')
        
    if args.device_name is None and args.braket_device_properties_path is None:
        raise ValueError('Device name cannot be None, please enter a valid device name.')
    
    curr_dataset_hyperparams = dataset_circuit_hyperparams[args.target_dataset]
        
    if args.num_qubits is None:
        args.num_qubits = curr_dataset_hyperparams['num_qubits']
        
    if args.num_params is None:
        args.num_params = curr_dataset_hyperparams['num_params']
        
    if args.num_embeds is None:
        args.num_embeds = curr_dataset_hyperparams['num_embeds']
       
    if args.num_meas_qubits is None:
        args.num_meas_qubits = curr_dataset_hyperparams['num_meas_qubits']    
    
    if args.braket_device_properties_path is None:
        try:
            provider = IBMQ.enable_account(
                'f9be8ebe6cc0b5c9970ca5ae86acad18c1dfb3844ed12b381a458536fcbf46499d62dbb33da9a07627774441860c64ac44e76a6f27dc6f09bba7e0f2ce68e9ff')
        except:
            provider = IBMQ.load_account()

        backend = provider.get_backend(args.device_name)
        dev_properties = None
    else:
        backend = None
        dev_properties = pkl.load(open(args.braket_device_properties_path, 'rb'))
    
    for i in range(args.num_circs):
        curr_circ_dir = os.path.join(args.save_dir, f'circ_{i + 1}')
        
        if not os.path.exists(curr_circ_dir):
            os.makedirs(curr_circ_dir)
            
        ent_prob = np.random.sample()
        
        circ_gates, gate_params, inputs_bounds, weights_bounds, selected_mapping, meas_qubits = generate_device_aware_gate_circ(
            backend, args.num_qubits, args.num_embeds, args.num_params, ent_prob, args.add_rotations, args.param_focus,
            args.num_meas_qubits, args.num_trial_mappings, args.temp, braket_device_properties=dev_properties
        )
        
        np.savetxt(os.path.join(curr_circ_dir, 'gates.txt'), circ_gates, fmt="%s")
        np.savetxt(os.path.join(curr_circ_dir, 'gate_params.txt'), gate_params, fmt="%s")
        np.savetxt(os.path.join(curr_circ_dir, 'inputs_bounds.txt'), inputs_bounds)
        np.savetxt(os.path.join(curr_circ_dir, 'weights_bounds.txt'), weights_bounds)
        np.savetxt(os.path.join(curr_circ_dir, 'qubit_mapping.txt'), selected_mapping)
        np.savetxt(os.path.join(curr_circ_dir, 'meas_qubits.txt'), meas_qubits)
    
if __name__ == '__main__':
    main()
