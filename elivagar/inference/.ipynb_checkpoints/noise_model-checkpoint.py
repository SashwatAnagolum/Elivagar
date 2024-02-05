import torch
import numpy as np
import pickle as pkl
import pennylane as qml
import os

from qiskit import execute
from qiskit.providers.aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.transpiler import Layout

from elivagar.circuits.arbitrary import get_circ_params
from elivagar.circuits.create_circuit import create_qiskit_circ, create_gate_circ
from elivagar.inference.quantumnat import quantize_and_normalize
from elivagar.inference.inference_metrics import mse_batch_loss, mse_vec_batch_loss, batch_acc, vec_batch_acc
from elivagar.utils.create_noise_models import noisy_dev_from_backend, get_noise_model
from elivagar.utils.datasets import load_dataset


def get_params_from_tq_model(model_dir, num_params):
    """
    Get the parameters stoerd in a torch quantum model.
    """
    model_data = torch.load(os.path.join(model_dir, 'model.pt'))
    model_params = np.zeros(num_params)
    
    for i in range(num_params):
        model_params[i] = model_data[f'var_gates.{i}.params'].numpy().item()
    
    return model_params


def run_qiskit_circ(circuit, dev, num_meas_qubits, num_shots=1024, transpile_circ=False, basis_gates=None,
                    coupling_map=None, mode='exp', opt_level=0, qubit_mapping=None):
    if qubit_mapping is None:
        qubit_mapping = [i for i in range(circuit.num_qubits)]
    
    if transpile_circ:
        circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map,
                           initial_layout=list(qubit_mapping), optimization_level=opt_level)
    
    outputs = execute(circuit, backend=dev, shots=num_shots).result().get_counts()
    
    if mode == 'exp':
        qubit_probs = np.zeros((num_meas_qubits, 2))

        for key in outputs.keys():
            for q in range(num_meas_qubits):
                qubit_probs[q, int(key[q])] += outputs[key]

        qubit_probs = qubit_probs[::-1, :] / num_shots
        ret_val = qubit_probs[:, 0] - qubit_probs[:, 1]
    elif mode == 'probs':
        ret_val = np.zeros(2 ** num_meas_qubits)
        
        for key in outputs.keys():
            key_bin = int(key[::-1], 2)
            
            ret_val[key_bin] = outputs[key]
            
        ret_val /= num_shots
    
    return ret_val


def tq_model_inference_on_noisy_sim_qiskit(circ_dir, device_name, num_runs, num_qubits, meas_qubits, noisy_dev, basis_gates,
                                           coupling_map, qubit_mapping, x_test, y_test, params=None, save=True,
                                           num_shots=1024, compute_noiseless=False, transpile_opt_level=0,
                                           file_suffix='', index_map=None, results_save_dir=None, pick_best=False,
                                           use_quantumnat=False, quantumnat_trained_dir=None):
    """
    Peform inference on test data x_test using a noisy simulator noisy_dev with a trained circuit stored in circ_dir.
    """
    num_meas_qubits = len(meas_qubits)
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    circ_creator = create_qiskit_circ(circ_gates, gate_params, inputs_bounds,
                                      weights_bounds, meas_qubits, num_qubits)
    
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    
    device_results_save_dir = os.path.join(results_save_dir, device_name)

    if not os.path.exists(device_results_save_dir):
        os.mkdir(device_results_save_dir) 
    
    losses_list = []
    accs_list = []
    
    pennylane_dev = qml.device('lightning.qubit', wires=num_qubits)
    curr_pennylane_circ = create_gate_circ(pennylane_dev, circ_gates, gate_params, inputs_bounds,
                                           weights_bounds, meas_qubits)
    
    if use_quantumnat:
        noiseless_accs = np.genfromtxt(os.path.join(circ_dir, quantumnat_trained_dir, 'accs.txt'))
    else:
        noiseless_accs = np.genfromtxt(os.path.join(circ_dir, 'accs.txt'))
        
    if len(noiseless_accs.shape) == 0:
        noiseless_accs = np.array([noiseless_accs])

    if index_map is not None and qubit_mapping is not None:
        qubit_mapping = [index_map[q] for q in qubit_mapping]

    if pick_best:
        ordering = np.argsort(-1 * noiseless_accs)
        chosen_run_indices = [(ordering[i], noiseless_accs[ordering[i]]) for i in range(num_runs)]
    else:
        chosen_run_indices = [(i, noiseless_accs[i]) for i in range(num_runs)]
    
    for run_index, run_noiseless_acc in chosen_run_indices:
        if use_quantumnat:
            curr_run_dir = os.path.join(circ_dir, quantumnat_trained_dir, f'run_{run_index + 1}')
        else:
            curr_run_dir = os.path.join(circ_dir, f'run_{run_index + 1}')
        
        if params is None:
            print('Fetching param values from', curr_run_dir)
            curr_params = get_params_from_tq_model(curr_run_dir, weights_bounds[-1])
        else:
            curr_params = params[run_index]

        val_exps = []
            
        circ_list = [circ_creator(sample, curr_params) for sample in x_test]
            
        for i in range(len(x_test)):            
            val_exps.append(
                run_qiskit_circ(
                    circ_list[i], noisy_dev, num_meas_qubits, num_shots,
                    True, basis_gates, coupling_map, 'exp', transpile_opt_level,
                    qubit_mapping
                )
            )
            
            if compute_noiseless:
                pennylane_outputs = curr_pennylane_circ(x_test[i], curr_params)

                print(f'Noiseless: {pennylane_outputs} | Noisy: {val_exps[-1]}')
            
        val_exps = np.array(val_exps)
        
        if use_quantumnat:
            val_exps = quantize_and_normalize(val_exps, curr_run_dir)

        if len(meas_qubits) > 1:
            val_exps = val_exps.reshape((len(x_test), len(meas_qubits)))
            acc = vec_batch_acc(val_exps, y_test)
            val_loss = mse_vec_batch_loss(val_exps, y_test)
        else:
            val_exps = val_exps.reshape(len(x_test))
            val_loss = mse_batch_loss(val_exps, y_test)
            acc = batch_acc(val_exps, y_test)

        losses_list.append(val_loss)
        accs_list.append(acc)
        
        print(f'Loss: {val_loss} | Acc: {acc} | Noiseless Acc: {run_noiseless_acc}')

    if save:
        np.savetxt(os.path.join(device_results_save_dir, f'val_losses_inference_only{file_suffix}.txt'), losses_list)
        np.savetxt(os.path.join(device_results_save_dir, f'accs_inference_only{file_suffix}.txt'), accs_list)  
        
    return losses_list, accs_list


def tq_model_inference_on_noisy_sim_pennylane(circ_dir, device_name, num_runs, meas_qubits, noisy_dev, x_test, y_test, params=None, save=True):
    """
    Peform inference on test data x_test using a noisy simulator noisy_dev with a trained circuit stored in circ_dir.
    """
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    circ = create_gate_circ(noisy_dev, circ_gates, gate_params, inputs_bounds,
                                                    weights_bounds, meas_qubits, 'exp')
    
    device_dir = os.path.join(circ_dir, device_name)

    if not os.path.exists(device_dir):
        os.mkdir(device_dir) 
    
    losses_list = []
    accs_list = []
    
    for j in range(num_runs):
        curr_run_dir = os.path.join(circ_dir, f'run_{j + 1}')
        
        if params is None:
            curr_params = get_params_from_tq_model(curr_run_dir, weights_bounds[-1])
        else:
            curr_params = params[j]

        val_exps = np.array([circ(x_test[i], curr_params) for i in range(len(x_test))])

        if len(meas_qubits) > 1:
            val_exps = val_exps.reshape((len(x_test), len(meas_qubits)))
            acc = vec_batch_acc(val_exps, y_test)
            val_loss = mse_vec_batch_loss(val_exps, y_test)
        else:
            val_exps = val_exps.reshape(len(x_test))
            val_loss = mse_batch_loss(val_exps, y_test)
            acc = batch_acc(val_exps, y_test)

        losses_list.append(val_loss)
        accs_list.append(acc)
        
        print(val_loss, acc)

    if save:
        np.savetxt(device_dir + '/val_losses_inference_only.txt', losses_list)
        np.savetxt(device_dir + '/accs_inference_only.txt', accs_list)  
        
    return losses_list, accs_list


def run_noisy_inference_for_tq_circuits_qiskit(circ_dir, circ_prefix, num_circs, num_runs, num_qubits, meas_qubits, device_name, noise_model, basis_gates,
                                               coupling_map, dataset, embed_type, num_data_reps, num_test_samples=None, human_design=False, compute_noiseless=False,
                                               use_qubit_mapping=False, num_shots=1024, qubit_mapping_file_name=None, transpile_opt_level=0, file_suffix='',
                                               index_map=None, results_save_dir=None, pick_best=False, use_quantumnat=False, quantumnat_trained_dir=None):
    """
    Run noisy inference for TQ circuits in the same folder - used to perform infrence for all ex. random, human designed, etc. circuits with one call.
    """
    x_train, y_train, x_test, y_test = load_dataset(dataset, embed_type, num_data_reps)
    
    if noise_model is None and 'ibm' in device_name:
        noise_model, basis_gates, coupling_map = get_noise_model(device_name)
        
    noisy_dev = AerSimulator(noise_model=noise_model)
    
    all_accs = []
    
    for i in range(num_circs):
        if num_test_samples:
            sel_inds = np.random.choice(len(x_test), num_test_samples, False)

            x_test = x_test[sel_inds]
            y_test = y_test[sel_inds]
        
        if human_design:
            curr_circ_dir = circ_dir
        else:
            curr_circ_dir = os.path.join(circ_dir, f'{circ_prefix}_{i + 1}')
            
        print('Saving results in', curr_circ_dir, results_save_dir)
        curr_results_save_dir = os.path.join(curr_circ_dir, results_save_dir)
            
        if use_qubit_mapping:
            qubit_mapping = np.genfromtxt(os.path.join(curr_circ_dir, qubit_mapping_file_name))
        else:
            qubit_mapping = None
        
        losses_list, accs_list = tq_model_inference_on_noisy_sim_qiskit(
            curr_circ_dir, device_name, num_runs, num_qubits, meas_qubits,
            noisy_dev, basis_gates, coupling_map, qubit_mapping,
            x_test, y_test, None, True, num_shots, compute_noiseless,
            transpile_opt_level, file_suffix, index_map, curr_results_save_dir,
            pick_best, use_quantumnat, quantumnat_trained_dir
        )
        
        all_accs.append(accs_list[0])
        
        print(i)
        
    print(np.mean(np.sort(all_accs)[5:]))
        
        
def run_noisy_inference_for_tq_circuits_pennylane(circ_dir, circ_prefix, num_circs, num_runs, num_qubits, meas_qubits, device_name, dataset,
                                        embed_type, num_data_reps, num_test_samples=None, human_design=False, compute_noiseless=False,
                                        noise_model=None, coupling_map=None, basis_gates=None):
    """
    Run noisy inference for TQ circuits in the same folder - used to perform infrence for all ex. random, human designed, etc. circuits with one call.
    """
    if noise_model is None:
        noisy_dev = noisy_dev_from_backend(device_name, num_qubits)
    else:
        noisy_dev = qml.device('qiskit.aer', wires=num_qubits, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)

    x_train, y_train, x_test, y_test = load_dataset(dataset, embed_type, num_data_reps)
    
    for i in range(num_circs):
        if num_test_samples:
            sel_inds = np.random.choice(len(x_test), num_test_samples, False)

            x_test = x_test[sel_inds]
            y_test = y_test[sel_inds]
        
        if human_design:
            curr_circ_dir = circ_dir
        else:
            curr_circ_dir = os.path.join(circ_dir, f'{circ_prefix}_{i + 1}')
        
        tq_model_inference_on_noisy_sim(curr_circ_dir, device_name, num_runs, meas_qubits, noisy_dev, x_test, y_test, None, True)
        
        print(i)
