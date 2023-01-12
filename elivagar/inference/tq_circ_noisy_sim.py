import torch
import numpy as np
import pennylane as qml
import pickle as pkl
import os

from create_gate_circs import get_circ_params, create_gate_circ
from elivagar.utils.inference_metrics import mse_batch_loss, mse_vec_batch_loss, batch_acc, vec_batch_acc
from create_noise_models import noisy_dev_from_backend
from datasets_nt import load_dataset

def get_params_from_tq_model(model_dir, num_params):
    """
    Get the parameters stoerd in a torch quantum model.
    """
    model_data = torch.load(os.path.join(model_dir, 'model.pt'))
    model_params = np.zeros(num_params)
    
    for i in range(num_params):
        model_params[i] = model_data[f'var_gates.{i}.params'].numpy().item()
    
    return model_params
    

def tq_model_inference_on_noisy_sim(circ_dir, device_name, num_runs, meas_qubits, noisy_dev, x_test, y_test, params=None, save=True):
    """
    Peform inference on test data x_test using anoisy simulator noisy_dev with a trained circuit stored in circ_dir.
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


def run_noisy_inference_for_tq_circuits(circ_dir, circ_prefix, num_circs, num_runs, num_qubits, meas_qubits, device_name, dataset,
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