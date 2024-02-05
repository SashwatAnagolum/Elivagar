import torch
import numpy as np
import os

from elivagar.circuits.arbitrary import get_circ_params
from elivagar.training.train_circ_np import TQMseLoss
from elivagar.utils.datasets import TorchDataset
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.metric_computation.compute_composite_scores import compute_composite_scores_for_circs

def train_elivagar_circuits(circ_dir, dataset, embed_type, num_data_reps, device_name, num_epochs, batch_size, num_qubits, num_meas_qubits,
                            num_data_for_rep_cap, num_params_for_rep_cap, num_cdcs, num_candidates_per_circuit=100,
                            num_circuits=2500, num_runs=5, noise_importance=0.5, results_dir=None, learning_rate=0.01):
    """
    Compute the composite scores for all the circuits in a directory and train the top-ranked circuits.
    """
    loss = TQMseLoss(num_meas_qubits, num_qubits)

    train_data = TorchDataset(dataset, embed_type, num_data_reps, reshape_labels=True)
    test_data = TorchDataset(dataset, embed_type, num_data_reps, False, reshape_labels=True)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.RandomSampler(train_data))
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=torch.utils.data.SequentialSampler(test_data))
    
    composite_scores, rep_cap_scores, cnr_scores = compute_composite_scores_for_circs(circ_dir, num_circuits,
        device_name, num_data_for_rep_cap, num_params_for_rep_cap, num_cdcs, noise_importance)
    
    ordering = np.random.permutation(num_circuits)
    
    num_trained_circuits = num_circuits // num_candidates_per_circuit
    
    if results_dir is None:
        if device_name is not None:
            results_dir = os.path.join(circ_dir, f'search_{num_candidates_per_circuit}_{device_name}')
        else:
            results_dir = os.path.join(circ_dir, f'search_{num_candidates_per_circuit}')
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    trial_losses = []
    trial_accs = []
    
    for i in range(num_trained_circuits):
        curr_circ_dir = os.path.join(results_dir, f'trial_{i + 1}')
        
        if not os.path.exists(curr_circ_dir):
            os.mkdir(curr_circ_dir)
        
        curr_candidate_inds = ordering[num_candidates_per_circuit * i:(i + 1) * num_candidates_per_circuit]
        curr_candidate_scores = composite_scores[curr_candidate_inds]
        curr_best_circuit_ind = curr_candidate_inds[np.argmax(curr_candidate_scores)]
        
        print(i)
        print(curr_candidate_inds, curr_best_circuit_ind)
        print(np.max(curr_candidate_scores))
        
        circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(os.path.join(circ_dir, f'circ_{curr_best_circuit_ind + 1}'))
        
        np.savetxt(curr_circ_dir + '/gates.txt', circ_gates, fmt="%s")
        np.savetxt(curr_circ_dir + '/gate_params.txt', gate_params, fmt="%s")
        np.savetxt(curr_circ_dir + '/inputs_bounds.txt', inputs_bounds)
        np.savetxt(curr_circ_dir + '/weights_bounds.txt', weights_bounds)
        
        np.savetxt(curr_circ_dir + '/searched_circuit_inds.txt', curr_candidate_inds)
        np.savetxt(curr_circ_dir + '/searched_circuit_scores.txt', curr_candidate_scores)
        np.savetxt(curr_circ_dir + '/sel_circuit_ind.txt', [curr_best_circuit_ind])
        np.savetxt(curr_circ_dir + '/sel_circuit_score.txt', [np.max(curr_candidate_scores)])
        
        circ_losses, circ_accs = train_tq_circ_and_save_results(curr_circ_dir, train_data_loader, test_data_loader, num_runs, num_epochs,
                                       num_qubits, num_meas_qubits, loss, learning_rate=learning_rate)
        
        trial_losses.append(circ_losses)
        trial_accs.append(circ_accs)

    mean_loss = np.mean(trial_losses)
    mean_acc = np.mean(trial_accs)
        
    return mean_loss, mean_acc