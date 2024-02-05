import pickle as pkl
import torch
import os
import numpy as np

from elivagar.utils.datasets import TorchDataset
from elivagar.circuits.arbitrary import get_circ_params, generate_true_random_gate_circ
from elivagar.circuits.create_circuit import TQCirc
from elivagar.training.train_circ_np import train_tq_model, TQMseLoss

def train_tq_circ_and_save_results(circ_dir, train_data_loader, test_data_loader, num_runs, num_train_epochs,
                                   num_qubits, num_meas_qubits, loss_fn, dataset_name=None,
                                   save_loss_curves=False, file_suffix='', learning_rate=0.01,
                                   results_save_dir=None, quantize=False, noise_strength=0.05,
                                   use_qtn_vqc=False, tt_input_size=None, tt_ranks=None, tt_output_size=None,
                                   device=None):
    """
    Train the TQ circuit in the directory passed in, and save the trained loss and accuracy values, as well as the trained model(s).
    """
    if device is None:
        device = torch.device('cpu')
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    
    if results_save_dir is None:
        results_save_dir = circ_dir
    
    if dataset_name is not None:
        results_save_dir = results_save_dir + '/' + dataset_name.replace(' ', '_')
        
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    
    losses_list = []
    accs_list = []

    for j in range(num_runs):
        curr_train_dir = os.path.join(results_save_dir, 'run_{}'.format(j + 1))

        if os.path.exists(curr_train_dir):
            pass
        else:
            os.mkdir(curr_train_dir)

        model = TQCirc(
            circ_gates, gate_params, inputs_bounds, weights_bounds,
            num_qubits, False, quantize, noise_strength, use_qtn_vqc,
            tt_input_size, tt_ranks, tt_output_size
        ).to(device)
        
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        curr_stats = train_tq_model(
            model, num_meas_qubits, opt, loss_fn, train_data_loader, test_data_loader,
            num_train_epochs, 100, 10, return_train_losses=save_loss_curves, quantize=quantize,
            device=device
        )

        print(f'\nAccuracy: {curr_stats[1]} | MSE Loss: {curr_stats[0]}\n')

        torch.save(model.state_dict(), os.path.join(curr_train_dir, 'model.pt'))

        if save_loss_curves:
            np.savetxt(os.path.join(curr_train_dir, 'train_losses.txt'), curr_stats[2])
        
        losses_list.append(curr_stats[0])
        accs_list.append(curr_stats[1])

    np.savetxt(os.path.join(results_save_dir, f'val_losses{file_suffix}.txt'), losses_list)
    np.savetxt(os.path.join(results_save_dir, f'accs{file_suffix}.txt'), accs_list)
    
    return losses_list, accs_list 