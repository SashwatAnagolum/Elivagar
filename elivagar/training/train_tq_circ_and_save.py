import pickle as pkl
import torch
import os
import numpy as np

from datasets import TorchDataset
from create_gate_circs_np import get_circ_params, TQCirc, generate_true_random_gate_circ
from train_circ_np import train_tq_model, TQMseLoss


def train_tq_circ_and_save_results(circ_dir, train_data_loader, test_data_loader, num_test_data, num_runs, num_train_steps,
                                   num_qubits, num_meas_qubits, loss_fn, dataset_name=None):
    """
    Train the TQ circuit in the directory passed in, and save the trained loss and accuray values, as well as the trained model(s).
    """
    device = 'cpu'
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    
    if dataset_name is not None:
        circ_dir = circ_dir + '/' + dataset_name.replace(' ', '_')
        
        if not os.path.exists(circ_dir):
            os.mkdir(circ_dir)
    
    losses_list = []
    accs_list = []

    for j in range(num_runs):
        curr_train_dir = os.path.join(circ_dir, 'run_{}'.format(j + 1))

        if os.path.exists(curr_train_dir):
            pass
        else:
            os.mkdir(curr_train_dir)

        model = TQCirc(circ_gates, gate_params, inputs_bounds, weights_bounds, num_qubits, False).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)

        curr_loss, curr_acc = train_tq_model(model, num_meas_qubits, opt, loss_fn, train_data_loader, test_data_loader,
                                             num_test_data, num_train_steps, 100, 10)

        print(curr_loss, curr_acc)

        torch.save(model.state_dict(), os.path.join(curr_train_dir, 'model.pt'))

        losses_list.append(curr_loss)
        accs_list.append(curr_acc)

    np.savetxt(os.path.join(circ_dir, 'val_losses.txt'), losses_list)
    np.savetxt(os.path.join(circ_dir, 'accs.txt'), accs_list)
    
    return losses_list, accs_list 