import torch
import numpy as np
import os

from tc.tc_fc import TTLinear
from torch.utils.data import DataLoader

from elivagar.circuits.create_circuit import TQCirc
from elivagar.inference.noise_model import get_params_from_tq_model
from elivagar.utils.datasets import TorchDataset

def preprocess_data_using_tt_layer(model_dir, dataset, tt_input_size, tt_ranks, tt_output_size,
                                   embed_type, num_data_reps, file_type):
    """
    Preprocess the samples in data using the Tensor Train Network used as part of 
    the model saved in model_dir.
    """
    train_ds = TorchDataset(dataset, embed_type, num_data_reps, True, True, file_type)
    test_ds = TorchDataset(dataset, embed_type, num_data_reps, False, True, file_type)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    model_data = torch.load(os.path.join(model_dir, 'model.pt'))
    
    tt_layer =  TTLinear(
        inp_modes=tt_input_size,
        out_modes=tt_output_size,
        tt_rank=tt_ranks
    )
    
    tt_layer.load_state_dict(model_data, strict=False)
    
    processed_train_data = []
    processed_test_data = []
    train_labels = []
    test_labels = []
    
    for data, labels in train_loader:
        train_labels.append(labels.detach().numpy())
        processed_train_data.append(tt_layer(data.float()).detach().numpy())
        
    for data, labels in test_loader:
        test_labels.append(labels.detach().numpy())
        processed_test_data.append(tt_layer(data.float()).detach().numpy())
        
    processed_train_data = np.concatenate(processed_train_data, 0)
    processed_test_data = np.concatenate(processed_test_data, 0)
    
    train_labels = np.concatenate(train_labels, 0)
    test_labels = np.concatenate(test_labels, 0)
    
    return processed_train_data, train_labels, processed_test_data, test_labels