import numpy as np
import torch

from datasets_nt import load_dataset

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, embed_type, reps, train=True, reshape_labels=False):
        x_train, y_train, x_test, y_test = load_dataset(dataset_name, embed_type, reps)
        
        if reshape_labels and len(y_train.shape) == 1:
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
        
        if train:  
            inds = np.random.permutation(len(x_train))
            
            self.x_train = x_train[inds]
            self.y_train = y_train[inds]
            
            self.length = len(x_train)
        else:
            inds = np.random.permutation(len(x_test))
            
            self.x_train = x_test[inds]
            self.y_train = y_test[inds]
            
            self.length = len(x_test)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        return self.x_train[ind], self.y_train[ind]