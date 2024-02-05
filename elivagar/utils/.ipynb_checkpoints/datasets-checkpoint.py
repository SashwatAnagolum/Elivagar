import numpy as np
import torch


def load_dataset(name, embedding_type, num_reps, file_type='txt'):
    if file_type == 'txt':
        x_train = np.genfromtxt('./experiment_data/{}/x_train.txt'.format(name))
        x_test = np.genfromtxt('./experiment_data/{}/x_test.txt'.format(name))
        y_train = np.genfromtxt('./experiment_data/{}/y_train.txt'.format(name))
        y_test = np.genfromtxt('./experiment_data/{}/y_test.txt'.format(name))
    elif file_type == 'npy':
        x_train = np.load('./experiment_data/{}/x_train.npy'.format(name))
        x_test = np.load('./experiment_data/{}/x_test.npy'.format(name))
        y_train = np.load('./experiment_data/{}/y_train.npy'.format(name))
        y_test = np.load('./experiment_data/{}/y_test.npy'.format(name))        
        
    if name == 'moons':
        x_train = x_train[:, :2]
        x_test = x_test[:, :2]
        
        y_train = 1 - 2 * y_train[:, 0]
        y_test = 1 - 2 * y_test[:, 0]
    elif name == 'bank':
        x_train = x_train[:, :4]
        x_test = x_test[:, :4]
        
        y_train = 1 - 2 * y_train[:, 0]
        y_test = 1 - 2 * y_test[:, 0]
    elif name in ['mnist_2', 'fmnist_2', 'mnist_4', 'fmnist_4']:
        x_train = x_train[:, :16]
        x_test = x_test[:, :16]
    elif name in ['mnist_10_6', 'mnist_10_8']:
        width = 8 if name == 'mnist_10_8' else 6
        x_train = x_train[:, :width * width]
        x_test = x_test[:, :width * width]
        
        y_train = 1 - 2 * y_train
        y_test = 1 - 2 * y_test
    elif name == 'mnist_10_6_nll':
        x_train = x_train[:, :36]
        x_test = x_test[:, :36]        
    elif name in ['vowel_2', 'vowel_4']:
        x_train = x_train[:, :10]
        x_test = x_test[:, :10]
    elif name in ['mnist_2_fullsize', 'mnist_4_fullsize', 'fmnist_2_fullsize', 'fmnist_4_fullsize']:
        y_train = 1 - 2 * y_train
        y_test = 1 - 2 * y_test
    elif name == 'mnist_10':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_train = 1 - 2 * y_train
        y_test = 1 - 2 * y_test
    else:
        print('Dataset not supported!')
        return
    
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    
    if embedding_type == 'angle':
        pass
    elif embedding_type == 'angle_layer':
        if name in ['vowel_2', 'vowel_4']:
            x_train = np.concatenate((x_train, np.zeros((len(x_train), 2))), 1)
            x_test = np.concatenate((x_test, np.zeros((len(x_test), 2))), 1)
    elif embedding_type == 'iqp':
        if name in ['mnist_2', 'mnist_4', 'fmnist_2', 'fmnist_4']:
            p_1 = np.prod(x_train[:, :4], 1).reshape((len(x_train), 1))
            p_2 = np.prod(x_train[:, 4:8], 1).reshape((len(x_train), 1))
            p_3 = np.prod(x_train[:, 8:12], 1).reshape((len(x_train), 1))
            p_4 = np.prod(x_train[:, 12:], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:4], 
                                      np.concatenate(([p_1 for i in range(6)]), 1),
                                      x_train[:, 4:8], 
                                      np.concatenate(([p_2 for i in range(6)]), 1),
                                      x_train[:, 8:12],
                                      np.concatenate(([p_3 for i in range(6)]), 1),
                                      x_train[:, 12:16],
                                      np.concatenate(([p_4 for i in range(6)]), 1)), 1)
            
            p_1 = np.prod(x_test[:, :4], 1).reshape((len(x_test), 1))
            p_2 = np.prod(x_test[:, 4:8], 1).reshape((len(x_test), 1))
            p_3 = np.prod(x_test[:, 8:12], 1).reshape((len(x_test), 1))
            p_4 = np.prod(x_test[:, 12:], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:4], 
                                      np.concatenate(([p_1 for i in range(6)]), 1),
                                      x_test[:, 4:8], 
                                      np.concatenate(([p_2 for i in range(6)]), 1),
                                      x_test[:, 8:12],
                                      np.concatenate(([p_3 for i in range(6)]), 1),
                                      x_test[:, 12:16],
                                      np.concatenate(([p_4 for i in range(6)]), 1)), 1)
        elif name in ['vowel_2', 'vowel_4']:
            x_train = np.concatenate((x_train, np.zeros((len(x_train), 2))), 1)
            x_test = np.concatenate((x_test, np.zeros((len(x_test), 2))), 1)
            
            p_1 = np.prod(x_train[:, :4], 1).reshape((len(x_train), 1))
            p_2 = np.prod(x_train[:, 4:8], 1).reshape((len(x_train), 1))
            p_3 = np.prod(x_train[:, 8:10], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:4], 
                                      np.concatenate(([p_1 for i in range(6)]), 1),
                                      x_train[:, 4:8], 
                                      np.concatenate(([p_2 for i in range(6)]), 1),
                                      x_train[:, 8:12],
                                      np.concatenate(([p_3 for i in range(6)]), 1)), 1)
            
            p_1 = np.prod(x_test[:, :4], 1).reshape((len(x_test), 1))
            p_2 = np.prod(x_test[:, 4:8], 1).reshape((len(x_test), 1))
            p_3 = np.prod(x_test[:, 8:10], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:4], 
                                      np.concatenate(([p_1 for i in range(6)]), 1),
                                      x_test[:, 4:8], 
                                      np.concatenate(([p_2 for i in range(6)]), 1),
                                      x_test[:, 8:12],
                                      np.concatenate(([p_3 for i in range(6)]), 1)), 1)
        elif name in ['bank', 'moons']:
            p_1 = np.prod(x_train[:, :], 1).reshape((len(x_train), 1))
            p_1 = np.concatenate([p_1 for i in range((len(x_train[0]) * (len(x_train[0]) - 1)) // 2)], 1)

            x_train = np.concatenate((x_train, p_1), 1)

            p_1 = np.prod(x_test[:, :], 1).reshape((len(x_test), 1))
            p_1 = np.concatenate([p_1 for i in range((len(x_test[0]) * (len(x_test[0]) - 1)) // 2)], 1)

            x_test = np.concatenate((x_test, p_1), 1)
        elif name in ['mnist_10_6', 'mnist_10_8', 'mnist_10_6_nll']:
            width = int(np.sqrt(len(x_train[0])))
            ps = []
            ps_test = []
            
            for i in range(width):
                ps.append(np.prod(x_train[:, width * i:width * (i + 1)], 1).reshape((len(x_train), 1)))
                ps.append(np.prod(x_test[:, width * i:width * (i + 1)], 1).reshape((len(x_test), 1)))
                
            new_x_train = []
            new_x_test = []
            
            for i in range(width):
                new_x_train = np.concatenate(
                    (
                        new_x_train,
                        x_train[:, width * i:width * (i + 1)],
                        *[ps[i] for j in range((width * (width - 1)) // 2)]
                    ), 1
                )
                
                new_x_test = np.concatenate(
                    (
                        new_x_test,
                        x_test[:, width * i:width * (i + 1)],
                        *[ps_test[i] for j in range((width * (width - 1)) // 2)]
                    ), 1
                ) 
            
            x_train = new_x_train
            x_test = new_x_test
    elif embedding_type == 'supernet':
        if name in ['mnist_2', 'mnist_4', 'fmnist_2', 'fmnist_4']:
            p_1 = np.prod(np.pi - x_train[:, :4], 1).reshape((len(x_train), 1))
            p_2 = np.prod(np.pi - x_train[:, 4:8], 1).reshape((len(x_train), 1))
            p_3 = np.prod(np.pi - x_train[:, 8:12], 1).reshape((len(x_train), 1))
            p_4 = np.prod(np.pi - x_train[:, 12:], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:4], p_1, p_1, p_1, x_train[:, 4:8], p_2, p_2,
                                      p_2, x_train[:, 8:12], p_3, p_3, p_3, x_train[:, 12:16],
                                      p_4, p_4, p_4), 1)
            
            p_1 = np.prod(np.pi - x_test[:, :4], 1).reshape((len(x_test), 1))
            p_2 = np.prod(np.pi - x_test[:, 4:8], 1).reshape((len(x_test), 1))
            p_3 = np.prod(np.pi - x_test[:, 8:12], 1).reshape((len(x_test), 1))
            p_4 = np.prod(np.pi - x_test[:, 12:], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:4], p_1, p_1, p_1, x_test[:, 4:8], p_2, p_2,
                                     p_2, x_test[:, 8:12], p_3, p_3, p_3, x_test[:, 12:16],
                                     p_4, p_4, p_4), 1)
        elif name == 'bank' or name == 'moons':
            p_1 = np.prod(np.pi - x_train[:, :], 1).reshape((len(x_train), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_train[0]) - 1)], 1)

            x_train = np.concatenate((x_train, p_1), 1)

            p_1 = np.prod(np.pi - x_test[:, :], 1).reshape((len(x_test), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_test[0]) - 1)], 1)

            x_test = np.concatenate((x_test, p_1), 1)
        elif name in ['mnist_10_6', 'mnist_10_6_nll']:
            p_1 = np.prod(np.pi - x_train[:, :6], 1).reshape((len(x_train), 1))
            p_2 = np.prod(np.pi - x_train[:, 6:12], 1).reshape((len(x_train), 1))
            p_3 = np.prod(np.pi - x_train[:, 12:18], 1).reshape((len(x_train), 1))
            p_4 = np.prod(np.pi - x_train[:, 18:24], 1).reshape((len(x_train), 1))
            p_5 = np.prod(np.pi - x_train[:, 24:30], 1).reshape((len(x_train), 1))
            p_6 = np.prod(np.pi - x_train[:, 30:36], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:6], p_1, p_1, p_1, p_1, p_1, 
                                      x_train[:, 6:12], p_2, p_2, p_2, p_2, p_2,
                                      x_train[:, 12:18], p_3, p_3, p_3, p_3, p_3, 
                                      x_train[:, 18:24], p_4, p_4, p_4, p_4, p_4, 
                                      x_train[:, 24:30], p_5, p_5, p_5, p_5, p_5, 
                                      x_train[:, 30:36], p_6, p_6, p_6, p_6, p_6), 1)
            
            p_1 = np.prod(np.pi - x_test[:, :6], 1).reshape((len(x_test), 1))
            p_2 = np.prod(np.pi - x_test[:, 6:12], 1).reshape((len(x_test), 1))
            p_3 = np.prod(np.pi - x_test[:, 12:18], 1).reshape((len(x_test), 1))
            p_4 = np.prod(np.pi - x_test[:, 18:24], 1).reshape((len(x_test), 1))
            p_5 = np.prod(np.pi - x_test[:, 24:30], 1).reshape((len(x_test), 1))
            p_6 = np.prod(np.pi - x_test[:, 30:36], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:6], p_1, p_1, p_1, p_1, p_1, 
                                      x_test[:, 6:12], p_2, p_2, p_2, p_2, p_2,
                                      x_test[:, 12:18], p_3, p_3, p_3, p_3, p_3, 
                                      x_test[:, 18:24], p_4, p_4, p_4, p_4, p_4, 
                                      x_test[:, 24:30], p_5, p_5, p_5, p_5, p_5, 
                                      x_test[:, 30:36], p_6, p_6, p_6, p_6, p_6), 1)
        elif name in ['vowel_2', 'vowel_4']:
            x_train = np.concatenate((x_train, np.zeros((len(x_train), 2))), 1)
            x_test = np.concatenate((x_test, np.zeros((len(x_test), 2))), 1)
            
            p_1 = np.prod(np.pi - x_train[:, :4], 1).reshape((len(x_train), 1))
            p_2 = np.prod(np.pi - x_train[:, 4:8], 1).reshape((len(x_train), 1))
            p_3 = np.prod(np.pi - x_train[:, 8:12], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, :4], p_1, p_1, p_1, x_train[:, 4:8],
                                      p_2, p_2, p_2, x_train[:, 8:12], p_3, p_3,
                                      p_3), 1)
            
            p_1 = np.prod(np.pi - x_test[:, :4], 1).reshape((len(x_test), 1))
            p_2 = np.prod(np.pi - x_test[:, 4:8], 1).reshape((len(x_test), 1))
            p_3 = np.prod(np.pi - x_test[:, 8:12], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, :4], p_1, p_1, p_1, x_test[:, 4:8],
                                      p_2, p_2, p_2, x_test[:, 8:12], p_3, p_3,
                                      p_3), 1)
        else:
            print('Dataset not supported!')
            return
            
    x_train = np.mod(np.concatenate([x_train for i in range(num_reps)], 1), 2 * np.pi)
    x_test = np.mod(np.concatenate([x_test for i in range(num_reps)], 1), 2 * np.pi)   
    
    return x_train, y_train, x_test, y_test


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, embed_type, reps, train=True, reshape_labels=False, file_type='txt'):
        x_train, y_train, x_test, y_test = load_dataset(dataset_name, embed_type, reps, file_type)
        
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
