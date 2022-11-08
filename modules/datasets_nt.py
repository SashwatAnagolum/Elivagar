import numpy as np

def load_dataset(name, embedding_type, num_reps):
    x_train = np.genfromtxt('./experiment_data/{}/x_train.txt'.format(name))
    x_test = np.genfromtxt('./experiment_data/{}/x_test.txt'.format(name))
    y_train = np.genfromtxt('./experiment_data/{}/y_train.txt'.format(name))
    y_test = np.genfromtxt('./experiment_data/{}/y_test.txt'.format(name))
    
    num_dims = 16
    
    if name == 'moons_300':
        x_train = x_train[:, :2]
        x_test = x_test[:, :2]
        
        y_train = 1 - 2 * y_train[:, 0]
        y_test = 1 - 2 * y_test[:, 0]
    elif name == 'bank':
        x_train = x_train[:, :4]
        x_test = x_test[:, :4]
        
        y_train = 1 - 2 * y_train[:, 0]
        y_test = 1 - 2 * y_test[:, 0]
    elif name == 'mnist_2' or 'fmnist_2':
        x_train = x_train[:, :16]
        x_test = x_test[:, :16]
    elif name == 'fmnist_4':
        x_train = x_train[:, :16]
        x_test = x_test[:, :16]
    elif name == 'mnist_10' or 'fmnist_10':
        x_train = x_train[:, :36]
        x_test = x_test[:, :36]
    elif name == 'vowel_2' or 'mnist_4' or 'vowel_4':
        pass
    else:
        print('Dataset not supported!')
        return
    
    if embedding_type == 'angle':
        pass
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
            pass
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
        elif name == 'bank' or name == 'moons_300':
            p_1 = np.prod(x_train[:, :], 1).reshape((len(x_train), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_train[0]) - 1)], 1)

            x_train = np.concatenate((x_train, p_1), 1)

            p_1 = np.prod(x_test[:, :], 1).reshape((len(x_test), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_test[0]) - 1)], 1)

            x_test = np.concatenate((x_test, p_1), 1)
        elif name == 'mnist_10' or 'fmnist_10':
            p_1 = np.prod(x_train[:, :6], 1).reshape((len(x_train), 1))
            p_2 = np.prod(x_train[:, 6:12], 1).reshape((len(x_train), 1))
            p_3 = np.prod(x_train[:, 12:18], 1).reshape((len(x_train), 1))
            p_4 = np.prod(x_train[:, 18:24], 1).reshape((len(x_train), 1))
            p_5 = np.prod(x_train[:, 24:30], 1).reshape((len(x_train), 1))
            p_6 = np.prod(x_train[:, 30:36], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:6], p_1, p_1, p_1, p_1, p_1, 
                                      x_train[:, 6:12], p_2, p_2, p_2, p_2, p_2,
                                      x_train[:, 12:18], p_3, p_3, p_3, p_3, p_3, 
                                      x_train[:, 18:24], p_4, p_4, p_4, p_4, p_4, 
                                      x_train[:, 24:30], p_5, p_5, p_5, p_5, p_5, 
                                      x_train[:, 30:36], p_6, p_6, p_6, p_6, p_6), 1)
            
            p_1 = np.prod(x_test[:, :6], 1).reshape((len(x_test), 1))
            p_2 = np.prod(x_test[:, 6:12], 1).reshape((len(x_test), 1))
            p_3 = np.prod(x_test[:, 12:18], 1).reshape((len(x_test), 1))
            p_4 = np.prod(x_test[:, 18:24], 1).reshape((len(x_test), 1))
            p_5 = np.prod(x_test[:, 24:30], 1).reshape((len(x_test), 1))
            p_6 = np.prod(x_test[:, 30:36], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:6], p_1, p_1, p_1, p_1, p_1, 
                                      x_test[:, 6:12], p_2, p_2, p_2, p_2, p_2,
                                      x_test[:, 12:18], p_3, p_3, p_3, p_3, p_3, 
                                      x_test[:, 18:24], p_4, p_4, p_4, p_4, p_4, 
                                      x_test[:, 24:30], p_5, p_5, p_5, p_5, p_5, 
                                      x_test[:, 30:36], p_6, p_6, p_6, p_6, p_6), 1)
        else:
            print('Dataset not supported!')
            return                
    elif embedding_type == 'supernet':
        if name == 'mnist_2' or name == 'fmnist_4':
            p_1 = np.prod(np.pi - x_train[:, :4], 1).reshape((len(x_train), 1))
            p_2 = np.prod(np.pi - x_train[:, 4:8], 1).reshape((len(x_train), 1))
            p_3 = np.prod(np.pi - x_train[:, 8:12], 1).reshape((len(x_train), 1))
            p_4 = np.prod(np.pi - x_train[:, 12:], 1).reshape((len(x_train), 1))
            
            x_train = np.concatenate((x_train[:, 0:4], p_1, p_1, p_1, x_train[:, 4:8], p_2, p_2, p_2, x_train[:, 8:12], p_3, p_3, p_3, x_train[:, 12:16], p_4, p_4, p_4), 1)
            
            p_1 = np.prod(np.pi - x_test[:, :4], 1).reshape((len(x_test), 1))
            p_2 = np.prod(np.pi - x_test[:, 4:8], 1).reshape((len(x_test), 1))
            p_3 = np.prod(np.pi - x_test[:, 8:12], 1).reshape((len(x_test), 1))
            p_4 = np.prod(np.pi - x_test[:, 12:], 1).reshape((len(x_test), 1))
            
            x_test = np.concatenate((x_test[:, 0:4], p_1, p_1, p_1, x_test[:, 4:8], p_2, p_2, p_2, x_test[:, 8:12], p_3, p_3, p_3, x_test[:, 12:16], p_4, p_4, p_4), 1)
        elif name == 'bank' or name == 'moons_300':
            p_1 = np.prod(np.pi - x_train[:, :], 1).reshape((len(x_train), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_train[0]) - 1)], 1)

            x_train = np.concatenate((x_train, p_1), 1)

            p_1 = np.prod(np.pi - x_test[:, :], 1).reshape((len(x_test), 1))
            p_1 = np.concatenate([p_1 for i in range(len(x_test[0]) - 1)], 1)

            x_test = np.concatenate((x_test, p_1), 1)
        elif name == 'mnist_10' or 'fmnist_10':
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
        else:
            print('Dataset not supported!')
            return
            
    x_train = np.mod(np.concatenate([x_train for i in range(num_reps)], 1), 2 * np.pi)
    x_test = np.mod(np.concatenate([x_test for i in range(num_reps)], 1), 2 * np.pi)   
    
    return x_train, y_train, x_test, y_test