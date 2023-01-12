import numpy as np
import os

from create_gate_circs import generate_random_gate_circ, generate_true_random_gate_circ, create_gate_circ, create_batched_gate_circ, get_circ_params
from create_human_design_circs import generate_human_design_circ
from metrics import compute_reduced_similarity
from datasets_nt import load_dataset

ideal = np.concatenate((np.ones((16, 16)), np.zeros((16, 16)), np.zeros((16, 16)), np.zeros((16, 16))))
ideal_2 = np.concatenate((np.zeros((16, 16)), np.ones((16, 16)), np.zeros((16, 16)), np.zeros((16, 16))))
ideal = np.concatenate((ideal, ideal_2, ideal_2[::-1, :], ideal[::-1, :]), 1)

# ideal = np.concatenate((np.ones((16, 16)), np.zeros((16, 16))))
# ideal = np.concatenate((ideal, ideal[::-1, :]), 1)

# ideal = 2 * ideal - 1

dataset = 'vowel_4'

curr_dir = f'./experiment_data/{dataset}/trained_circuits'
num_qubits = 4
meas_qubits = [0, 1]
num_samples_per_class = 150

x_train, y_train, x_test, y_test = load_dataset(dataset, 'angle', 1)

class_0_sel = np.random.choice(num_samples_per_class, 16, False)
class_1_sel = np.random.choice(num_samples_per_class, 16, False) + num_samples_per_class
class_2_sel = np.random.choice(num_samples_per_class, 16, False) + num_samples_per_class * 2
class_3_sel = np.random.choice(num_samples_per_class, 16, False) + num_samples_per_class * 3
sel_inds = np.concatenate((class_0_sel, class_1_sel, class_2_sel, class_3_sel))

# class_0_sel = np.random.choice(len(x_test) // 2, 16, False)
# class_1_sel = np.random.choice(len(x_test) // 2, 16, False) + (len(x_test) // 2)
# sel_inds = np.concatenate((class_0_sel, class_1_sel))

sel_data = x_train[sel_inds]
# sel_data = np.genfromtxt(curr_dir + '/sel_data.txt')

num_params = 32
num_data = len(sel_data)

d2_min_scores = []
d2_mean_scores = []
d2_var_scores = []
    
d2_t_min_scores = []
d2_t_mean_scores = []
d2_t_var_scores = []

mean_mat_scores = []
mean_t_mat_scores = []

np.savetxt(curr_dir + '/sel_data.txt', sel_data)

for i in range(1000):
    circ_dir = curr_dir + '/circ_{}'.format(i + 1)
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    
    batched_circ = create_batched_gate_circ(qml.device('lightning.qubit', wires=num_qubits), circ_gates, gate_params, inputs_bounds,
                                        weights_bounds, meas_qubits, 'matrix') 
    
    params = 2 * np.pi * np.random.sample((num_params, weights_bounds[-1]))
    
    if not os.path.exists(circ_dir + '/fid_mats'):
        os.mkdir(circ_dir + '/fid_mats')
    
    np.savetxt(circ_dir + '/fid_mats/metric_params.txt', params) 
    
    circ_d2_scores = []
    circ_d2_t_scores = []
    circ_mean_mat = np.zeros((num_data, num_data))
    circ_t_mean_mat = np.zeros((num_data, num_data))
    
    for j in range(num_params):
        curr_params = np.concatenate([params[j] for k in range(num_data)]).reshape((num_data, weights_bounds[-1]))
        mat = compute_reduced_similarity(batched_circ, curr_params, sel_data)
        
        t_mat = mat > ((np.sum(mat) - num_data) / (num_data * (num_data - 1)))
        
        diff_mat = mat - ideal
        diff_2d = np.sum(np.multiply(diff_mat, diff_mat))
            
        diff_t_mat = t_mat - ideal
        diff_2d_t = np.sum(np.multiply(diff_t_mat, diff_t_mat))
            
        circ_d2_scores.append(diff_2d)
        circ_d2_t_scores.append(diff_2d_t)
            
        circ_mean_mat += mat / num_params
        circ_t_mean_mat += t_mat / num_params

    np.savetxt(circ_dir + '/fid_mats/d2_scores_2.txt', circ_d2_scores)
    np.savetxt(circ_dir + '/fid_mats/d2_t_scores_2.txt', circ_d2_t_scores)
        
    d2_min_scores.append(np.min(circ_d2_scores))
    d2_mean_scores.append(np.mean(circ_d2_scores))
    d2_var_scores.append(np.var(circ_d2_scores))
        
    d2_t_min_scores.append(np.min(circ_d2_t_scores))
    d2_t_mean_scores.append(np.mean(circ_d2_t_scores))
    d2_t_var_scores.append(np.var(circ_d2_t_scores))
        
    diff_mean_mat = ideal - circ_mean_mat
    diff_t_mean_mat = ideal - circ_t_mean_mat   
    
    mean_mat_scores.append(np.sum(np.multiply(diff_mean_mat, diff_mean_mat)))
    mean_t_mat_scores.append(np.sum(np.multiply(diff_t_mean_mat, diff_t_mean_mat)))
    
    np.savetxt(circ_dir + '/fid_mats/mean_t_mat.txt', circ_t_mean_mat)
    
    print(i)
        
np.savetxt(curr_dir + '/d2_mean_scores_2.txt', d2_mean_scores)
np.savetxt(curr_dir + '/d2_min_scores_2.txt', d2_min_scores)
np.savetxt(curr_dir + '/d2_var_scores_2.txt', d2_var_scores)
    
np.savetxt(curr_dir + '/d2_t_mean_scores_2.txt', d2_t_mean_scores)
np.savetxt(curr_dir + '/d2_t_min_scores_2.txt', d2_t_min_scores)
np.savetxt(curr_dir + '/d2_t_var_scores_2.txt', d2_t_var_scores)
    
np.savetxt(curr_dir + '/d2_mean_mat_scores_kta.txt', mean_mat_scores)
np.savetxt(curr_dir + '/d2_mean_t_mat_scores_kta.txt', mean_t_mat_scores)