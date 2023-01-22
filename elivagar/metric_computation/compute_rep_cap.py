import numpy as np
import os

from create_gate_circs import generate_random_gate_circ, generate_true_random_gate_circ, create_gate_circ, create_batched_gate_circ, get_circ_params
from create_human_design_circs import generate_human_design_circ
from metrics import compute_reduced_similarity
from datasets import load_dataset


def compute_reduced_similarity(circ, params, data, single_qubit=False):
    """
    Compute the similarity between the reduced density matrices obtained from a circuit's outputs.
    """
    num_data = len(data)
    traces = []
    circ_fids = np.zeros((num_data, num_data))
    
    if not single_qubit:
        circ_dms = circ(data, params)

        for i in range(num_data):
            traces.append(np.trace(circ_dms[i]))

        for s1 in range(num_data):
            for s2 in range(s1 + 1, num_data):
                trace_1 = traces[s1]
                trace_2 = traces[s2]
                fid_trace = np.trace(np.matmul(circ_dms[s1], circ_dms[s2]))

                curr_score = ((fid_trace) ** 2 / (trace_1 * trace_2)).real

                circ_fids[s1, s2] = curr_score
                circ_fids[s2, s1] = curr_score  

        circ_fids += np.eye(num_data)
    else:
        num_meas_qubits = len(circ)
        circ_dms = [circ[i](data, params) for i in range(num_meas_qubits)]
        all_fids = np.zeros((num_meas_qubits, num_data, num_data))
        
        for i in range(num_meas_qubits):
            traces = []
            
            for j in range(num_data):
                traces.append(np.trace(circ_dms[i][j]))

            for s1 in range(num_data):
                for s2 in range(s1 + 1, num_data):
                    trace_1 = traces[s1]
                    trace_2 = traces[s2]
                    fid_trace = np.trace(np.matmul(circ_dms[i][s1], circ_dms[i][s2]))

                    curr_score = ((fid_trace) ** 2 / (trace_1 * trace_2)).real

                    all_fids[i][s1, s2] = curr_score
                    all_fids[i][s2, s1] = curr_score  

            all_fids[i] += np.eye(num_data) 
            
        circ_fids = np.mean(all_fids, 0)
            
    return circ_fids


def compute_rep_cap(circ_dir, num_qubits, meas_qubits, data, ideal_matrix, num_param_samples, num_classes):
    """
    Compute the representational capacity of a circuit w.r.t. some passed in data.
    """
    importance_matrix = 1 - ideal_matrix
    importance_matrix /= (num_classes - 1)
    importance_matrix += ideal_matrix
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    
    batched_circ = create_batched_gate_circ(qml.device('lightning.qubit', wires=num_qubits), circ_gates, gate_params, inputs_bounds,
                                        weights_bounds, meas_qubits, 'matrix') 
    
    if not os.path.exists(circ_dir + '/rep_cap'):
        os.mkdir(circ_dir + '/rep_cap')

    params = 2 * np.pi * np.random.sample((num_params, weights_bounds[-1]))
    circ_mean_thres_mat = np.zeros((num_data, num_data))
    num_data = data.shape[0]
    
    np.savetxt(circ_dir + '/rep_cap/params.txt', params) 
    
    for i in range(num_param_samples):
        curr_params = np.concatenate([params[i] for k in range(num_data)]).reshape((num_data, weights_bounds[-1]))
        curr_mat = compute_reduced_similarity(batched_circ, curr_params, data)
        thres_mat = mat > ((np.sum(mat) - num_data) / (num_data * (num_data - 1)))
        
        circ_mean_thres_mat += thres_mat / num_param_samples

    diff_thres_mean_mat = ideal_matrix - circ_t_mean_mat
    scaled_diff_thres_mean_mat = np.multiply(diff_thres_mean_mat, importance_matrix)
    rep_cap = (num_data ** 2) - np.sum(np.power(scaled_diff_thres_mean_mat, 2))
    
    return rep_cap, circ_thres_mean_mat
    

def compute_rep_cap_for_circuits(circs_dir, num_circs, circ_prefix, num_qubits, meas_qubits, dataset_name, num_classes,
                                 samples_per_class, sel_samples_per_class, num_param_samples, encoding_type, 
                                 num_data_reps, save_circ_mats=False):
    """
    Compute representational capacity for a group of circuits in the same folder.
    """
    num_sel_samples = num_classes * sel_samples_per_class
    ideal = np.zeros((num_sel_samples, num_sel_samples))
    
    for i in range(num_classes):
        start_index = i * sel_samples_per_class
        end_index = (i + 1) * sel_samples_per_class
        
        ideal[start_index:end_index] = 1
    
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, encoding_type, num_data_reps)
    sel_inds = []
    
    for i in range(num_classes):
        sel_inds.append(np.random.choice(samples_per_class, 16, False) + num_samples_per_class * i)
        
    sel_inds = np.concatenate(sel_inds)
    sel_data = x_train[sel_inds]
    
    circ_rep_caps = []
    
    for i in range(num_circs):
        curr_circ_dir = os.path.join(circs_dir, circ_prefix + f'_{i + 1}')
        circ_rep_cap, circ_thres_mean_mat = compute_rep_cap(curr_circ_dir, num_qubits, meas_qubits, sel_data, ideal, num_param_samples, num_classes)   
        
        if save_circ_mats:
            np.savetxt(os.path.join(curr_circ_dir, 'rep_cap/thres_mean_mat.txt'), circ_thres_mean_mat)
            
        circ_rep_caps.append(circ_rep_cap)
            
    np.savetxt(os.path.join(circs_dir 'rep_caps.txt'), circ_rep_caps)
