import numpy as np
import os

def compute_composite_scores_for_circs(circ_dir, num_circs, device_name, num_data, noise_metric_dir_name='noise_metric', noise_importance=0.5):
    rep_cap_scores = np.genfromtxt(os.path.join(circ_dir, 'd2_mean_t_mat_scores.txt'))[:num_circs]
    matrix_dim = num_data ** 2
    
    rep_cap_scores = (matrix_dim - rep_cap_scores) / matrix_dim
    noise_scores = []
    
    if device_name is not None:
        for i in range(num_circs):
            curr_noise_score_file = os.path.join(circ_dir, f'circ_{i + 1}/{noise_metric_dir_name}/{device_name}/metric_tvd_score.txt')
            noise_scores.append(np.genfromtxt(curr_noise_score_file))
        
        composite_scores = np.multiply(rep_cap_scores, np.power(np.array(noise_scores).flatten(), noise_importance))
    else:
        composite_scores = rep_cap_scores
        
    return composite_scores