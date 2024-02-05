import numpy as np
import os

def compute_composite_scores_for_circs(circ_dir, num_circs, device_name, num_data, num_params_for_rep_cap,
                                       num_cdcs, noise_importance=0.5):
    rep_cap_scores = []
    cnr_scores = []
    
    for i in range(num_circs):
        curr_rep_cap = np.genfromtxt(os.path.join(circ_dir, f'circ_{i + 1}', 'rep_cap', f'{num_data}_{num_params_for_rep_cap}', 'score.txt'))
        rep_cap_scores.append(curr_rep_cap)
        
    if device_name is not None:
        for i in range(num_circs):
            curr_cnr = np.genfromtxt(os.path.join(circ_dir, f'circ_{i + 1}', f'cnr_{num_cdcs}', f'{device_name}', 'cnr.txt'))
            cnr_scores.append(curr_cnr)
    else:
        cnr_scores = [1.0 for i in range(num_circs)]
        
    rep_cap_scores = np.array(rep_cap_scores)
    cnr_scores = np.array(cnr_scores)
    
    composite_scores = np.multiply(rep_cap_scores, np.power(cnr_scores, noise_importance))
        
    return composite_scores, rep_cap_scores, cnr_scores