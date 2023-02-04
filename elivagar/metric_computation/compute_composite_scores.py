import numpy as np
import os

def compute_composite_scores_for_circs(circ_dir, num_circs, device_name, num_data, num_params_for_rep_cap,
                                       num_cdcs, noise_importance=0.5):
    rep_cap_scores = np.genfromtxt(
        os.path.join(circ_dir, f'rep_cap_scores_{num_data}_{num_params_for_rep_cap}.txt'))[:num_circs]
    
    cnr_scores = np.genfromtxt(
        os.path.join(circ_dir, f'cnr_scores_{device_name}_{num_cdcs}.txt'))[:num_circs]
    
    composite_scores = np.multiply(rep_cap_scores, np.power(cnr_scores, noise_importance))
        
    return composite_scores, rep_cap_scores, cnr_scores