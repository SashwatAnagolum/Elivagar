import torch
import numpy as np
import os

from quantumnat.quantize import PACTActivationQuantizer

class QuantumNATModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        with torch.no_grad():
            return x

def quantize_and_normalize(exp_vals, model_dir):
    """
    Quantize and normalize the expectation values obtained from a circuit using the techniques from QuantumNAS.
    
    Quantization -> use the PACTACtivation quantizer
    Normalization -> use a BatchNorm1d layer.
    """
    model_data = torch.load(os.path.join(model_dir, 'model.pt'))
    norm_weights = (model_data['normalizer.weight'][:exp_vals.shape[1]]).repeat(exp_vals.shape[0], 1)
    norm_bias = (model_data['normalizer.bias'][:exp_vals.shape[1]]).repeat(exp_vals.shape[0], 1)
    
    quantumnat_model = QuantumNATModel()
    quantizer = PACTActivationQuantizer(
        quantumnat_model, precision=4, alpha=1.0, backprop_alpha=False,
        device=torch.device('cpu'), lower_bound=-5, upper_bound=5
    )
    
    quantizer.register_hook() 

    exp_vals = torch.from_numpy(exp_vals)
    
    batch_mean = torch.mean(exp_vals, dim=0, keepdim=True)
    batch_var = torch.var(exp_vals, dim=0, correction=0, keepdim=True)
    
    exp_vals = torch.divide(exp_vals - batch_mean, torch.sqrt(torch.add(batch_var, 1e-5)))
    exp_vals = torch.add(torch.mul(exp_vals, norm_weights), norm_bias) 
    
    exp_vals = quantumnat_model(exp_vals)
    
    return exp_vals.detach().numpy()