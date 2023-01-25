import numpy as np
import pennylane as qml


def generate_human_design_circ(dev, num_qubits, enc_layer, var_layer, num_enc_layers, num_var_layers, enc_layer_options, var_layer_options, measured_qubits, ret_type='exp'):
    """
    Generate a human-designed circuit using layer templates from the Pennylane library.
    """
    mapping = {
        'angle': qml.AngleEmbedding,
        'iqp': qml.IQPEmbedding,
        'amp': qml.AmplitudeEmbedding,
        'basic': qml.BasicEntanglerLayers,
        'strong': qml.StronglyEntanglingLayers
    }
    
    @qml.qnode(dev, interface='tf')
    def human_design_circ(inputs, weights):
        for i in range(num_enc_layers):
            mapping[enc_layer](inputs[i], range(num_qubits), *enc_layer_options[i])
            
        for i in range(num_var_layers):
            mapping[var_layer](weights[i], range(num_qubits), *var_layer_options[i])
            
        if ret_type == 'exp':
            return [qml.expval(qml.PauliZ(wires=i)) for i in measured_qubits]
        elif ret_type == 'state':
            return qml.state()
        elif ret_type == 'sample':
            return qml.sample(wires=measured_qubits)
        elif ret_type == 'matrix':
            return qml.density_matrix(wires=measured_qubits)
        else:
            return qml.probs(dev.wires)
    
    return human_design_circ


def get_iqp_embedding_layer(num_qubits, enc_layer_num, circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Get the gates in an IPQ embedding layer.
    """
    num_iqp_rzz_gates = (num_qubits * (num_qubits - 1)) // 2
            
    circ_gates += ['h' for i in range(num_qubits)]
    gate_params += [[i] for i in range(num_qubits)]
    inputs_bounds += [inputs_bounds[-1] for i in range(num_qubits)]
    weights_bounds += [weights_bounds[-1] for i in range(num_qubits)]

    circ_gates += ['rz' for i in range(num_qubits)]
    gate_params += [[i] for i in range(num_qubits)]
    inputs_bounds += [inputs_bounds[-1] + i + 1 for i in range(num_qubits)]
    weights_bounds += [weights_bounds[-1] for i in range(num_qubits)]
            
    circ_gates += ['rzz' for i in range(num_iqp_rzz_gates)]
    gate_params += [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    inputs_bounds += [inputs_bounds[-1] + i + 1 for i in range(num_iqp_rzz_gates)]
    weights_bounds += [weights_bounds[-1] for i in range(num_iqp_rzz_gates)]


def get_angle_embedding_layer(num_qubits, enc_layer_num, circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Get the gates in an angle embedding layer.
    """
    gate_used = 'ry' if (enc_layer_num % 2) == 1 else 'rx'
    
    circ_gates += [gate_used for i in range(num_qubits)]
    gate_params += [[i] for i in range(num_qubits)]
    inputs_bounds += [inputs_bounds[-1] + i + 1 for i in range(num_qubits)]
    weights_bounds += [weights_bounds[-1] for i in range(num_qubits)]  


def get_amp_encoding_layer(num_qubits, enc_layer_num, circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Get an amplitude encoding layer. WARNING: only use as the first gate / thing in a circuit.
    """
    circ_gates += ['amp_enc']
    gate_params += [[0]]
    inputs_bounds += [2 ** num_qubits]
    weights_bounds += [0]


def get_basic_var_layer(num_qubits, var_layer_num, circ_gates, gate_params, inputs_bounds, weights_bounds):
    """
    Get the gates in an angle embedding layer.
    """
    gate_used = 'ry' if (var_layer_num % 2) == 1 else 'rx'
    
    circ_gates += [gate_used for i in range(num_qubits)]
    gate_params += [[i] for i in range(num_qubits)]
    inputs_bounds += [inputs_bounds[-1] for i in range(num_qubits)]
    weights_bounds += [weights_bounds[-1] + i + 1 for i in range(num_qubits)]  \
    
    circ_gates += ['cx' for i in range(num_qubits)]
    gate_params += [[i, (i + 1) % num_qubits] for i in range(num_qubits)]
    inputs_bounds += [inputs_bounds[-1] for i in range(num_qubits)]
    weights_bounds += [weights_bounds[-1] for i in range(num_qubits)]  


def convert_human_design_circ_to_gate_circ(num_qubits, enc_layer, var_layer, num_enc_layers, num_var_layers):
    """
    Convert a human-designed circuit using layer templates from pennylane into a gate-based circuit. Currently only supports:
        BasicEntangler layers with alternating Ry and Rx rotations.
        IQPEmbedding layers with 1 repetition and the default pattern.
        AngleEmbedding layers using X and Y rotations.
    """
    circ_gates = []
    gate_params = []
    inputs_bounds = [0]
    weights_bounds = [0]
    
    layer_dict = {
        'angle': get_angle_embedding_layer,
        'iqp': get_iqp_embedding_layer,
        'basic': get_basic_var_layer,
        'amp': get_amp_encoding_layer}
    
    for i in range(num_enc_layers):
        layer_dict[enc_layer](num_qubits, i, circ_gates, gate_params, inputs_bounds, weights_bounds)
        
    for i in range(num_var_layers):
        layer_dict[var_layer](num_qubits, i, circ_gates, gate_params, inputs_bounds, weights_bounds)
        
    return circ_gates, gate_params, inputs_bounds, weights_bounds
        