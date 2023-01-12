import pennylane as qml
import numpy as np

def rot_layer(_, params):
    for i in range(len(params) // 3):
        qml.Rot(params[3 * i], params[3 * i + 1], params[3 * i + 2], wires=i)


def ry_layer(rot_map, params):
    if not rot_map:
        for i in range(len(params)):
            qml.RY(params[i], wires=i)
    elif rot_map == 'middle':
        if not num_qubits % 2:
            qml.RY(params[0], wires=num_qubits // 2)
            qml.RY(params[1], wires=num_qubits // 2 + 1)
        else:
            qml.RY(params[0], wires=num_qubits // 2 + 1)
    

def rx_layer(rot_map, params):
    if not rot_map:
        for i in range(len(params)):
            qml.RX(params[i], wires=i)
    elif rot_map == 'middle':
        if num_qubits % 2:
            qml.RX(params[0], wires=num_qubits // 2)
            qml.RX(params[1], wires=num_qubits // 2 + 1)
        else:
            qml.RX(params[0], wires=num_qubits // 2 + 1)
        

def rz_layer(rot_map, params):
    if not rot_map:
        for i in range(len(params)):
            qml.RZ(params[i], wires=i)
    elif rot_map == 'middle':
        if num_qubits % 2:
            qml.RZ(params[0], wires=num_qubits // 2)
            qml.RZ(params[1], wires=num_qubits // 2 + 1)
        else:
            qml.RZ(params[0], wires=num_qubits // 2 + 1)
        

def cr_layer(gate, cr_map, num_qubits, params):
    gate = {'crz': qml.CRZ, 'cry': qml.CRY, 'crx': qml.CRX}[gate]

    if cr_map == 'linear':
        for i, theta in enumerate(params):
            gate(theta, wires=[i % num_qubits, (i + 1) % num_qubits])
            
    if cr_map == 'linear_rev':
        for i, theta in enumerate(params):
            gate(theta, wires=[(i + 1) % num_qubits, i % num_qubits])
            
    if cr_map == 'full':
        index = 0
        
        for i in range(num_qubits):
            offset = i * (num_qubits - 1)
            
            for j in range(num_qubits):
                if i != j:
                    gate(params[index], wires=[i, j])
                    index += 1
                    
    if cr_map == 'full_rev':
        index = 0
        
        for i in range(num_qubits - 1,  -1,  -1):
            offset = (num_qubits - i - 1) * (num_qubits - 1)
            print(offset)
            for j in range(num_qubits - 1,  -1,  -1):
                if i != j:
                    gate(params[index], wires=[i, j])
                    index += 1
                    
    if cr_map == 'triangle':
        index = 0
        
        for i in range(num_qubits // 2):
            for j in range(num_qubits // 2 - i):
                gate(params[index], wires=[i + (2 * j), i + (2 * j) + 1])
                index += 1
                
    if cr_map == 'alt':
        for i in range(num_qubits // 2):
            gate(params[i], wires=[2 * i, 2 * i + 1])
            
    if cr_map == 'middle':
        gate(params[0], wires=[(num_qubits - 1) // 2, (num_qubits - 1) // 2 + 1])
        
    if cr_map == 'circular':
        for i in range(num_qubits):
            gate(params[i], wires=[i, (i + 1) % num_qubits])
            
    if cr_map == 'circular_rev':
        for i in range(num_qubits):
            gate(params[i], wires=[(i + 1) % num_qubits, i])
        
        
def double_pauli_layer(gate, p_map, num_qubits, params):
    gate = {'xx': qml.IsingXX, 'yy': qml.IsingYY, 'zz': qml.IsingZZ}[gate]

    if p_map == 'linear':
        for i, theta in enumerate(params):
            gate(theta, wires=[i % num_qubits, (i + 1) % num_qubits])
            
    if p_map == 'linear_rev':
        for i, theta in enumerate(params):
            gate(theta, wires=[(i + 1) % num_qubits, i % num_qubits])
            
    if p_map == 'full':
        index = 0
        
        for i in range(num_qubits):
            offset = i * (num_qubits - 1)
            
            for j in range(num_qubits):
                if i != j:
                    gate(params[index], wires=[i, j])
                    index += 1
                    
    if p_map == 'full_rev':
        index = 0
        
        for i in range(num_qubits - 1,  -1,  -1):
            offset = (num_qubits - i - 1) * (num_qubits - 1)
            print(offset)
            for j in range(num_qubits - 1,  -1,  -1):
                if i != j:
                    gate(params[index], wires=[i, j])
                    index += 1
                    
    if p_map == 'triangle':
        index = 0
        
        for i in range(num_qubits // 2):
            for j in range(num_qubits // 2 - i):
                gate(params[index], wires=[i + (2 * j), i + (2 * j) + 1])
                index += 1
                
    if p_map == 'alt':
        for i in range(num_qubits // 2):
            gate(params[i], wires=[2 * i, 2 * i + 1])
            
    if p_map == 'middle':
        gate(params[0], wires=[(num_qubits - 1) // 2, (num_qubits - 1) // 2 + 1])
        
    if p_map == 'circular':
        for i in range(num_qubits):
            gate(params[i], wires=[i, (i + 1) % num_qubits])
            
    if p_map == 'circular_rev':
        for i in range(num_qubits):
            gate(params[i], wires=[(i + 1) % num_qubits, i])


def hadamard_layer(num_qubits, _):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        

def c_layer(gate, c_map, num_qubits, _):
    gate = {'cz': qml.CZ, 'cy': qml.CY, 'cx': qml.CNOT}[gate]
    
    if c_map == 'full':
        for i in range(num_qubits - 1):
            for j in range(num_qubits - i - 1):
                gate(wires=[i, i + j + 1])
                
    if c_map == 'full_rev':
        for i in range(num_qubits - 1):
            for j in range(num_qubits - i - 1):
                gate(wires=[i, i + j + 1])
        
    if c_map == 'linear':
        for i in range(num_qubits - 1):
            gate(wires=[i, i + 1])
                
    if c_map == 'linear_adj':
        for i in range(num_qubits - 2, -1, -1):
            gate(wires=[i, i + 1])            
                
    if c_map == 'linear_rev':
        for i in range(num_qubits - 1):
            gate(wires=[num_qubits - i - 1, num_qubits - i - 2]) 
                
    if c_map == 'linear_rev_adj':
        for i in range(num_qubits - 1):
            gate(wires=[i + 1, i])

    if c_map == 'circular':
        for i in range(num_qubits):
            gate(wires=[i, (i + 1) % num_qubits])
            
    if c_map == 'circular_adj':
        for i in range(num_qubits - 1, -1 , -1):
            gate(wires=[i, (i + 1) % num_qubits])
            
    if c_map == 'circular_rev':
        for i in range(num_qubits):
            gate(wires=[(i + 1) % num_qubits, i])        
            
    if c_map == 'triangle':      
        for i in range(num_qubits // 2):
            for j in range(num_qubits // 2 - i):
                gate(wires=[i + (2 * j), i + (2 * j) + 1])
                
    if c_map == 'alt':
        for i in range(num_qubits // 2):
            gate(wires=[2 * i, 2 * i + 1])
            
    if c_map == 'middle':
        gate(wires=[(num_qubits - 1) // 2, (num_qubits - 1) // 2 + 1])
        

def torch_qnn_generator(dev, num_qubits, layers, layer_extra_params, weights_bounds, inputs_bounds, 
                        measured_qubits, ret_type='exp'):
    @qml.qnode(dev, interface='tf')
    def torch_qnn(inputs, weights): 
        for i, layer in enumerate(layers):
            if weights_bounds[i] == weights_bounds[i + 1]:
                data_in = inputs[inputs_bounds[i]: inputs_bounds[i + 1]]
            else:
                data_in = weights[weights_bounds[i]: weights_bounds[i + 1]]   

            layer(*layer_extra_params[i], data_in)     
        
        if ret_type == 'exp':
            return [qml.expval(qml.PauliZ(wires=i)) for i in measured_qubits]
        elif ret_type == 'state':
            return qml.state()
        else:
            return qml.probs(range(num_qubits))
    
    return torch_qnn


def get_inner_prod_circ(dev, num_qubits, layers, layer_extra_params, weights_bounds, inputs_bounds, measured_qubits):    
    new_layers = layers + layers[::-1]
    new_layer_extra_params = copy.deepcopy(layer_extra_params) + copy.deepcopy(layer_extra_params)[::-1]
    new_weights_bounds = weights_bounds.copy()
    new_inputs_bounds = inputs_bounds.copy()
    
    for i in range(1, len(weights_bounds)):
        new_weights_bounds.append(new_weights_bounds[-1] + weights_bounds[-i] - weights_bounds[-(i + 1)])
        new_inputs_bounds.append(new_inputs_bounds[-1] + inputs_bounds[-i] - inputs_bounds[-(i + 1)])
        
    for i in range(len(layers), 2 * len(layers)):
        if new_layers[i] == c_layer:
            new_layer_extra_params[i][1] += '_adj'

    inner_prod_circ = torch_qnn_generator(dev, num_qubits, new_layers, 
                                       new_layer_extra_params, new_weights_bounds, new_inputs_bounds, measured_qubits, 'prob')
    
    def get_adj_params(params):
        adj_params = -1 * params.numpy()

        for i in range(len(layers)):
            if layers[i] in [rx_layer, rz_layer, ry_layer]:
                temp = np.copy(adj_params[weights_bounds[i]:weights_bounds[i + 1]])[::-1]

                for j in range(weights_bounds[i], weights_bounds[i + 1]):
                    adj_params[j] = temp[j - weights_bounds[i]]
        return tf.constant(adj_params[::-1], dtype=tf.float64)
    
    return inner_prod_circ, get_adj_params


def generate_random_circ(dev, num_embeds, num_var_layers, ratio_cx, data_dim, num_qubits, seeds):
    rots = [qml.RY, qml.RX]

    var_layer_dist = [0]
    weights_shape = (num_var_layers, num_qubits)
    avg_layers_btw_encodings = num_var_layers / (num_embeds + 1)
    
    for i in range(num_embeds):
        upper_lim = 1 + num_var_layers - (num_embeds - i) - sum(var_layer_dist)
        probs = []
        
        for i in range(1, upper_lim):
            probs.append(1 / np.sqrt(np.abs(i - avg_layers_btw_encodings) + 1))
            
        probs = np.array(probs) / np.sum(probs)
        
        var_layer_dist.append(np.random.choice(range(1, upper_lim), p=probs))
        
    var_layer_dist.append(num_var_layers - sum(var_layer_dist))
    
    @qml.qnode(dev, interface='tf')
    def circ(data, weights):
        for i in range(num_embeds):  
            data = np.copy(data).reshape(num_embeds, data_dim)
            weights_beg = var_layer_dist[i]
            weights_end = var_layer_dist[i + 1]
            
            RandomLayers(weights=weights[weights_beg:weights_end], imprimitive=qml.CNOT, rotations=rots, 
                         wires=range(num_qubits), seed=seeds[i], ratio_imprim=ratio_cx)
            
            RandomLayers(weights=data[i:i + 1], imprimitive=qml.CNOT, rotations=rots, wires=range(num_qubits), 
                         seed=seeds[-1], ratio_imprim=0)
            
        RandomLayers(weights=weights[weights_end:], imprimitive=qml.CNOT, rotations=rots, wires=range(num_qubits), 
                     seed=seeds[-2], ratio_imprim=ratio_cx)    
        
        return qml.expval(qml.PauliZ(0))    
    
    return circ, var_layer_dist, weights_shape


def generate_random_layer_circ(dev, num_qubits, num_embeds, num_vars, ent_prob=1, var_ent_prob=0.3):
    nc = np.random.choice
    
    def get_num_params(layer_props):
        params_mapping = {
            'linear': num_qubits - 1,
            'linear_rev': num_qubits - 1,
            'circular': num_qubits,
            'circular_rev': num_qubits,
            'full': num_qubits * (num_qubits - 1),
            'full_rev': num_qubits * (num_qubits - 1),
            'triangle': ((num_qubits // 2) * (num_qubits // 2 + 1)) // 2,
            'middle': 1,
            'alt': num_qubits // 2
        }
        
        if layer_props[0] in [rx_layer, rz_layer, ry_layer]:
            return num_qubits
        elif layer_props[0] in [c_layer, hadamard_layer]:
            return 0
        else:
            return params_mapping[layer_props[1][1]]

    embed_choices = ['ry', 'rx', 'rz', 'double_pauli', 'cr']
    var_choices = ['ry', 'rx', 'rz', 'cr', 'double_pauli', 'h', 'c']
    var_probs = [(1 - var_ent_prob) / 12 for i in range(3)] + [(1 - var_ent_prob) / 4 for i in range(3)] + [var_ent_prob]
    cr_choices = ['crz', 'cry', 'crx']
    pauli_choices = ['xx', 'yy', 'zz']
    c_choices = ['cz', 'cx']
    ent_choices = ['linear', 'linear_rev', 'circular', 'circular_rev', 'full', 'full_rev', 'triangle', 'middle', 'alt']
    
    layer_mapping = {
        'ry': ry_layer,
        'rx': rx_layer,
        'rz': rz_layer,
        'cr': cr_layer,
        'c': c_layer,
        'double_pauli': double_pauli_layer,
        'h': hadamard_layer
    }
    
    var_layer_dist = []
    avg_layers_btw_encodings = num_vars / (num_embeds + 1)
    
    for i in range(num_embeds):
        upper_lim = 1 + num_vars - (num_embeds - i) - sum(var_layer_dist)
        probs = []
        
        for j in range(1, upper_lim):
            probs.append(1 / np.sqrt(np.abs(j - avg_layers_btw_encodings) + 1e-1))
            
        probs = np.array(probs) / np.sum(probs)
        
        var_layer_dist.append(np.random.choice(range(1, upper_lim), p=probs))
        
    var_layer_dist.append(num_vars - sum(var_layer_dist))   

    var_layers = []
    
    for i in var_layer_dist:
        var_layers.append(np.random.choice(var_choices, i, p=var_probs))
        
    embed_layers = np.random.choice(embed_choices, num_embeds)
    
    layers = []
    layer_extra_params = []
    weights_bounds = [0]
    inputs_bounds = [0]
    
    for i in range(len(var_layers)):
        for j in range(len(var_layers[i])):
            var_layer_name = layer_mapping[var_layers[i][j]]
            ent_layer_props = [c_layer, [nc(c_choices), nc(ent_choices), num_qubits]]
            
            layers.append(var_layer_name)
            
            if var_layers[i][j] == 'cr':
                layer_params = [nc(cr_choices), nc(ent_choices), num_qubits]
            elif var_layers[i][j] == 'c':
                layer_params = [nc(c_choices), nc(ent_choices), num_qubits]
            elif var_layers[i][j] == 'double_pauli':
                layer_params = [nc(pauli_choices), nc(ent_choices), num_qubits]
            elif var_layers[i][j] == 'h':
                layer_params = [num_qubits]
            else:
                layer_params = [None]
        
            layer_extra_params.append(layer_params)
            weights_bounds.append(weights_bounds[-1] + get_num_params([var_layer_name, layer_params]))
            inputs_bounds.append(inputs_bounds[-1])
            
            if np.random.sample() < ent_prob:
                layers.append(ent_layer_props[0])
                layer_extra_params.append(ent_layer_props[1])  
                weights_bounds.append(weights_bounds[-1])
                inputs_bounds.append(inputs_bounds[-1])
            
        if not i == len(var_layers) - 1:
            enc_layer_name = layer_mapping[embed_layers[i]]   
            ent_layer_props = [c_layer, [nc(c_choices), nc(ent_choices), num_qubits]]
            
            layers.append(enc_layer_name)
            
            if embed_layers[i] == 'cr':
                layer_params = [nc(cr_choices), np.random.choice(['circular', 'circular_rev']), num_qubits]
            elif embed_layers[i] =='double_pauli':
                layer_params = [nc(pauli_choices), np.random.choice(['circular', 'circular_rev']), num_qubits]
            else:
                layer_params = [None]            

            layer_extra_params.append(layer_params)
            inputs_bounds.append(inputs_bounds[-1] + get_num_params([enc_layer_name, layer_params]))
            weights_bounds.append(weights_bounds[-1])
    
            if np.random.sample() < ent_prob:            
                layers.append(ent_layer_props[0])
                layer_extra_params.append(ent_layer_props[1])  
                weights_bounds.append(weights_bounds[-1])
                inputs_bounds.append(inputs_bounds[-1])
            
    return layers, layer_extra_params, weights_bounds, inputs_bounds


def get_rand_circ_params(dir_path):
    mapping = {
        ry_layer: 'ry',
        rx_layer: 'rx',
        rz_layer: 'rz',
        c_layer: 'c',
        cr_layer: 'cr',
        double_pauli_layer: 'double_pauli',
        hadamard_layer: 'h'
    }

    inv_mapping = {mapping[k]: k for k in mapping.keys()}
    num_qubits = 0

    vals = np.genfromtxt(dir_path + '/layers.txt', dtype='str')
    layers = [inv_mapping[l] for l in vals]
    
    layer_params = open(dir_path + '/layer_extra_params.txt', 'r')
    layer_extra_params = []
        
    layer_extra_params = [j[1:-1].replace(' ', '').replace("'", '').split(',') 
                          if '[' in j else j.split(',') for j in layer_params.read().split('\n')[:-1]]
    
    for l in layer_extra_params:
        if len(l) == 1:
            if l[0] == 'None':
                l[0] = None
            elif l[0].isnumeric():
                l[0] = int(l[0])
        elif len(l) == 3:
            l[2] = int(l[2])
            num_qubits = l[2]
            
    weights_bounds = list(np.genfromtxt(dir_path + '/weights_bounds.txt').astype(int))
    inputs_bounds = list(np.genfromtxt(dir_path + '/inputs_bounds.txt').astype(int))
    
    return layers, layer_extra_params, weights_bounds, inputs_bounds, num_qubits

def get_metric_circ_at_point(layers, layer_params, weights_bounds, inputs_bounds, num_qubits, dev, measured_qubits, 
                             ret_type, sample):
    @qml.qnode(dev, interface='tf')
    def torch_qnn(weights): 
        for i, layer in enumerate(layers):
            if weights_bounds[i] == weights_bounds[i + 1]:
                data_in = sample[inputs_bounds[i]: inputs_bounds[i + 1]]
            else:
                data_in = weights[weights_bounds[i]: weights_bounds[i + 1]]   

            layer(*layer_extra_params[i], data_in)     
        
        if ret_type == 'exp':
            return [qml.expval(qml.PauliZ(wires=i)) for i in measured_qubits]
        elif ret_type == 'state':
            return qml.state()
        else:
            return qml.probs(range(num_qubits))
    
    return torch_qnn  