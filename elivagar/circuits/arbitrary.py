import numpy as np

def generate_random_gate_circ(num_qubits, num_embed_gates, num_var_params=None, ent_prob=0.5, 
                              cxz_prob=0.2, pauli_prob=0.1, consecutive_embeds=True):   
    circ_gates = ['cx']
    inputs_bounds = [0]
    weights_bounds = [0]
    gate_params = []

    qubit_choices = [0, 1]
    gate_choices = [['ry', 'rz', 'rx'], ['cx', 'cz', 'crz', 'crx', 'cry', 'xx', 'yy', 'zz']]
    probs = [1 - ent_prob, ent_prob]
    c_probs = [cxz_prob / (2 * ent_prob) for i in range(2)] + [(1 - ((cxz_prob + pauli_prob) / ent_prob)) / 3 for i in range(3)]
    c_probs += [pauli_prob / (3 * ent_prob) for i in range(3)]

    r_probs = [1 / 3 for i in range(3)]
    gate_probs = [r_probs, c_probs]
    gate_qubits = 1
    
    max_params = num_embed_gates + num_var_params
    
    if consecutive_embeds:
        embed_positions = np.sort(np.random.choice(max_params, num_embed_gates, False))
    else:
        consecutive = True
        
        while consecutive:
            embed_positions = np.sort(np.random.choice(max_params, num_embed_gates, False))
            diffs = [embed_positions[i + 1] - embed_positions[i] for i in range(len(embed_positions) - 1)]
            consecutive = True if 1 in diffs else False

            if num_embed_gates > num_var_params:
                consecutive = False
        
    while weights_bounds[-1] + inputs_bounds[-1] < max_params:
        flag = False
            
        while not flag:
            gate_qubits = np.random.choice(qubit_choices, p=probs)
            curr_gate = np.random.choice(gate_choices[gate_qubits], p=gate_probs[gate_qubits])

            if curr_gate == circ_gates[-1]:
                pass
            else:
                if curr_gate in ['cx', 'cz']:
                    if len(circ_gates) in embed_positions or circ_gates[-1] in ['cx', 'cz']:
                        pass
                    else:
                        flag = True
                        inputs_bounds.append(inputs_bounds[-1])
                        weights_bounds.append(weights_bounds[-1])
                else:
                    flag = True

                    if weights_bounds[-1] + inputs_bounds[-1] in embed_positions:
                        inputs_bounds.append(inputs_bounds[-1] + 1)
                        weights_bounds.append(weights_bounds[-1])
                    else:
                        inputs_bounds.append(inputs_bounds[-1])
                        weights_bounds.append(weights_bounds[-1] + 1)                        

        comp_index = 0
            
        for i in range(len(gate_params)):
            if len(gate_params[-(i + 1)]) == gate_qubits + 1:
                comp_index = -(i + 1)
                break
                    
        new_params = np.random.choice(num_qubits, gate_qubits + 1, False) 
            
        if comp_index != 0:
            while (np.all(new_params == gate_params[comp_index]) or np.all(new_params[::1] == gate_params[comp_index])):
                new_params = np.random.choice(num_qubits, gate_qubits + 1, False)
        
        curr_comp_index = 0
        
        for i in range(len(gate_params) - 1):
            if len(gate_params[-(i + 1)] == len(new_params)):
                if np.all(gate_params[-(i + 1)] == new_params) or np.all(gate_params[-(i + 1)] == new_params[::-1]):
                    curr_comp_index = -(i + 1)
                    break
        
        new_gate = curr_gate
        
        if curr_comp_index != 0:
            if inputs_bounds[-1] > inputs_bounds[-2] or weights_bounds[-1] > weights_bounds[-2]:
                new_gate = np.random.choice(gate_choices[gate_qubits], p=gate_probs[gate_qubits])

                while new_gate in ['cx', 'cz', circ_gates[curr_comp_index]]:
                    new_gate = np.random.choice(gate_choices[gate_qubits], p=gate_probs[gate_qubits])
            else:
                if circ_gates[curr_comp_index] == 'cx':
                    new_gate = 'cz'
                else:
                    new_gate = 'cx'
            
        gate_params.append(new_params)
        circ_gates.append(new_gate)

    return circ_gates[1:], gate_params, inputs_bounds, weights_bounds


def generate_true_random_gate_circ(num_qubits, num_embed_gates, num_var_params=None, ent_prob=0.5, 
                                   cxz_prob=0.2, pauli_prob=0.1, gateset=None, ind_gate_probs=None,
                                   gateset_param_nums=None):
    circ_gates = []
    inputs_bounds = [0]
    weights_bounds = [0]
    gate_params = []

    qubit_choices = [0, 1]
    
    if gateset is None:
        gate_choices = [['ry', 'rz', 'rx'], ['cx', 'cz', 'crz', 'crx', 'cry', 'rxx', 'ryy', 'rzz']]
        gate_param_nums = [[1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]]
        
        c_probs = [cxz_prob / (2 * ent_prob) for i in range(2)] + [(1 - ((cxz_prob + pauli_prob) / ent_prob)) / 3 for i in range(3)]
        c_probs += [pauli_prob / (3 * ent_prob) for i in range(3)]
        
        r_probs = [1 / 3 for i in range(3)]
        gate_probs = [r_probs, c_probs] 
    else:
        gate_choices = gateset
        gate_param_nums = gateset_param_nums

        if ind_gate_probs is None:
            num_1q = len(gateset[0])
            num_2q = len(gateset[1])
            
            gate_probs = [[1 / num_1q for i in range(num_1q)], [1 / num_2q for i in range(num_2q)]]
        else:
            gate_probs = ind_gate_probs
        
    probs = [1 - ent_prob, ent_prob]
    max_params = num_embed_gates + num_var_params
    param_indices = []
    
    circ_params = 0
    curr_index = 0
    
    while circ_params < max_params:
        gate_qubits = np.random.choice(qubit_choices, p=probs)
        chosen_gate_index = np.random.choice(len(gate_choices[gate_qubits]), p=gate_probs[gate_qubits])
        
        curr_gate_num_params = gate_param_nums[gate_qubits][chosen_gate_index]
        
        if circ_params + curr_gate_num_params <= max_params:
            circ_gates.append(gate_choices[gate_qubits][chosen_gate_index])
            gate_params.append(np.random.choice(num_qubits, gate_qubits + 1, False))
            
            circ_params += curr_gate_num_params
            param_indices += [curr_index for i in range(curr_gate_num_params)]

            curr_index += 1
     
    param_indices = np.array(param_indices)
    embed_inds = np.zeros(len(param_indices)).astype(bool)
    embed_inds[np.random.choice(param_indices, num_embed_gates, False)] = True
    var_inds = param_indices[np.invert(embed_inds)]
    embed_inds = param_indices[embed_inds]
    
    for i in range(len(circ_gates)):
        if i not in param_indices:
            inputs_bounds.append(inputs_bounds[-1])
            weights_bounds.append(weights_bounds[-1])
        else:
            if i in embed_inds:
                inputs_bounds.append(inputs_bounds[-1] + np.sum(embed_inds == i))
            else:
                inputs_bounds.append(inputs_bounds[-1])
            
            if i in var_inds:
                weights_bounds.append(weights_bounds[-1] + np.sum(var_inds == i))
            else:
                weights_bounds.append(weights_bounds[-1])                           
                
    return circ_gates, gate_params, inputs_bounds, weights_bounds


def append_adjoint_to_circuit(circ_gates, gate_params, inputs_bounds, weights_bounds):
    new_circ_gates = [i for i in circ_gates]
    new_gate_params = [[j for j in i] for i in gate_params]
    new_weights_bounds = [i for i in weights_bounds]
    new_inputs_bounds = [i for i in inputs_bounds]


def generate_random_embedding(num_qubits, gates, gate_params, inputs_bounds, weights_bounds, ent_prob=0.5):
    qubit_choices = [0, 1]
    gate_choices = [['ry', 'rz', 'rx'], ['cx', 'cz', 'crz', 'crx', 'cry', 'xx', 'yy', 'zz']]
    probs = [1 - ent_prob, ent_prob]    
    
    embed_positions = [inputs_bounds[i + 1] - inputs_bounds[i] for i in range(len(inputs_bounds) - 1)]
    embed_positions = np.argwhere(embed_positions).flatten()
    
    new_gates = []
    new_gate_params = []
    
    for i in range(len(gates)):
        if i not in embed_positions:
            new_gates.append(gates[i])
            new_gate_params.append(gate_params[i])
        else:
            flag = False
            
            while not flag:
                gate_qubits = np.random.choice(qubit_choices, p=probs)
                curr_gate = np.random.choice(gate_choices[gate_qubits])
                
                if curr_gate in ['cx', 'cz'] or curr_gate == gates[i]:
                        pass
                else:
                    flag = True                 

            new_gates.append(curr_gate)
            new_gate_params.append(np.random.choice(num_qubits, gate_qubits + 1, False))
            
    return new_gates, new_gate_params


def generate_random_variational(num_qubits, gates, gate_params, inputs_bounds, weights_bounds, ent_prob=0.5):
    qubit_choices = [0, 1]
    gate_choices = [['ry', 'rz', 'rx'], ['cx', 'cz', 'crz', 'crx', 'cry', 'xx', 'yy', 'zz']]
    probs = [1 - ent_prob, ent_prob]    
    
    var_positions = [weights_bounds[i + 1] - weights_bounds[i] for i in range(len(weights_bounds) - 1)]
    var_positions = np.argwhere(var_positions).flatten()
    
    new_gates = []
    new_gate_params = []
    
    for i in range(len(gates)):
        if i not in var_positions:
            new_gates.append(gates[i])
            new_gate_params.append(gate_params[i])
        else:
            flag = False
            
            while not flag:
                gate_qubits = np.random.choice(qubit_choices, p=probs)
                curr_gate = np.random.choice(gate_choices[gate_qubits])
                
                if curr_gate in ['cx', 'cz'] or curr_gate == gates[i]:
                        pass
                else:
                    flag = True                 

            new_gates.append(curr_gate)
            new_gate_params.append(np.random.choice(num_qubits, gate_qubits + 1, False))
            
    return new_gates, new_gate_params


def replace_embedding(old_embed_dir, new_embed_dir):
    inputs_bounds = np.genfromtxt(old_embed_dir + '/inputs_bounds.txt')
    old_gates = open(old_embed_dir + '/gates.txt').read().split('\n')
    old_gate_params = [[int(k) for k in j[1:-1].replace(',', '').split(' ')] for j in open(old_embed_dir + '/gate_params.txt').read().split('\n')[:-1]]
    
    new_gates = open(new_embed_dir + '/gates.txt').read().split('\n')
    new_gate_params = [[int(k) for k in j[1:-1].replace(',', '').split(' ')] for j in open(new_embed_dir + '/gate_params.txt').read().split('\n')[:-1]]
    
    embed_positions = [inputs_bounds[i + 1] - inputs_bounds[i] for i in range(len(inputs_bounds) - 1)]
    embed_positions = np.argwhere(embed_positions).flatten()
    
    new_inputs_bounds = np.genfromtxt(new_embed_dir + '/inputs_bounds.txt')
    new_embed_positions = [new_inputs_bounds[i + 1] - new_inputs_bounds[i] for i in range(len(new_inputs_bounds) - 1)]
    new_embed_positions = np.argwhere(new_embed_positions).flatten()
    
    for i in range(len(embed_positions)):
        old_gates[embed_positions[i]] = new_gates[new_embed_positions[i]]
        old_gate_params[embed_positions[i]] = np.array(new_gate_params[new_embed_positions[i]])

    np.savetxt(old_embed_dir + '/gates.txt', old_gates, fmt="%s")
    np.savetxt(old_embed_dir + '/gate_params.txt', np.array(old_gate_params, dtype='object'), fmt="%s")


def get_var_part_only(gates, gate_params, inputs_bounds, weights_bounds):
    embeds = [inputs_bounds[i + 1] - inputs_bounds[i] for i in range(len(inputs_bounds) - 1)]
    embeds_pos = np.argwhere(embeds).flatten()
    
    new_gates = []
    new_gate_params = []
    new_weights_bounds = [0]
    
    for i in range(len(gates)):
        if i not in embeds_pos:
            new_gates.append(gates[i])
            new_gate_params.append(gate_params[i])
            new_weights_bounds.append(weights_bounds[i + 1])
            
    return new_gates, new_gate_params, new_weights_bounds


def sample_subcircuit(gates, gate_params, inputs_bounds, weights_bounds, num_subcircuit_params, num_subcircuit_embeds):
    num_gates = len(gates)
    min_gates = num_subcircuit_params + num_subcircuit_embeds
    
    num_sel_params = 0
    num_sel_embeds = 0
    
    while (num_sel_params != num_subcircuit_params) or (num_sel_embeds != num_subcircuit_embeds):
        num_sel_params = 0
        num_sel_embeds = 0   

        num_sel_gates = np.random.randint(min_gates, num_gates)
        gate_inds = np.sort(np.random.choice(num_gates, num_sel_gates, False))
        sel_gates = np.array(gates)[gate_inds]
        
        for i in gate_inds:
            num_sel_params += weights_bounds[i + 1] - weights_bounds[i]
            num_sel_embeds += inputs_bounds[i + 1] - inputs_bounds[i]
            
        if num_sel_params >= num_subcircuit_params and num_sel_embeds >= num_subcircuit_embeds:
            enc_gate_inds = []
            var_gate_inds = []
            other_gate_inds = []
            
            for i in gate_inds:       
                if (inputs_bounds[i + 1] - inputs_bounds[i]):
                    enc_gate_inds.append(i) 
                elif (weights_bounds[i + 1] - weights_bounds[i]):
                    var_gate_inds.append(i)
                else:
                    other_gate_inds.append(i)
                    
            sel_enc_gate_inds = np.random.choice(enc_gate_inds, num_subcircuit_embeds, False)
            sel_var_gate_inds = np.random.choice(var_gate_inds, num_subcircuit_params, False)
            
            if len(other_gate_inds):
                gate_inds = np.sort(np.concatenate((sel_enc_gate_inds, sel_var_gate_inds, other_gate_inds)))
            else:
                gate_inds = np.sort(np.concatenate((sel_enc_gate_inds, sel_var_gate_inds)))

            sel_gates = np.array(gates)[gate_inds]
            
            break
        
    sel_gate_params = [gate_params[i] for i in gate_inds]
    
    sel_inputs_bounds = [0]
    sel_weights_bounds = [0]
    
    inputs_filt = []
    weights_filt = []
    
    for i in gate_inds:
        sel_inputs_bounds.append(sel_inputs_bounds[-1] + (inputs_bounds[i + 1] - inputs_bounds[i]))
        sel_weights_bounds.append(sel_weights_bounds[-1] + (weights_bounds[i + 1] - weights_bounds[i]))
        
        if (inputs_bounds[i + 1] - inputs_bounds[i]):
            inputs_filt.append(inputs_bounds[i])
            
        if (weights_bounds[i + 1] - weights_bounds[i]):
            weights_filt.append(weights_bounds[i])
        
    return sel_gates, sel_gate_params, sel_inputs_bounds, sel_weights_bounds, inputs_filt, weights_filt


def get_circ_params(dir_path):
    inputs_bounds = [int(i) for i in np.genfromtxt(dir_path + '/inputs_bounds.txt')]
    weights_bounds = [int(i) for i in np.genfromtxt(dir_path + '/weights_bounds.txt')]
    gates = open(dir_path + '/gates.txt').read().split('\n')
    
    if '[' in open(dir_path + '/gate_params.txt').read().split('\n')[0]:
        gate_params = [[int(k) for k in j[1:-1].replace(',', '').split(' ')] for j in open(dir_path + '/gate_params.txt').read().split('\n')[:-1]]
    else:
        gate_params = [[int(k) for k in j.replace(',', '').split(' ')] for j in open(dir_path + '/gate_params.txt').read().split('\n')[:-1]]
    
    gates = list(filter(lambda x: True if x != '' else False, gates))
    
    return gates, gate_params, inputs_bounds, weights_bounds