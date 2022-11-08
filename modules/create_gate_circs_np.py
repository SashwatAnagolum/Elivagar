import numpy as np

import torch
import torchquantum as tq
import torchquantum.functional as tqf

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
                              cxz_prob=0.2, pauli_prob=0.1):
    circ_gates = []
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
    max_params = num_embed_gates + num_var_params
    param_indices = []
    
    circ_params = 0
    curr_index = 0
    
    while circ_params < max_params:
        gate_qubits = np.random.choice(qubit_choices, p=probs)
        circ_gates.append(np.random.choice(gate_choices[gate_qubits], p=gate_probs[gate_qubits]))
        gate_params.append(np.random.choice(num_qubits, gate_qubits + 1, False))
        
        if circ_gates[-1] not in ['cx', 'cz']:
            circ_params += 1
            param_indices.append(curr_index)
            
        curr_index += 1
    
    embeds_indices = np.random.choice(param_indices, num_embed_gates, False)
    
    for i in range(len(circ_gates)):
        if circ_gates[i] in ['cx', 'cz']:
            inputs_bounds.append(inputs_bounds[-1])
            weights_bounds.append(weights_bounds[-1])
        else:
            if i in embeds_indices:
                inputs_bounds.append(inputs_bounds[-1] + 1)
                weights_bounds.append(weights_bounds[-1])
            else:
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1] + 1)
                
    return circ_gates, gate_params, inputs_bounds, weights_bounds


class TQCirc(tq.QuantumModule):
    def __init__(self, gates, gate_params, inputs_bounds, weights_bounds, num_qubits, use_softmax=False):            
        super().__init__()
        
        trainable_mapping = {
            'ry': tq.operators.RY,
            'rx': tq.operators.RX,
            'rz': tq.operators.RZ,
            'cry': tq.operators.CRY,
            'crz': tq.operators.CRZ,
            'crx': tq.operators.CRX
        }  

        non_trainable_mapping = {
            'ry': tqf.ry,
            'rx': tqf.rx,
            'rz': tqf.rz,
            'cry': tqf.cry,
            'crz': tqf.crz,
            'crx': tqf.crx,
            'cx': tqf.cnot,
            'cz': tqf.cz,
            'rzz': tqf.rzz,
            'h': tqf.hadamard
            
        } 
        
        self.n_wires = num_qubits
        self.device = tq.QuantumDevice(n_wires=self.n_wires)
        self.use_softmax = use_softmax

        self.measure = tq.MeasureAll(tq.PauliZ)

        self.embed_gates = []
        self.var_gates = []
        self.ent_gates = []
        self.gate_params = []
        self.embed_flags = []
        self.param_flags = []
        self.gate_wires = []
        
        self.num_gates = len(gates)
        
        for i in range(len(gates)):
            self.gate_wires.append([int(j) for j in gate_params[i]])
            
            if weights_bounds[i] != weights_bounds[i + 1]:                
                self.var_gates.append(trainable_mapping[gates[i]](has_params=True, trainable=True))
                self.embed_flags.append(0)
                self.param_flags.append(1)
            else:
                self.param_flags.append(0)
                
                if inputs_bounds[i] != inputs_bounds[i + 1]:
                    self.embed_gates.append(non_trainable_mapping[gates[i]])
                    self.gate_params.append(inputs_bounds[i])
                    self.embed_flags.append(1)
                else:
                    self.ent_gates.append(non_trainable_mapping[gates[i]])
                    self.embed_flags.append(0)
                    
        self.var_gates = torch.nn.ModuleList(self.var_gates)

    def forward(self, x):
        emb_ind = 0
        ent_ind = 0
        var_ind = 0
        
        self.device.reset_states(x.shape[0])

        for i in range(self.num_gates):
            if self.embed_flags[i]:
                self.embed_gates[emb_ind](self.device, wires=self.gate_wires[i],
                    params=x[:, self.gate_params[emb_ind]])
                emb_ind += 1
            elif self.param_flags[i]:
                self.var_gates[var_ind](self.device, wires=self.gate_wires[i])
                var_ind += 1
            else:
                self.ent_gates[ent_ind](self.device, wires=self.gate_wires[i])
                ent_ind += 1
            
        meas = self.measure(self.device)

        if self.use_softmax:
            meas = torch.nn.functional.log_softmax(meas, 1)
        
        return meas


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