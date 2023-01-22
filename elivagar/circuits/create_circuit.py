import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf

from torchquantum.functional import gate_wrapper
from torchquantum.operators import Operation


def cphase_matrix(params):
    theta = params.type(torch.complex64)
    exp_theta = torch.exp(1j * theta)
    
    matrix = torch.eye(4, dtype=torch.complex64,
                         device=params.device).unsqueeze(0).repeat(exp_theta.shape[0], 1, 1)
    
    matrix[:, 3, 3] = exp_theta[:, 0]
    
    return matrix.squeeze(0)


def rxy_matrix(params):
    theta = params.type(torch.complex64)
    cos_theta = torch.cos(theta * 0.5)
    sin_theta = torch.sin(theta * 0.5)
    
    matrix = torch.eye(4, device=params.device,
                      dtype=torch.complex64).unsqueeze(0).repeat(cos_theta.shape[0], 1, 1)
    
    matrix[:, 1, 1] = cos_theta[:, 0]
    matrix[:, 2, 2] = cos_theta[:, 0]
    matrix[:, 1, 2] = -1j * sin_theta[:, 0]
    matrix[:, 2, 1] = -1j * sin_theta[:, 0]
    
    return matrix.squeeze(0)


def cphase(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='cp', mat=cphase_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)
    
    
def rxy(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='rxy', mat=rxy_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)


class RXY(Operation):
    num_params = 1
    num_wires = 2
    func = staticmethod(rxy)
    
    @classmethod
    def _matrix(cls, params):
        return rxy_matrix(params)
    
    
class CPhase(Operation):
    num_params = 1
    num_wires = 2
    func = staticmethod(cphase)
    
    @classmethod
    def _matrix(cls, params):
        return cphase_matrix(params)


def create_gate_circ(dev, gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, ret_type='exp', diff_method='best', interface='tf'):
    mapping = {
        'ry': qml.RY,
        'rx': qml.RX,
        'rz': qml.RZ,
        'cx': qml.CNOT,
        'cz': qml.CZ,
        'cry': qml.CRY,
        'crz': qml.CRZ,
        'crx': qml.CRX,
        'rot': qml.Rot,
        'meas': qml.measure,
        'h': qml.Hadamard,
        's': qml.S,
        'x': qml.PauliX,
        'y': qml.PauliY,
        'z': qml.PauliZ,
        'iqp': qml.IQPEmbedding,
        'ryy': qml.IsingYY,
        'rzz': qml.IsingZZ,
        'rxx': qml.IsingXX,
        'sx': qml.SX,
        'cp': qml.CPhase,
        'rxy': qml.IsingXY
    }   

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def torch_qnn(inputs, weights): 
        for i, gate in enumerate(gates):
            if weights_bounds[i] == weights_bounds[i + 1]:
                data_in = inputs[inputs_bounds[i]: inputs_bounds[i + 1]]
            else:
                data_in = weights[weights_bounds[i]: weights_bounds[i + 1]]   

            mapping[gate](*data_in, wires=gate_params[i])     
        
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
    
    return torch_qnn


def create_batched_gate_circ(dev, gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, ret_type='exp'):
    mapping = {
        'ry': qml.RY,
        'rx': qml.RX,
        'rz': qml.RZ,
        'cx': qml.CNOT,
        'cz': qml.CZ,
        'cry': qml.CRY,
        'crz': qml.CRZ,
        'crx': qml.CRX,
        'rot': qml.Rot,
        'meas': qml.measure,
        'h': qml.Hadamard,
        's': qml.S,
        'x': qml.PauliX,
        'y': qml.PauliY,
        'z': qml.PauliZ,
        'iqp': qml.IQPEmbedding,
        'ryy': qml.IsingYY,
        'rzz': qml.IsingZZ,
        'rxx': qml.IsingXX,
        'sx': qml.SX,
        'cp': qml.CPhase,
        'rxy': qml.IsingXY
    }   

    @qml.batch_params(all_operations=True)
    @qml.qnode(dev, interface=None, diff_method=None)
    def batched_qnn(inputs, weights): 
        for i, gate in enumerate(gates):
            is_not_param = weights_bounds[i] == weights_bounds[i + 1]
            is_not_data = inputs_bounds[i] == inputs_bounds[i + 1]
    
            if is_not_param and is_not_data:
                data_in = None
            elif is_not_param:
                data_in = inputs[:, inputs_bounds[i]: inputs_bounds[i + 1]]
            else:
                data_in = weights[:, weights_bounds[i]: weights_bounds[i + 1]]   
            
            data_in = data_in.flatten() if data_in is not None else data_in
            
            if np.any(data_in != None):    
                mapping[gate](data_in, wires=gate_params[i]) 
            else:
                mapping[gate](wires=gate_params[i])
        
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
    
    return batched_qnn


def create_qiskit_circ(gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, num_qubits):
    mapping = {
        'ry': lambda circ, data, qubit_inds: circ.ry(data[0], qubit_inds[0]),
        'rx': lambda circ, data, qubit_inds: circ.rx(data[0], qubit_inds[0]),
        'rz': lambda circ, data, qubit_inds: circ.rz(data[0], qubit_inds[0]),
        'cx': lambda circ, data, qubit_inds: circ.cx(qubit_inds[0], qubit_inds[1]),
        'cz': lambda circ, data, qubit_inds: circ.cz(qubit_inds[0], qubit_inds[1]),
        'cry': lambda circ, data, qubit_inds: circ.cry(data[0], qubit_inds[0], qubit_inds[1]),
        'crz': lambda circ, data, qubit_inds: circ.crz(data[0], qubit_inds[0], qubit_inds[1]),
        'crx': lambda circ, data, qubit_inds: circ.crx(data[0], qubit_inds[0], qubit_inds[1]),
        'ryy': lambda circ, data, qubit_inds: circ.ryy(data[0], qubit_inds[0], qubit_inds[1]),
        'rzz': lambda circ, data, qubit_inds: circ.ryy(data[0], qubit_inds[0], qubit_inds[1]),
        'rxx': lambda circ, data, qubit_inds: circ.rxx(data[0], qubit_inds[0], qubit_inds[1]),
        'h': lambda circ, data, qubit_inds: circ.h(qubit_inds[0]),
        's': lambda circ, data, qubit_inds: circ.s(qubit_inds[0]),
        'x': lambda circ, data, qubit_inds: circ.x(qubit_inds[0]),
        'y': lambda circ, data, qubit_inds: circ.y(qubit_inds[0]),
        'z': lambda circ, data, qubit_inds: circ.z(qubit_inds[0]),
        'sx': lambda circ, data, qubit_inds: circ.sx(qubit_inds[0]),
        'rxy': lambda circ, data, qubit_inds: circ.append(CircuitInstruction(XXPlusYYGate(data[0]), qubit_inds, [])),
        'cp': lambda circ, data, qubit_inds: circ.cp(data[0], qubit_inds[0], qubit_inds[1]),
        'rzx': lambda circ, data, qubit_inds: circ.rzx(data[0], qubit_inds[0], qubit_inds[1]) 
    }
    
    circuit = QuantumCircuit(num_qubits, len(measured_qubits))
    input_params = [Parameter('x_{}'.format(i)) for i in range(inputs_bounds[-1])]
    var_params = [Parameter('t_{}'.format(i)) for i in range(weights_bounds[-1])]
    
    for i, gate in enumerate(gates):
        if weights_bounds[i] == weights_bounds[i + 1]:
            data_in = input_params[inputs_bounds[i]: inputs_bounds[i + 1]]
        else:
            data_in = var_params[weights_bounds[i]: weights_bounds[i + 1]]   

        mapping[gate](circuit, data_in, gate_params[i])     
        
    for i in range(len(measured_qubits)):
        circuit.measure(measured_qubits[i], i)
    
    def qiskit_qnn(inputs, weights): 
        param_mapping = dict()
        
        for i in range(len(inputs)):
            param_mapping[input_params[i]] = inputs[i]
            
        for i in range(len(weights)):
            param_mapping[var_params[i]] = weights[i]
        
        return circuit.bind_parameters(param_mapping)
    
    return qiskit_qnn

    
class TQCirc(tq.QuantumModule):
    def __init__(self, gates, gate_params, inputs_bounds, weights_bounds, num_qubits, use_softmax=False):            
        super().__init__()
        
        trainable_mapping = {
            'ry': tq.operators.RY,
            'rx': tq.operators.RX,
            'rz': tq.operators.RZ,
            'cry': tq.operators.CRY,
            'crz': tq.operators.CRZ,
            'crx': tq.operators.CRX,
            'rxx': tq.operators.RXX,
            'ryy': tq.operators.RYY,
            'rzz': tq.operators.RZZ,
            'rzx': tq.operators.RZX,
            'cp': CPhase,
            'rxy': RXY
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
            'rxx': tqf.rxx,
            'ryy': tqf.ryy,
            'h': tqf.hadamard,
            'rzx': tqf.rzx,
            'sx': tqf.sx,
            'x': tqf.paulix,
            'cp': cphase,
            'rxy': rxy
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
