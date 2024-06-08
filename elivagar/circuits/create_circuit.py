import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import pennylane as qml

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction
from qiskit.circuit.library import XXPlusYYGate
from torchquantum.functional import gate_wrapper
from torchquantum.operators import Operation
from tc.tc_fc import TTLinear
from braket.circuits import Circuit


def gpi_matrix(params):
    phi = params.type(torch.complex64)[:, 0]

    matrix = torch.zeros(
        (2, 2), device=params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)  

    matrix[:, 0, 1] = torch.exp(-1j * phi)
    matrix[:, 1, 0] = torch.exp(1j * phi)

    return matrix.squeeze(0) 


def gpi2_matrix(params):
    phi = params.type(torch.complex64)[:, 0]

    matrix = torch.eye(
        2, device=params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)  

    matrix[:, 0, 1] = -1j * torch.exp(-1j * phi)
    matrix[:, 1, 0] = -1j * torch.exp(1j * phi)

    matrix *= 1 / (2 ** 0.5)

    return matrix.squeeze(0)


def ms_matrix(params):
    phi_0 = params[:, 0].type(torch.complex64)
    phi_1 = params[:, 1].type(torch.complex64)
    theta = params[:, 2].type(torch.complex64)

    cos_theta = torch.cos(theta * 0.5)
    sin_theta = torch.sin(theta * 0.5)
    phis_sum_exp = -1j * torch.exp(1j * (phi_0 + phi_1))
    phis_diff_exp = -1j * torch.exp(1j * (phi_0 - phi_1))

    matrix = torch.zeros(
        (4, 4), params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)

    matrix[:, 0, 0] = cos_theta
    matrix[:, 1, 1] = cos_theta
    matrix[:, 2, 2] = cos_theta
    matrix[:, 3, 3] = cos_theta

    matrix[:, 0, 3] = phis_sum_exp * sin_theta
    matrix[:, 3, 0] = phis_sum_exp * sin_theta

    matrix[:, 1, 2] = phis_diff_exp * sin_theta
    matrix[:, 2, 1] = phis_diff_exp * sin_theta

    return matrix.squeeze(0)


def ecr_matrix(params):
    matrix = torch.tensor(
        [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]],
        dtype=torch.complex64
    ) * (1 / np.sqrt(2))
    
    return matrix.squeeze(0)


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


def gpi(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='gpi', mat=gpi_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def gpi2(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='gpi2', mat=gpi2_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def ms(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='ms', mat=ms_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def ecr(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='ecr', mat=ecr_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)


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
    

class GPI(Operation):
    num_params = 1
    num_wires = 1
    func = staticmethod(gpi)
    
    @classmethod
    def _matrix(cls, params):
        return gpi_matrix(params)
    

class GPI2(Operation):
    num_params = 1
    num_wires = 1
    func = staticmethod(gpi2)
    
    @classmethod
    def _matrix(cls, params):
        return gpi2_matrix(params)
    

class MS(Operation):
    num_params = 3
    num_wires = 2
    func = staticmethod(ms)
    
    @classmethod
    def _matrix(cls, params):
        return ms_matrix(params)


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

    
def add_amp_encoding(circ, data):
    num_qubits = circ.num_qubits
    
    state = np.zeros(2 ** num_qubits)
    state[:len(data)] = data
    
    state /= np.sqrt(np.sum(np.power(data, 2)))
    
    circ.initialize(state, [i for i in range(num_qubits)])


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
            data_in = []
            
            if inputs_bounds[i] != inputs_bounds[i + 1]:
                mapping[gate](*inputs[inputs_bounds[i]: inputs_bounds[i + 1]], wires=gate_params[i]) 
            elif weights_bounds[i] != weights_bounds[i + 1]:
                mapping[gate](*weights[weights_bounds[i]: weights_bounds[i + 1]], wires=gate_params[i]) 
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
    
    return torch_qnn


def create_batched_gate_circ(dev, gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, ret_type='exp'):
    mapping = {
        'ry': qml.RY,
        'rx': qml.RX,
        'rz': qml.RZ,
        'cx': qml.CNOT,
        'ecr': qml.ECR,
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
        'cphaseshift': qml.CPhase,
        'rxy': qml.IsingXY,
        'xy': qml.IsingXY,
        'v': qml.SX
    }   
    
    @qml.batch_params(all_operations=True)
    @qml.qnode(dev, interface=None, diff_method=None)
    def batched_qnn(inputs, weights): 
        for i, gate in enumerate(gates):            
            data_in = []
            data_in.append(inputs[:, inputs_bounds[i]: inputs_bounds[i + 1]])
            data_in.append(weights[:, weights_bounds[i]: weights_bounds[i + 1]])

            data_in = np.concatenate(data_in, 1)
            
            if data_in.shape[1] == 1:
                data_in = data_in.flatten()
            
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


def create_qiskit_circ(gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, num_qubits, unbound=False):
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
        'rzz': lambda circ, data, qubit_inds: circ.rzz(data[0], qubit_inds[0], qubit_inds[1]),
        'zz': lambda circ, data, qubit_inds: circ.rzz(data[0], qubit_inds[0], qubit_inds[1]),
        'rxx': lambda circ, data, qubit_inds: circ.rxx(data[0], qubit_inds[0], qubit_inds[1]),
        'h': lambda circ, data, qubit_inds: circ.h(qubit_inds[0]),
        's': lambda circ, data, qubit_inds: circ.s(qubit_inds[0]),
        'x': lambda circ, data, qubit_inds: circ.x(qubit_inds[0]),
        'y': lambda circ, data, qubit_inds: circ.y(qubit_inds[0]),
        'z': lambda circ, data, qubit_inds: circ.z(qubit_inds[0]),
        'sx': lambda circ, data, qubit_inds: circ.sx(qubit_inds[0]),
        'rxy': lambda circ, data, qubit_inds: circ.append(CircuitInstruction(XXPlusYYGate(data[0]), qubit_inds, [])),
        'xy': lambda circ, data, qubit_inds: circ.append(CircuitInstruction(XXPlusYYGate(data[0]), qubit_inds, [])),
        'cp': lambda circ, data, qubit_inds: circ.cp(data[0], qubit_inds[0], qubit_inds[1]),
        'rzx': lambda circ, data, qubit_inds: circ.rzx(data[0], qubit_inds[0], qubit_inds[1]),
        'amp_enc': lambda circ, data, qubit_inds: add_amp_encoding(circ, data),
        'cphaseshift': lambda circ, data, qubit_inds: circ.cp(data[0], qubit_inds[0], qubit_inds[1]),
        'ecr': lambda circ, data, qubit_inds: circ.ecr(qubit_inds[0], qubit_inds[1]),
        'v': lambda circ, data, qubit_inds: circ.sx(qubit_inds[0])
    }
    
    circuit = QuantumCircuit(num_qubits, len(measured_qubits))
    input_params = [Parameter('x_{}'.format(i)) for i in range(inputs_bounds[-1])]
    var_params = [Parameter('t_{}'.format(i)) for i in range(weights_bounds[-1])]
    
    for i, gate in enumerate(gates):
        data_in = []
        data_in.append(input_params[inputs_bounds[i]: inputs_bounds[i + 1]])
        data_in.append(var_params[weights_bounds[i]: weights_bounds[i + 1]])

        data_in = data_in[0] + data_in[1]
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
    
    if unbound:
        return circuit
    else:
        return qiskit_qnn


def create_braket_gate_circ(gates, gate_params, inputs_bounds, weights_bounds, num_qubits, 
                            qubit_mapping=None, verbatim_box=False):
    mapping = {
        'ry': lambda c, ind, theta: c.ry(ind[0], theta[0]),
        'rx': lambda c, ind, theta: c.rx(ind[0], theta[0]),
        'rz': lambda c, ind, theta: c.rz(ind[0], theta[0]),
        'cx': lambda c, ind, theta: c.cnot(ind[0], ind[1]),
        'cz': lambda c, ind, theta: c.cz(ind[0], ind[1]),
        'ryy': lambda c, ind, theta: c.yy(ind[0], ind[1], theta[0]),
        'rzz': lambda c, ind, theta: c.zz(ind[0], ind[1], theta[0]),
        'rxx': lambda c, ind, theta: c.xx(ind[0], ind[1], theta[0]),
        'h': lambda c, ind, theta: c.h(ind[0]),
        's': lambda c, ind, theta: c.s(ind[0]),
        'x': lambda c, ind, theta: c.x(ind[0]),
        'y': lambda c, ind, theta: c.y(ind[0]),
        'z': lambda c, ind, theta: c.z(ind[0]),
        'cp': lambda c, ind, theta: c.cphaseshift(ind[0], ind[1], theta[0]),
        'rxy': lambda c, ind, theta: c.xy(ind[0], ind[1], theta[0]),
        'gpi': lambda c, ind, theta: c.gpi(ind[0], theta[0]),
        'gpi2': lambda c, ind, theta: c.gpi2(ind[0], theta[0]),
        'ms': lambda c, ind, theta: c.ms(ind[0], ind[1], theta[0], theta[1]),
        'v': lambda c, ind, theta: c.v(ind[0]),
        'ecr': lambda c, ind, theta: c.ecr(ind[0], ind[1]),
        'xy': lambda c, ind, theta: c.xy(ind[0], ind[1], theta[0]),
        'cphaseshift': lambda c, ind, theta: c.cphaseshift(ind[0], ind[1], theta[0])
    }   
    
    if np.all(qubit_mapping == None):
        qubit_mapping = np.array([i for i in range(num_qubits)])
    else:
        qubit_mapping = np.array(qubit_mapping)
    
    def generate_braket_circ(inputs, weights): 
        braket_circ = Circuit()
        
        for i, gate in enumerate(gates):
            data_in = []
            data_in.append(inputs[inputs_bounds[i]: inputs_bounds[i + 1]])
            data_in.append(weights[weights_bounds[i]: weights_bounds[i + 1]])
            data_in = np.concatenate(data_in, 0) 
            
            mapping[gate](braket_circ, qubit_mapping[gate_params[i]], data_in)

        if verbatim_box:
            braket_circ = Circuit().add_verbatim_box(braket_circ)
                
        return braket_circ

    return generate_braket_circ


class TQCirc(tq.QuantumModule):
    def __init__(self, gates, gate_params, inputs_bounds,
                 weights_bounds, num_qubits, use_softmax=False,
                 quantize=False, noise_strength=0.05, use_tt=False,
                 tt_input_size=None, tt_ranks=None,
                 tt_output_size=None):            
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
            'cphaseshift': CPhase,
            'rxy': RXY,
            'xy': RXY,
            'gpi': GPI,
            'gpi2': GPI2,
            'ms': MS
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
            'ecr': ecr,
            'v': tqf.sx,
            'cp': cphase,
            'cphaseshift': cphase,
            'rxy': rxy,
            'xy': rxy,
            'amp_enc': tq.StateEncoder(),
            'gpi': gpi,
            'gpi2': gpi2,
            'ms': ms
        }
        
        self.n_wires = num_qubits
        self.device = tq.QuantumDevice(n_wires=self.n_wires)
        self.use_softmax = use_softmax
        self.quantize = quantize
        
        if quantize:
            self.normalizer = torch.nn.BatchNorm1d(num_qubits, track_running_stats=False)
            self.noise_injector = lambda x: x + noise_strength * torch.randn(x.shape)
        else:
            self.normalizer = lambda x: x
            self.noise_injector = lambda x: x
        
        if use_tt:
            self.tt_layer = TTLinear(
                inp_modes=tt_input_size,
                out_modes=tt_output_size,
                tt_rank=tt_ranks
            )
        else:
            self.tt_layer = torch.nn.Identity()
        
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.embed_gates = []
        self.var_gates = []
        self.ent_gates = []
        self.gate_params = []
        self.embed_flags = []
        self.param_flags = []
        self.gate_wires = []
        self.gate_list = gates
        self.inject_noise = False
        
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

    def set_noise_injection(self, inject_noise):
        self.inject_noise = inject_noise
        
    def forward(self, x):
        emb_ind = 0
        ent_ind = 0
        var_ind = 0
        
        if self.inject_noise:
            x = self.noise_injector(x)
        
        self.device.reset_states(x.shape[0])

        x = self.tt_layer(x)
        
        for i in range(self.num_gates):
            if self.embed_flags[i]:
                if self.gate_list[i] == 'amp_enc':
                    self.embed_gates[emb_ind](self.device, x)
                    continue 
                
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

        meas = self.normalizer(meas)
            
        return meas
