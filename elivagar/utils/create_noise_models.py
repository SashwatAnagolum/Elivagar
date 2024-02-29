import qiskit
import pennylane as qml
import pickle as pkl
import os
import qiskit.providers.aer.noise as noise

from qiskit import IBMQ
import qiskit.circuit.library as gate_lib

from elivagar.circuits.device_aware import (
    extract_properties_from_braket_device, get_braket_device_qubit_edge_info,
    get_braket_device_1q_succ_probs
)


def convert_braket_properties_to_qiskit_noise_model(device_properties_path, symmetric=False,
                                                    device_properties=None):
    """
    Use a braket device properties file to get information for a Qiskit backend noise model.
    """
    gate_names_to_qiskit_gate_names = {
        'xy': gate_lib.XXPlusYYGate(1).name,
        'rxy': gate_lib.XXPlusYYGate(1).name,
        'cphaseshift': gate_lib.CPhaseGate(1).name,
        'cz': gate_lib.CZGate().name,
        'rx': gate_lib.RXGate(1).name,
        'rz': gate_lib.RZGate(1).name,
        'v': gate_lib.SXGate().name,
        'x': gate_lib.XGate().name,
        'ecr': gate_lib.ECRGate().name,
        'cx': gate_lib.CXGate().name,
        'sx': gate_lib.SXGate().name
    }
    
    basis_gates_to_qiskit_gate_errors = {
        'v': ['v', 'sx'],
        'ecr': ['ecr', 'cx']
    }
    
    if device_properties_path is None:
        return None, None
    
    noise_model = noise.NoiseModel()
    
    if device_properties is None:
        dev_properties = pkl.load(open(device_properties_path, 'rb'))
    else:
        dev_properties = device_properties
    
    (
        num_device_qubits, connectivity, t1_times,
        t2_times, readout_probs, basis_gates,
        gate_num_params, gate_times,
        two_qubit_gate_succ_probs, one_qubit_succ_probs, qubit_indices
    ) = extract_properties_from_braket_device(None, dev_properties, symmetric)
    
    index_map = {qubit: idx for idx, qubit in enumerate(qubit_indices)}
    
    coupling_map = []
    
    for qubit in qubit_indices:
        coupling_map += [[index_map[qubit], index_map[adj_qubit]] for adj_qubit in connectivity[qubit]]
        
    # basis gates
    for basis_gate in basis_gates[0] + basis_gates[1]:
        noise_model.add_basis_gates(gate_names_to_qiskit_gate_names[basis_gate])
        
    # 1q errors
    for qubit in qubit_indices:
        for gate in basis_gates[0]:
            if gate in basis_gates_to_qiskit_gate_errors:
                gate_list = basis_gates_to_qiskit_gate_errors[gate]
            else:
                gate_list = [gate]
                
            for errored_gate in gate_list:
                noise_model.add_quantum_error(
                    noise.errors.depolarizing_error(
                        1 - one_qubit_succ_probs[qubit][gate],
                        1
                    ),
                    [gate_names_to_qiskit_gate_names[errored_gate]],
                    [index_map[qubit]]
                )
    
    # readout errors
    for qubit in qubit_indices:
        noise_model.add_readout_error(
            noise.errors.ReadoutError(
                [
                    [readout_probs[qubit], 1 - readout_probs[qubit]],
                    [1 - readout_probs[qubit], readout_probs[qubit]]
                ]
            ),
            [index_map[qubit]]
        )
    
    # 2q errors
    for qubit in qubit_indices:
        for adj_qubit in connectivity[qubit]:           
            for gate in basis_gates[1]:
                if gate in basis_gates_to_qiskit_gate_errors:
                    gate_list = basis_gates_to_qiskit_gate_errors[gate]
                else:
                    gate_list = [gate]
                    
                for errored_gate in gate_list:
                    noise_model.add_quantum_error(
                        noise.errors.depolarizing_error(
                            1 - two_qubit_gate_succ_probs[qubit][gate][adj_qubit],
                            2
                        ),
                        [gate_names_to_qiskit_gate_names[errored_gate]],
                        [index_map[qubit], index_map[adj_qubit]]
                    )
    
    if 'v' in basis_gates[0]:
        basis_gates[0] += ['sx']
    
    if 'ecr' in basis_gates[1]:
        basis_gates[1] += ['cx']
        
    return noise_model, index_map, basis_gates[0] + basis_gates[1], coupling_map
    

def noisy_dev_from_backend(backend_name, num_qubits):
    """
    Create a pennylane device that uses a noise model based on the backend with the name passed in.
    """
    try:
        provider = IBMQ.enable_account(
            'f9be8ebe6cc0b5c9970ca5ae86acad18c1dfb3844ed12b381a458536fcbf46499d62dbb33da9a07627774441860c64ac44e76a6f27dc6f09bba7e0f2ce68e9ff')
    except:
        provider = IBMQ.load_account()
    
    backend = provider.get_backend(backend_name)
    noise_model = noise.NoiseModel.from_backend(backend)
    
    dev = qml.device('qiskit.aer', wires=num_qubits, noise_model=noise_model)
    
    return dev


def get_real_backend_dev(backend_name, num_qubits):
    """
    Get the real IBMQ backend with the backend name passed in.
    """
    try:
        provider = IBMQ.enable_account(
            'f9be8ebe6cc0b5c9970ca5ae86acad18c1dfb3844ed12b381a458536fcbf46499d62dbb33da9a07627774441860c64ac44e76a6f27dc6f09bba7e0f2ce68e9ff')
    except:
        provider = IBMQ.load_account()
        
    dev = qml.device('qiskit.ibmq', wires=num_qubits, backend=backend_name, provider=provider)
    
    return dev


def get_noise_model(device_name):
    try:
        provider = IBMQ.enable_account(
            'f9be8ebe6cc0b5c9970ca5ae86acad18c1dfb3844ed12b381a458536fcbf46499d62dbb33da9a07627774441860c64ac44e76a6f27dc6f09bba7e0f2ce68e9ff')
    except:
        provider = IBMQ.load_account()
        
    backend = provider.get_backend(device_name)
    noise_model = noise.NoiseModel.from_backend(backend)
    config = backend.configuration().to_dict()
    
    device_properties_folder = f'./device_properties/ibm'
    
    if not os.path.exists(device_properties_folder):
        os.makedirs(device_properties_folder)
    
    pkl.dump(
        (config['basis_gates'], config['coupling_map']),
        open(
            os.path.join(device_properties_folder, '{device_name}.data'),
            'wb'
        )
    )
    
    return noise_model, config['basis_gates'], config['coupling_map']