import qiskit
import pennylane as qml

import qiskit.providers.aer.noise as noise
from qiskit import IBMQ

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
    
    return noise_model, config['basis_gates'], config['coupling_map']