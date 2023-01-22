dataset_circuit_hyperparams = dict()
gatesets = dict()

dataset_names = ['moons', 'bank', 'mnist_2', 'fmnist_4', 'fmnist_2', 'vowel_2', 'vowel_4', 'mnist_4']
circuit_params = [16, 20, 20, 24, 32, 32, 40, 40]
circuit_embeds = [4, 8, 16, 16, 16, 10, 10, 16]
circuit_qubits = [2, 4, 4, 4, 4, 4, 4, 4]
num_data_reps = [2, 2, 1, 1, 1, 1, 1, 1]
num_meas_qubits = [1, 1, 1, 2, 1, 1, 2, 2]

for i in range(len(dataset_names)):
    dataset_circuit_hyperparams[dataset_names[i]] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_params'] = circuit_params[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embeds'] = circuit_embeds[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_qubits'] = circuit_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_data_reps'] = num_data_reps[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_meas_qubits'] = num_meas_qubits[i]

gateset_names = ['rxyz', 'rzx_xx', 'ibm_basis', 'rigetti_aspen_m2_basis', 'oqc_lucy_basis']
gateset_gates = [[['rx', 'ry', 'rz'], ['cz']], [[], ['rzx', 'rxx']], [['rz', 'sx', 'x'], ['cx']], 
                [[], []], [[], []]]

for i in range(len(gateset_names)):
    gatesets[gateset_names[i]] = gateset_gates[i]
