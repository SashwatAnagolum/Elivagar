dataset_circuit_hyperparams = dict()
gatesets = dict()

dataset_names = [
    'moons', 'bank', 'mnist_2', 'fmnist_4',
    'fmnist_2', 'vowel_2', 'vowel_4', 'mnist_4',
    'mnist_10_8', 'mnist_10_6', 'mnist_10_6_nll',
    'mnist_2_fullsize', 'fmnist_2_fullsize',
    'mnist_4_fullsize', 'fmnist_4_fullsize',
    'mnist_10'
]

circuit_params = [16, 20, 20, 24, 32, 32, 40, 40, 120, 60, 72, 20, 32, 40, 24, 72]
circuit_embeds = [4, 8, 16, 16, 16, 10, 10, 16, 64, 36, 36, 16, 16, 16, 16, 36]
circuit_qubits = [2, 4, 4, 4, 4, 4, 4, 4, 8, 6, 10, 4, 4, 4, 4, 6]
num_data_reps = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
num_meas_qubits = [1, 1, 1, 2, 1, 1, 2, 2, 4, 4, 10, 1, 1, 2, 2, 4]
num_embed_layers_angle_iqp = [2, 2, 4, 4, 4, 3, 3, 4, 8, 6, 4, 4, 4, 4, 4, 6]
num_embed_layers_amp = [1 for i in range(16)]
num_var_layers = [8, 5, 5, 6, 8, 8, 10, 10, 15, 10, 8, 5, 8, 10, 6, 12]
num_classes = [2, 2, 2, 4, 2, 2, 4, 4, 10, 10, 10, 2, 2, 4, 4, 10]

for i in range(len(dataset_names)):
    dataset_circuit_hyperparams[dataset_names[i]] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_params'] = circuit_params[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embeds'] = circuit_embeds[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_qubits'] = circuit_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_data_reps'] = num_data_reps[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_meas_qubits'] = num_meas_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_var_layers'] = num_var_layers[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers'] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['angle'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['iqp'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['amp'] = num_embed_layers_amp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_classes'] = num_classes[i]

gateset_names = ['rxyz_cz', 'rzx_rxx', 'ibm_basis', 'rigetti_aspen_m2_basis', 'oqc_lucy_basis']
gateset_gates = [[['rx', 'ry', 'rz'], ['cz']], [[], ['rzx', 'rxx']], [['rz', 'sx', 'x'], ['cx']], 
                [[], []], [[], []]]

gateset_param_nums = [[[1, 1, 1], [0]], [[], [1, 1]], [[1, 0, 0], [0]], 
                [[], []], [[], []]]

for i in range(len(gateset_names)):
    gatesets[gateset_names[i]] = (gateset_gates[i], gateset_param_nums[i])
