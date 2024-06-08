import numpy as np


def get_braket_device_qubit_edge_info(device=None, dev_properties=None,
                                      device_name=None):
    # Need to check the exact location for each device currently due to
    # inconsistencies in the Braket API.
    if device:
        dev_properties = device.properties

    if device_name == 'oqc_lucy':
        calibration_data = dev_properties.provider.properties
        one_qubit_data = calibration_data['one_qubit']
        two_qubit_data = calibration_data['two_qubit']
        gate_names_2q = {'ecr': 'fCX'}
        gate_names_1q = {'rz': None, 'x': 'fRB', 'v': 'fRB'}
    elif device_name == 'rigetti_aspen_m_3':
        calibration_data = dev_properties.provider.specs
        one_qubit_data = calibration_data['1Q']
        two_qubit_data = calibration_data['2Q']
        gate_names_2q = {'cphaseshift': 'fCPHASE', 'xy': 'fXY', 'cz': 'fCZ'} 
        gate_names_1q = {'rx': 'f1QRB', 'rz': None}
    elif device_name == 'ionq_harmony':
        qubit_count = dev_properties.paradigm.qubitCount
        fidelities = dev_properties.provider.fidelity
        timings = dev_properties.provider.timing

        one_qubit_data = {
            str(i): {
                '1Q': fidelities['1Q']['mean'],
                'fRO': fidelities['spam']['mean'],
                'T1': timings['T1'],
                'T2': timings['T2']
            }
            for i in range(qubit_count)
        }

        two_qubit_data = {
            f'{i}-{j}': {'2Q': fidelities['2Q']['mean']}
            for i in range(qubit_count) for j in range(qubit_count)
            if i != j
        }

        gate_names_1q = {'GPI': '1Q', 'GPI2': '1Q'}
        gate_names_2q = {'MS': '2Q'}
    else:
        raise ValueError(f'Device {device_name} not supported!')
        
    return one_qubit_data, two_qubit_data, gate_names_2q, gate_names_1q


def get_braket_device_qubit_inds(one_qubit_data):
    qubit_indices = [int(i) for i in list(one_qubit_data.keys())]
    qubit_indices.sort()
    
    return qubit_indices


def get_braket_device_connectivity(device, qubit_inds, dev_properties=None):
    if dev_properties is None:
        dev_properties = device.properties
    
    connectivity_raw = dev_properties.paradigm.connectivity.connectivityGraph
    fully_connected = False

    if 'fullyConnected' in dir(dev_properties.paradigm.connectivity):
        fully_connected = dev_properties.paradigm.connectivity.fullyConnected
    
    if fully_connected:
        connectivity_raw = {
            str(qubit): [str(q) for q in qubit_inds if q != qubit]
            for qubit in qubit_inds
        }

    keys = list(connectivity_raw.keys())

    connectivity = dict()
    
    for qubit in qubit_inds:
        connectivity[qubit] = []
        
        if str(qubit) in connectivity_raw.keys():
            connectivity[qubit] = [int(j) for j in connectivity_raw[str(qubit)]]

    keys = list(connectivity.keys())

    coupling_map = [(i, j) for i in keys for j in connectivity[i]]
    
    return connectivity, coupling_map


def get_braket_device_native_gates(device, dev_properties=None):
    if dev_properties is None:
        dev_properties = device.properties

    # need to hard-code for now
    gate_num_qubits = {'ecr': 2, 'cx': 2, 'rx': 1, 'rz': 1, 'v': 1, 'x': 1,
                       'xy': 2, 'cz': 2, 'cphaseshift': 2, 'gpi': 1, 'gpi2': 1,
                       'ms': 2}

    native_gates_raw = dev_properties.paradigm.nativeGateSet

    if 'i' in native_gates_raw:
        native_gates_raw.remove('i')

    basis_gates = [[], []]

    for gate in native_gates_raw:
        basis_gates[gate_num_qubits[gate.lower()] - 1].append(gate.lower())

    return basis_gates


def get_braket_device_gate_num_params(basis_gates):
    # need to hard-code for now
    gates_parameters = {'rz': 1, 'cx': 0, 'sx': 0, 'x': 0, 'y': 0, 'z': 0,
                        'cz': 0, 'h': 0, 'rx': 1, 'ry': 1, 'ecr': 0,
                        'v': 0, 'xy': 1, 'cphaseshift': 1, 'gpi': 1,
                        'gpi2': 1, 'ms': 3}    

    gate_params = [
        [gates_parameters[gate] for gate in basis_gates[i]]
        for i in range(2)
    ]

    return gate_params


def get_braket_device_coherence_times(one_qubit_data, qubit_indices):
    t1_times = {
        qubit: one_qubit_data[str(qubit)]['T1']
        for qubit in qubit_indices
    }

    t2_times = {
        qubit: one_qubit_data[str(qubit)]['T2']
        for qubit in qubit_indices
    }

    return t1_times, t2_times


def get_braket_device_1q_succ_probs(qubit_indices, one_qubit_data, gate_names_1q):
    one_qubit_gate_succ_probs = dict()

    for qubit in qubit_indices:
        one_qubit_gate_succ_probs[qubit] = dict()

        for gate, gate_error_name in gate_names_1q.items():
            if gate_error_name is None:
                one_qubit_gate_succ_probs[qubit][gate] = 1.0
            else:
                one_qubit_gate_succ_probs[qubit][gate] = one_qubit_data[str(qubit)][gate_error_name]

    return one_qubit_gate_succ_probs
    
    
def get_braket_device_2q_succ_probs(connectivity, qubit_indices, basis_gates, gate_names_2q,
                                    two_qubit_data, symmetric=False):
    two_qubit_gate_succ_probs = dict()
    
    for idx, qubit in enumerate(qubit_indices):
        two_qubit_gate_succ_probs[qubit] = dict()
        
        for gate in basis_gates[1]:
            two_qubit_gate_succ_probs[qubit][gate] = dict()

            if qubit in connectivity.keys():
                for paired_qubit in connectivity[qubit]:
                    try:
                        curr_gate_fid = two_qubit_data[f'{qubit}-{paired_qubit}'][gate_names_2q[gate]]
                    except KeyError:
                        curr_gate_fid = 0
                        
                    if symmetric and curr_gate_fid == 0:
                        try:
                            curr_gate_fid = two_qubit_data[f'{paired_qubit}-{qubit}'][gate_names_2q[gate]]
                        except KeyError:
                            pass                       

                    two_qubit_gate_succ_probs[qubit][gate][paired_qubit] = curr_gate_fid

    return two_qubit_gate_succ_probs
    
    
def get_braket_device_readout_probs(one_qubit_data, qubit_inds):
    readout_probs = {qubit: one_qubit_data[str(qubit)]['fRO'] for qubit in qubit_inds}
    
    return readout_probs


def get_braket_device_gate_times(qubit_inds, connectivity, one_qubit_data,
                                 two_qubit_data, basis_gates):
    gate_times = dict()
    
    for qubit in qubit_inds:
        gate_times[qubit] = dict()
    
    for gate in basis_gates[0]:
        for qubit in qubit_inds:
            gate_times[qubit][gate] = 0.01 # no gate times in device data right now

    for qubit in qubit_inds:
        for gate in basis_gates[1]:
            gate_times[qubit][gate] = dict()

            for paired_qubit in connectivity[qubit]:
                gate_times[qubit][gate][paired_qubit] = 0.1 # no gate times right now   
                
    return gate_times


def extract_properties_from_braket_device(device_name, device=None,
                                          dev_properties=None,
                                          symmetric=False):
    (
        one_qubit_data, two_qubit_data, gate_names_2q, gate_names_1q
    ) = get_braket_device_qubit_edge_info(device, dev_properties, device_name)

    qubit_indices = get_braket_device_qubit_inds(one_qubit_data)
    connectivity, coupling_map = get_braket_device_connectivity(device, qubit_indices, dev_properties)
    basis_gates = get_braket_device_native_gates(device, dev_properties)
    
    gate_num_params = get_braket_device_gate_num_params(basis_gates)
    
    t1_times, t2_times = get_braket_device_coherence_times(one_qubit_data, qubit_indices)

    gate_times = get_braket_device_gate_times(qubit_indices, connectivity,
                                              one_qubit_data, two_qubit_data, basis_gates)
    
    two_qubit_gate_succ_probs = get_braket_device_2q_succ_probs(connectivity, qubit_indices,
                                                                basis_gates, gate_names_2q,
                                                                two_qubit_data, symmetric)

    one_qubit_gate_succ_probs = get_braket_device_1q_succ_probs(
        qubit_indices, one_qubit_data, gate_names_1q
    )

    readout_probs = get_braket_device_readout_probs(one_qubit_data, qubit_indices)
    
    num_device_qubits = len(qubit_indices)
    
    return (
        num_device_qubits, connectivity, t1_times, t2_times, readout_probs, basis_gates,
        gate_num_params, gate_times, two_qubit_gate_succ_probs, one_qubit_gate_succ_probs,
        qubit_indices
    )


def extract_properties_from_ibm_device(backend):
    gates_parameters = {
        'rz': 1, 'cx': 0, 'sx': 0, 'x': 0, 'y': 0,
        'z': 0, 'cz': 0, 'h': 0, 'rx': 1, 'ry': 1,
        'ecr': 0
    }
    
    properties = backend.properties()
    config = backend.configuration()

    coupling_map = config.coupling_map
    num_device_qubits = config.num_qubits
    connectivity = {i : [] for i in range(num_device_qubits)}

    for i in coupling_map:
        connectivity[i[0]].append(i[1])

    t1_times = [properties.t1(i) for i in range(num_device_qubits)]
    t2_times = [properties.t2(i) for i in range(num_device_qubits)]

    readout_success_probs = [1 - properties.readout_error(i) for i in range(num_device_qubits)]
    basis_gates_raw = config.basis_gates
    basis_gates = [[], []]
    gate_param_nums = [[], []]

    for gate in basis_gates_raw:
        basis_gates[len(list(properties.gate_property(gate).keys())[0]) - 1].append(gate)

    basis_gates[0].remove('id')
    basis_gates[0].remove('reset')

    for gate in basis_gates[0] + basis_gates[1]:
        gate_param_nums[len(list(properties.gate_property(gate).keys())[0]) - 1].append(gates_parameters[gate])
    
    gate_times = [dict() for i in range(num_device_qubits)]
    
    for gate in basis_gates[0]:
        for i in range(num_device_qubits):
            gate_times[i][gate] = properties.gate_length(gate, i)

    for i in range(num_device_qubits):
        for gate in basis_gates[1]:
            gate_times[i][gate] = dict()

            for paired_qubit in connectivity[i]:
                gate_times[i][gate][paired_qubit] = properties.gate_length(gate, [i, paired_qubit])

    two_qubit_gate_success_probs = [dict() for i in range(num_device_qubits)]
    one_qubit_gate_success_probs = [dict() for i in range(num_device_qubits)]

    for i in range(num_device_qubits):
        for gate in basis_gates[1]:
            two_qubit_gate_success_probs[i][gate] = dict()

            for paired_qubit in connectivity[i]:
                two_qubit_gate_success_probs[i][gate][paired_qubit] = 1 - properties.gate_error(gate, [i, paired_qubit])
                
        for gate in basis_gates[0]:
            one_qubit_gate_success_probs[i][gate] = properties.gate_error(gate, [i])
                
    qubit_indices = [i for i in range(num_device_qubits)]
                
    return (
        num_device_qubits, connectivity, t1_times, t2_times, readout_success_probs,
        basis_gates, gate_param_nums, gate_times, two_qubit_gate_success_probs, 
        one_qubit_gate_success_probs, qubit_indices
    )


def generate_qubit_mappings(connectivity, num_device_qubits, num_qubits, qubit_inds, num_trials):
    qubit_mappings = []
    mapping_edges = []

    sample_probs_connectivity = np.exp(
        [
            len(connectivity[qubit_inds[i]])
            for i in range(num_device_qubits)
        ]
    ).astype(np.float32)

    sample_probs_connectivity /= np.sum(sample_probs_connectivity)

    for i in range(num_trials):
        curr_mapping = []
        curr_mapping.append(
            np.random.choice(qubit_inds, p=sample_probs_connectivity)
        )

        mapping_found = True

        while len(curr_mapping) < num_qubits:
            qubit_added = False

            for selected_qubit in np.random.permutation(curr_mapping):
                for connected_qubit in np.random.permutation(connectivity[selected_qubit]):
                    if connected_qubit not in curr_mapping:
                        qubit_added = True
                        curr_mapping.append(connected_qubit)
                        break
                        
                if qubit_added:
                    break

            if not qubit_added:
                mapping_found = False
                break

        if mapping_found and curr_mapping not in qubit_mappings:
            qubit_mappings.append(curr_mapping)   
            curr_edges = [[[qubit, paired_qubit]
                           for paired_qubit in connectivity[qubit] if paired_qubit in curr_mapping]
                          for qubit in curr_mapping]
    
            mapping_edges.append([i for j in curr_edges for i in j])

    return qubit_mappings, mapping_edges


def compute_mappings_quality(mappings, edges, t1_times, t2_times, meas_succ_probs, ent_succ_probs, gate_times,
                             num_meas_qubits, coherence_times, two_qubit_gates):
    mappings_quality = []
    
    for i in range(len(mappings)):
        curr_cnot_succ_prob = np.mean([[ent_succ_probs[e[0]][gate][e[1]] for e in edges[i]]
                                       for gate in two_qubit_gates])

        curr_coherence_time = np.mean([coherence_times[q] for q in mappings[i]])
        curr_readout_succ_prob = np.mean(np.sort([meas_succ_probs[sel_qubit]
                                                  for sel_qubit in mappings[i]])[-1 * num_meas_qubits:])

        curr_cnot_time = np.mean([[gate_times[e[0]][gate][e[1]] for e in edges[i]]
                                 for gate in two_qubit_gates])
        
        mappings_quality.append(curr_cnot_succ_prob * curr_readout_succ_prob * (curr_coherence_time / curr_cnot_time))
    
    mappings_quality = np.array(mappings_quality)
    mappings_quality -= np.min(mappings_quality)
    mappings_quality /= (np.max(mappings_quality) + 1e-5)
        
    return mappings_quality


def generate_softmax_dist(values, temp):
    """
    Return the softmax distribution corresponding to the values and temperature passed in.
    """
    logits = np.divide(values, temp)
    logits -= np.max(logits)
    raw_probs = np.exp(logits)
    
    return raw_probs / np.sum(raw_probs)


def find_qubit_placement(circ_gates, gate_params, new_gate, num_gates_on_qubits, coherence_times, temp, use_coherence=False):
    """
    Find a qubit placement for a new 1 qubit gate.
    """
    num_qubits = len(num_gates_on_qubits)
    normalized_coherence_times = np.array(coherence_times) / (np.std(coherence_times) + 1e-6)
    
    if use_coherence:
        probs_dist = generate_softmax_dist(np.divide(-1 * np.array(num_gates_on_qubits), normalized_coherence_times), temp)
    else:
        probs_dist = generate_softmax_dist(-1 * np.array(num_gates_on_qubits), temp)
        
    probs_dist += 1e-5
    probs_dist /= np.sum(probs_dist)
    
    last_gates_on_qubits = [None for i in range(num_qubits)]
    
    for i in range(len(gate_params)):
        for j in range(len(gate_params[i])):
            last_gates_on_qubits[gate_params[i][j]] = circ_gates[i]

    tried_placements = np.array([False for i in range(num_qubits)])
    chosen_placement = None
    
    while (True):
        placement = np.random.choice(num_qubits, p=probs_dist)
        
        if new_gate != last_gates_on_qubits[placement]:
            chosen_placement = placement
            break
        
        tried_placements[placement] = True
        
        probs_dist[placement] = 0
        
        if np.all(tried_placements):
            break
        
        probs_dist /= np.sum(probs_dist)
    
    return chosen_placement


def find_edge_placement(circ_gates, gate_params, new_gate, edges, num_gates_on_qubits,
                        edge_success_rates, temp):
    """
    Find an edge placement for a new 2 qubit gate.
    """
    num_edges = len(edge_success_rates)
    num_qubits = len(num_gates_on_qubits)
    average_gates_for_edges = [np.mean([num_gates_on_qubits[e[0]], num_gates_on_qubits[e[1]]]) for e in edges]
    extra_gates_for_edges = []
    
    for i in range(num_edges):
        extra_gates_for_edges.append(
            max(num_gates_on_qubits[edges[i][0]], num_gates_on_qubits[edges[i][1]]) - min(
            num_gates_on_qubits[edges[i][0]], num_gates_on_qubits[edges[i][1]]))
    
    probs_dist = generate_softmax_dist(-1 * np.divide(np.array(average_gates_for_edges), np.array(edge_success_rates) + 1e-8), temp)
    probs_dist_2 = generate_softmax_dist(-1 * np.array(extra_gates_for_edges), temp)
    
    probs_dist = np.multiply(probs_dist, probs_dist_2)
    probs_dist /= np.sum(probs_dist)
    
    tried_placements = np.array([False for i in range(num_edges)])
    
    for i in range(num_edges):
        if edge_success_rates[i] < 1e-1:
            probs_dist[i] = 0
            tried_placements[i] = True
            
        if average_gates_for_edges[i] < 1:
            probs_dist[i] = 0
            
    if np.abs(np.sum(probs_dist)) < 1e-10:
        return None
    
    probs_dist /= np.sum(probs_dist)
    
    last_gates_on_qubits = [None for i in range(num_qubits)]
    
    for i in range(len(gate_params)):
        for j in range(len(gate_params[i])):
            last_gates_on_qubits[gate_params[i][j]] = circ_gates[i]
    
    chosen_placement = None
    
    while (True):
        placement = np.random.choice(num_edges, p=probs_dist)
        
        if (new_gate != last_gates_on_qubits[edges[placement][0]]) and (new_gate != last_gates_on_qubits[edges[placement][0]]):
            chosen_placement = placement
            break
        
        tried_placements[placement] = True
        
        if edges[placement][::-1] in edges:
            tried_placements[edges.index(edges[placement][::-1])] = True

        if np.all(tried_placements):
            break
            
        probs_dist[placement] = 0
        
        if edges[placement][::-1] in edges:
            probs_dist[edges.index(edges[placement][::-1])] = 0

        if np.abs(np.sum(probs_dist)) < 1e-6:
            return None
        
        probs_dist /= np.sum(probs_dist)
        

    if chosen_placement is not None:
        return edges[chosen_placement]    
    else:
        return None
    
    
    
def generate_device_aware_gate_circ(ibm_backend, num_qubits, num_embed_gates,
                                    num_var_params=0, ent_prob=0.5,
                                    add_rotation_gates=False, param_focus=2,
                                    num_meas_qubits=1, num_trial_mappings=100,
                                    temp=1e-2, num_device_qubits=None,
                                    connectivity=None, t1_times=None,
                                    t2_times=None, meas_succ_probs=None,
                                    basis_gates=None, gates_param_nums=None,
                                    gate_times=None, ent_succ_probs=None, 
                                    braket_device_properties=None,
                                    braket_device_name=None,
                                    symmetric_connectivity=False):
    """
    Generate a device-aware circuit via biased random sampling,
    along with a qubit mapping from logical to physical
    qubits based on device connectivity and calibration data.
    """
    if ibm_backend is not None:
        (
            num_device_qubits, connectivity, t1_times, t2_times,
            meas_succ_probs,basis_gates, gates_param_nums, gate_times,
            ent_succ_probs, one_qubit_succ_probs, qubit_inds
        ) = extract_properties_from_ibm_device(
            ibm_backend
        )
        
    if braket_device_properties is not None:
        (
            num_device_qubits, connectivity, t1_times, t2_times, meas_succ_probs,
            basis_gates, gates_param_nums, gate_times, ent_succ_probs,
            one_qubit_succ_probs, qubit_inds
        ) = extract_properties_from_braket_device(
            braket_device_name, None,
            braket_device_properties, symmetric_connectivity
        )

    # get a good qubit mapping
    
    potential_mappings, mapping_edges = generate_qubit_mappings(connectivity, num_device_qubits, num_qubits,
                                                                qubit_inds, num_trial_mappings)

    coherence_times = {qubit: np.mean([t1_times[qubit], t2_times[qubit]]) for qubit in qubit_inds}
    mappings_quality = compute_mappings_quality(potential_mappings, mapping_edges, t1_times, t2_times,
                                                meas_succ_probs, ent_succ_probs, gate_times, num_meas_qubits,
                                                coherence_times, basis_gates[1])

    mapping_probs = generate_softmax_dist(mappings_quality, temp)
    best_ind = np.random.choice(len(mapping_probs), p=mapping_probs)
    selected_mapping, selected_mapping_edges = potential_mappings[best_ind], mapping_edges[best_ind]
    sel_map = np.array(selected_mapping)
    sel_edges = [[np.argwhere(sel_map == e[i]).flatten().item() for i in range(2)] for e in selected_mapping_edges]
    
    # select gates based on coherence times for each qubit, current time spent running gates on each qubit, 2q error rates on each qubit pair

    if add_rotation_gates:
        for rot_gate in ['ry', 'rz', 'rx']:
            if rot_gate not in basis_gates[0]:
                basis_gates[0].append(rot_gate)
                gates_param_nums[0].append(1)

    qubit_coherence_times = np.array([coherence_times[qubit] for qubit in selected_mapping])
    edge_coherence_times = np.array([min(coherence_times[e[0]], coherence_times[e[1]]) for e in selected_mapping_edges])
    num_edges = len(sel_edges)
    
    circ_gates = []
    gate_params = []
    inputs_bounds = [0]
    weights_bounds = [0]
    
    num_params_in_circ = 0
    index = 0
    max_params = num_embed_gates + num_var_params
    num_gates_on_qubits = np.zeros(num_qubits)
    
    gate_sample_probs = [generate_softmax_dist(param_focus * np.array(gates_param_nums[0]), 1),
                         generate_softmax_dist(param_focus * np.array(gates_param_nums[1]), 1)]
    gate_qubit_probs = [1 - ent_prob, ent_prob]
    
    param_indices = []
    
    while (num_params_in_circ < max_params):
        success = False
        
        curr_sample_probs = [np.copy(gate_sample_probs[i]) for i in range(2)]
        curr_qubit_probs = [gate_qubit_probs[i] for i in range(2)]
        
        while success == False:
            if num_params_in_circ > 0:
                gate_num_qubits = np.random.choice(2, p=curr_qubit_probs)
            else:
                gate_num_qubits = 0

            gate_index = np.random.choice(len(basis_gates[gate_num_qubits]), p=curr_sample_probs[gate_num_qubits])
            new_gate = basis_gates[gate_num_qubits][gate_index]
            
            if gate_num_qubits == 0:
                gate_placement = find_qubit_placement(circ_gates, gate_params, new_gate, num_gates_on_qubits, qubit_coherence_times, temp)
            else:
                edge_success_rates = np.array([ent_succ_probs[e[0]][new_gate][e[1]] for e in selected_mapping_edges])
                
                gate_placement = find_edge_placement(circ_gates, gate_params, new_gate, sel_edges, num_gates_on_qubits,
                                                     edge_success_rates, temp)

            if gate_placement is not None:
                success = True
                
                if gate_num_qubits == 0:
                    num_gates_on_qubits[gate_placement] += 1
                    
                    gate_params.append([gate_placement])
                else:
                    max_gates_among_pair = max(num_gates_on_qubits[gate_placement[0]], num_gates_on_qubits[gate_placement[1]])
                    num_gates_on_qubits[gate_placement[0]] = max_gates_among_pair + 1
                    num_gates_on_qubits[gate_placement[0]] = max_gates_among_pair + 1
                    
                    gate_params.append(gate_placement)

                new_params = gates_param_nums[gate_num_qubits][gate_index]
                num_params_in_circ += new_params
                    
                if new_params:
                    param_indices.append(index)
                    
                circ_gates.append(new_gate)
                index += 1
            else:
                curr_sample_probs[gate_num_qubits][gate_index] = 0
                
                if (np.sum(curr_sample_probs[gate_num_qubits]) == 0):
                    curr_qubit_probs[gate_num_qubits] = 0
                    curr_qubit_probs[1 - gate_num_qubits] = 1
                else:
                    curr_sample_probs[gate_num_qubits] = [prob / np.sum(curr_sample_probs[gate_num_qubits]) for prob in curr_sample_probs[gate_num_qubits]]

    # randomly choose embed indices
        
    embeds_indices = np.random.choice(param_indices, num_embed_gates, False)
    
    for i, circ_gate in enumerate(circ_gates):
        num_qubits_for_gate = len(gate_params[i])
        num_params_in_gate = gates_param_nums[
            num_qubits_for_gate - 1
        ][basis_gates[num_qubits_for_gate - 1].index(circ_gate)]

        if i not in param_indices:
            inputs_bounds.append(inputs_bounds[-1])
            weights_bounds.append(weights_bounds[-1])
        else:
            if i in embeds_indices:
                inputs_bounds.append(
                    inputs_bounds[-1] + num_params_in_gate
                )
                weights_bounds.append(weights_bounds[-1])
            else:
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1] + num_params_in_gate)
                    
    # choose the best qubits to measure
    
    curr_meas_probs = generate_softmax_dist([meas_succ_probs[qubit] for qubit in sel_map], temp)
    meas_qubits = np.random.choice(num_qubits, num_meas_qubits, False, p=curr_meas_probs)
        
    return circ_gates, gate_params, inputs_bounds, weights_bounds, selected_mapping, meas_qubits