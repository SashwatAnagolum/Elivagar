#!/bin/bash

python compute_clifford_nr.py --dataset bank --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/bank/20_params/ibm_osaka/elivagar