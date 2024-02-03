#!/bin/bash

python compute_clifford_nr.py --dataset mnist_4 --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar