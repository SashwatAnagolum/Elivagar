#!/bin/bash

python compute_clifford_nr.py --dataset mnist_10 --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --dataset_file_extension npy