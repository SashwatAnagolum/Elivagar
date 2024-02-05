#!/bin/bash

python circuit_inference.py --dataset vowel_2 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/vowel_2/16_params/ibm_osaka/random_ryxz_cz --num_test_samples 120 --results_save_dir noise_sim --transpiler_opt_level 3
python circuit_inference.py --dataset vowel_2 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/vowel_2/16_params/ibm_osaka/random_rxz_rxx --num_test_samples 120 --results_save_dir noise_sim --transpiler_opt_level 3