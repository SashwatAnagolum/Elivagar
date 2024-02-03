#!/bin/bash

python circuit_inference.py --dataset bank --num_circs 1 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/bank/20_params/ibm_osaka/human_design_angle_basic --num_test_samples 122 --results_save_dir noise_sim --transpiler_opt_level 3 --human_design
python circuit_inference.py --dataset bank --num_circs 1 --num_runs_per_circ 5 --encoding_type iqp --device_name ibm_osaka --circs_dir ./experiments/bank/20_params/ibm_osaka/human_design_iqp_basic --num_test_samples 122 --results_save_dir noise_sim --transpiler_opt_level 3 --human_design