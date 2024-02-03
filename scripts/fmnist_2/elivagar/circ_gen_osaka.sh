#!/bin/bash

python generate_device_aware_circuits.py --target_dataset fmnist_2 --num_circs 2500 --device_name ibm_osaka --save_dir ./experiments/fmnist_2/32_params/ibm_osaka/elivagar --temp 0.5 --add_rotations