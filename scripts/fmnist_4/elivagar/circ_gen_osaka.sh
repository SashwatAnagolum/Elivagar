#!/bin/bash

python generate_device_aware_circuits.py --target_dataset fmnist_4 --num_circs 2500 --device_name ibm_osaka --save_dir ./experiments/fmnist_4/24_params/ibm_osaka/elivagar --temp 0.5 --add_rotations