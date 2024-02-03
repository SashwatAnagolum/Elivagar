#!/bin/bash

python generate_device_aware_circuits.py --target_dataset bank --num_circs 2500 --device_name ibm_osaka --save_dir ./experiments/bank/20_params/ibm_osaka/elivagar --temp 0.5 --add_rotations