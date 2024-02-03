#!/bin/bash

python train_human_designed.py --dataset mnist_10_6_nll --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_runs_per_circ 5 --encoding_type angle --save_dir ./experiments/mnist_10/72_params/ibm_osaka/human_design_angle_basic
python train_human_designed.py --dataset mnist_10_6_nll --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_runs_per_circ 5 --encoding_type iqp --save_dir ./experiments/mnist_10/72_params/ibm_osaka/human_design_iqp_basic