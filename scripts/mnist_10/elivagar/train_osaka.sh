#!/bin/bash

python train_elivagar_circuits.py --dataset mnist_10 --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --device_name ibm_osaka --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 160 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --dataset_file_extension npy --loss nll