#!/bin/bash
# random circuits, rxyz + cz gateset, angle encoding
python train_random.py --dataset fmnist_2 --num_params 32 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/fmnist_2/32_params
python train_random.py --dataset vowel_2 --num_params 32 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/vowel_2/32_params
python train_random.py --dataset vowel_4 --num_params 40 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/vowel_4/40_params
python train_random.py --dataset mnist_4 --num_params 40 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/mnist_4/40_params