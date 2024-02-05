#!/bin/bash
# random circuits, rxyz + cz gateset, angle encoding
python train_random.py --dataset moons --num_params 16 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/moons/16_params
python train_random.py --dataset bank --num_params 20 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/bank/20_params
python train_random.py --dataset mnist_2 --num_params 20 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/mnist_2/20_params
python train_random.py --dataset fmnist_4 --num_params 24 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/fmnist_4/24_params

python train_random.py --dataset mnist_10_6_nll --num_params 72 --num_epochs 200 --batch_size 1024 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 1 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/mnist_10/72_params --file_type npy --use_classification_loss

python train_random.py --dataset mnist_10_6_nll --num_params 100 --num_epochs 200 --batch_size 1024 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 1 --gateset rxyz_cz --encoding_type angle --save_dir ./random/rxyz_cz/angle/mnist_10/100_params --file_type npy --use_classification_loss