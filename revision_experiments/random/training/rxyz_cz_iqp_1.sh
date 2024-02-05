#!/bin/bash
# random circuits, rxyz + cz gateset, iqp encoding
python train_random.py --dataset moons --num_params 16 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type iqp --save_dir ./random/rxyz_cz/iqp/moons/16_params/
python train_random.py --dataset bank --num_params 20 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type iqp --save_dir ./random/rxyz_cz/iqp/bank/20_params/
python train_random.py --dataset mnist_2 --num_params 20 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type iqp --save_dir ./random/rxyz_cz/iqp/mnist_2/20_params/
python train_random.py --dataset fmnist_4 --num_params 24 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --num_circs 25 --num_runs_per_circ 5 --gateset rxyz_cz --encoding_type iqp --save_dir ./random/rxyz_cz/iqp/fmnist_4/24_params/