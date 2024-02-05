#!/bin/bash
# random circuits, rxyz + cz gateset, angle encoding
python train_circuits.py --dataset moons --circs_dir ./random/rxyz_cz/angle/moons/16_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/moons/16_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset bank --circs_dir ./random/rxyz_cz/angle/bank/20_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/bank/20_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset mnist_2 --circs_dir ./random/rxyz_cz/angle/mnist_2/20_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/mnist_2/20_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset fmnist_4 --circs_dir ./random/rxyz_cz/angle/fmnist_4/24_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/fmnist_4/24_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset fmnist_2 --circs_dir ./random/rxyz_cz/angle/fmnist_2/32_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/fmnist_2/32_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset vowel_2 --circs_dir ./random/rxyz_cz/angle/vowel_2/32_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/vowel_2/32_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset vowel_4 --circs_dir ./random/rxyz_cz/angle/vowel_4/40_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/vowel_4/40_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25

python train_circuits.py --dataset mnist_4 --circs_dir ./random/rxyz_cz/angle/mnist_4/40_params --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./random/rxyz_cz/angle/mnist_4/40_params/{}/with_quantumnat --use_quantumnat --contains_multiple --num_circs 25
