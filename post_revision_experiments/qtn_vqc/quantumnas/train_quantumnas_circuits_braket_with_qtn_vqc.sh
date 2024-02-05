#!/bin/bash

# quantumnas + qtn_vqc, rxyz + cz

python train_circuits.py --dataset moons --circs_dir ./quantumnas/searched_circuits/rxyz/revision/moons_16_rigetti_aspen_m_3 --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/moons_16_rigetti_aspen_m_3/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 2 --tt_ranks 1 1 1 --tt_output_size 2 2

python train_circuits.py --dataset bank --circs_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_rigetti_aspen_m_3 --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_rigetti_aspen_m_3/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 2 2 --tt_ranks 1 1 1 1 --tt_output_size 2 2 2

python train_circuits.py --dataset mnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_oqc_lucy --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_oqc_lucy/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy --use_gpu

python train_circuits.py --dataset fmnist_4_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_4_24_oqc_lucy --num_epochs 200 --batch_size 2048 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_4_24_oqc_lucy/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset fmnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_rigetti_aspen_m_3 --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_rigetti_aspen_m_3/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset vowel_2 --circs_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_oqc_lucy --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_oqc_lucy/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 5 --tt_ranks 1 1 1 --tt_output_size 2 5

python train_circuits.py --dataset vowel_4 --circs_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_4_40_oqc_lucy --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_4_40_oqc_lucy/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 5 --tt_ranks 1 1 1 --tt_output_size 2 5

python train_circuits.py --dataset mnist_4_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_4_40_rigetti_aspen_m_3 --num_epochs 200 --batch_size 1024 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_4_40_rigetti_aspen_m_3/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy