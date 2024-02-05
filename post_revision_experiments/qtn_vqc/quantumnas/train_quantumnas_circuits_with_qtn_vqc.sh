#!/bin/bash
# quantumnat rxyz + cz
python train_circuits.py --dataset moons --circs_dir ./quantumnas/searched_circuits/rxyz/revision/moons_16_ibm_lagos --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/moons_16_ibm_lagos/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 2 --tt_ranks 1 1 1 --tt_output_size 2 2

python train_circuits.py --dataset bank --circs_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_ibm_perth --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_ibm_perth/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 2 2 --tt_ranks 1 1 1 1 --tt_output_size 2 2 2

python train_circuits.py --dataset mnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_ibm_nairobi --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_ibm_nairobi/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset fmnist_4_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_4_24_ibmq_jakarta --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_4_24_ibmq_jakarta/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset fmnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_ibm_perth --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_ibm_perth/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset vowel_2 --circs_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_ibm_nairobi --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_ibm_nairobi/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 5 --tt_ranks 1 1 1 --tt_output_size 2 5

python train_circuits.py --dataset vowel_4 --circs_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_4_40_ibmq_jakarta --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_4_40_ibmq_jakarta/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 5 --tt_ranks 1 1 1 --tt_output_size 2 5

python train_circuits.py --dataset mnist_4_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_4_40_ibm_lagos --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_4_40_ibm_lagos/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy







python train_circuits.py --dataset bank --circs_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_ibm_kyoto --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/bank_20_ibm_kyoto/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 2 2 --tt_ranks 1 1 1 1 --tt_output_size 2 2 2

python train_circuits.py --dataset mnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_ibm_osaka --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/mnist_2_20_ibm_osaka/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset fmnist_2_fullsize --circs_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_ibm_kyoto --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/fmnist_2_32_ibm_kyoto/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 7 16 7 --tt_ranks 1 1 1 1 --tt_output_size 2 2 4 --dataset_file_extension npy

python train_circuits.py --dataset vowel_2 --circs_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_ibm_osaka --num_epochs 200 --batch_size 256 --num_runs_per_circ 5 --encoding_type angle --learning_rate 0.01 --save_dir ./quantumnas/searched_circuits/rxyz/revision/vowel_2_32_ibm_osaka/with_qtn_vqc_small --use_qtn_vqc --num_circs 1 --tt_input_size 2 5 --tt_ranks 1 1 1 --tt_output_size 2 5