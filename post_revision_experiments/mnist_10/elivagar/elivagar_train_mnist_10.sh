python train_circuits.py --dataset mnist_10_6_nll --circs_dir ./ours/device_aware/ibm_kyoto/mnist_10/72_params/ --num_epochs 100 --batch_size 512 --num_runs_per_circ 1 --encoding_type angle --learning_rate 0.01 --save_dir ./ours/device_aware/ibm_kyoto/mnist_10/72_params/{} --num_circs 100 --contains_multiple --circ_prefix circ --dataset_file_extension npy --loss nll --use_gpu