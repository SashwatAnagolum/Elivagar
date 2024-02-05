#!/bin/bash
python examples/train.py examples/configs/ours/rxyz/train_sub/aspen_m_3/moons_300_16.yaml --ckpt-dir=es_runs/rxyz/revision/aspen_m_3/ours.rxyz.es.aspen_m_3.moons_300_16.yaml --save_dir=../searched_circuits/rxyz/revision/moons_300_16_rigetti_aspen_m_3
python examples/train.py examples/configs/ours/rxyz/train_sub/aspen_m_3/mnist_4_40.yaml --ckpt-dir=es_runs/rxyz/revision/aspen_m_3/ours.rxyz.es.aspen_m_3.mnist_4_40.yaml --save_dir=../searched_circuits/rxyz/revision/mnist_4_40_rigetti_aspen_m_3
python examples/train.py examples/configs/ours/rxyz/train_sub/aspen_m_3/bank_20.yaml --ckpt-dir=es_runs/rxyz/revision/aspen_m_3/ours.rxyz.es.aspen_m_3.bank_20.yaml --save_dir=../searched_circuits/rxyz/revision/bank_20_rigetti_aspen_m_3
python examples/train.py examples/configs/ours/rxyz/train_sub/aspen_m_3/fmnist_2_32.yaml --ckpt-dir=es_runs/rxyz/revision/aspen_m_3/ours.rxyz.es.aspen_m_3.fmnist_2_32.yaml --save_dir=../searched_circuits/rxyz/revision/fmnist_2_32_rigetti_aspen_m_3
python examples/train.py examples/configs/ours/rxyz/train_sub/lucy/vowel_2_32.yaml --ckpt-dir=es_runs/rxyz/revision/lucy/ours.rxyz.es.lucy.vowel_2_32.yaml --save_dir=../searched_circuits/rxyz/revision/vowel_2_32_oqc_lucy
python examples/train.py examples/configs/ours/rxyz/train_sub/lucy/mnist_2_20.yaml --ckpt-dir=es_runs/rxyz/revision/lucy/ours.rxyz.es.lucy.mnist_2_20.yaml --save_dir=../searched_circuits/rxyz/revision/mnist_2_20_oqc_lucy
python examples/train.py examples/configs/ours/rxyz/train_sub/lucy/vowel_4_40.yaml --ckpt-dir=es_runs/rxyz/revision/lucy/ours.rxyz.es.lucy.vowel_4_40.yaml --save_dir=../searched_circuits/rxyz/revision/vowel_4_40_oqc_lucy
python examples/train.py examples/configs/ours/rxyz/train_sub/lucy/fmnist_4_24.yaml --ckpt-dir=es_runs/rxyz/revision/lucy/ours.rxyz.es.lucy.fmnist_4_24.yaml --save_dir=../searched_circuits/rxyz/revision/fmnist_4_24_oqc_lucy