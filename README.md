# Élivágar: Efficient Quantum Circuit Search for Classification

Élivágar is a training-free, device- and noise-aware framework for Quantum Circuit Search (QCS) for classification tasks.

## Setup

To setup the environment required to run Élivágar, first clone this repository:

```
git clone https://github.com/SashwatAnagolum/Elivagar.git
```

Then, create a new virtual environment, and activate it:

```
python -m venv elivagar_venv
elivagar_venv/Scripts/activate
```

Install all of the required packages via pip:

```
pip install -r requirements.txt
```

Now, install the Pytorch-Tensor-Train-Network package:

```
cd qtn_vqc/Pytorch-Tensor-Train-Network
python setup.py install
```

Setup should now be complete.

## Example usage

As an example, we can use Élivágar to search for a circuit for the Moons dataset, targeting the IBM-Osaka device. To do so, we first generate candidate circuits for the target dataset-device combination:

```
./scripts/moons/elivagar/circ_gen_osaka.sh
```

Next, we compute the Clifford Noise Resilience (CNR) of each of the generated candidate circuits to estimate circuit noise robustness:

```
./scripts/moons/elivagar/cnr_osaka.sh
```

We then compute the representation capacity (RepCap) of candidate circuits, to estimate circuit performance:

```
./scripts/moons/elivagar/repcap_osaka.sh
```

Next, we can compute composite scores using the CNR and RepCap scores for each circuit, select the circuits to be trained, and train them on the Moons dataset:

```
./scripts/moons/elivagar/train_osaka.sh
```

Finally, we can evaluate the trained circuits on a withheld test set from the Moons dataset to check circuit performance:

```
./scripts/moons/elivagar/eval_osaka.sh
```
