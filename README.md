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

Now, clone the following repositories to enable support for QTN-VQC:

```
git clone https://github.com/uwjunqi/PreTrained-TTN_VQC
git clone https://github.com/uwjunqi/Pytorch-Tensor-Train-Network.git
```

Install the Pytorch-Tensor-Train-Network package:

```
cd Pytorch-Tensor-Train-Network
python setup.py install
```

Setup should now be complete.

## Setup check

To test whether the environment is set up correctly, first activate the created virtual environment, and then execute the following script:



## Example usage

As an example, we can run 