# Consensus-based Optimization for Closed-Box Adversarial Attacks and a Connection to Evolution Strategies

This repository contains the code for the experiments in the paper:  
**"Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies"** ([arXiv:2506.24048](https://arxiv.org/abs/2506.24048)).

## Installation

This project is written in Python and mostly depends on the [CBXPy](https://github.com/PdIPS/CBXpy) package and [PyTorch](https://pytorch.org/).

To install `advcbx` and its dependencies locally:

1. Clone this repository:
    ```sh
    git clone https://github.com/TimRoith/AdversarialCBO.git
    ```

2. Install `advcbx` (from this repository's root):
    ```sh
    pip install -e .
    ```

Or, if you use a virtual environment, activate it before running the above commands.

---

## Overview

This repository implements and evaluates consensus-based optimization (CBO) for closed-box adversarial attacks.

### 1. Optimizers
The module [`advcbx/optim`](advcbx/optim/) contains the core optimizers. Most importantly:
- `base.py`: implements the core objective functions.
- `CBOattack.py`: create the CBO attack class and utilities, which can employ existing CBO algorithms.
- `HoppingCBO.py`: implements the HoppingCBO variant and the NES version used in the paper.
-`load_optim.py`: loads optimizers based on a configuration file.

### 2. Attack Spaces
The module [`advcbx/attackspace`](advcbx/attackspace/) defines various attack spaces for adversarial attacks. The attacks can be loaded with the functions in [`advcbx/attackspace/load_attack.py`](advcbx/attackspace/load_attack.py) based on a configuration file. The attack spaces include:

- [`advcbx/attackspace/attack_space.py`](advcbx/attackspace/attack_space.py): Implements the base class for attack spaces
- [`advcbx/attackspace/low_res`](advcbx/attackspace/low_res.py): Implements low-resolution attack spaces.
- [`advcbx/attackspace/dct`](advcbx/attackspace/dct.py): Implements DCT-based attack spaces.
- [`advcbx/attackspace/square`](advcbx/attackspace/square.py): Implements square attack spaces.
- [`advcbx/attackspace/index`](advcbx/attackspace/index.py): Implements $P$-pixel attacks.

### 3. Models and Data
The module [`advcbx/models`](advcbx/models/) contains the model definitions and loading utilities. The models can be loaded with the functions in [`advcbx/models/load_model.py`](advcbx/models/load_model.py) based on a configuration file.

The data loading utilities are in [`advcbx/data`](advcbx/data/), which includes functions to load standard datasets like CIFAR-10, MNIST, and custom CNNs.

### 4. Experimental Setup

- Experiments are managed via Hydra configuration files in [`experiments/conf/`](experiments/conf/).
- Main experiment script: [`experiments/main.py`](experiments/main.py).

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@misc{roith2025advcbo,
      title={Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies}, 
      author={Tim Roith and Leon Bungert and Philipp Wacker},
      year={2025},
      eprint={2506.24048},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2506.24048}, 
}
```