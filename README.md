# Equivariant Diffusion for Molecule Generation in 3D
## Harold Ruiter, Marina Orozco Gonzalez, Ricardo Chavez Torres, Robin Reitsma


The main objectives of our project are:
1. Do a thorough review of the current literature on geometric deep learning
for molecule generation.

This review can be found in our [blogpost](e3-molecule-diffusion\blogpost.md). There, we summarize the key elements of Diffusion Models, Graphs and Equivariance, and how can they help molecules generation. Also, we make an introduction to JAX,  a machine learning framework that has excelent results in terms of speed.

2. Reproduce the results of Equivariant Diffusion Hoogeboom et al. (2022)

In order to do this, we runned the code from the [original repository](https://github.com/ehoogeboom) following the steps they indicated in their README.

3. Implement the Equivariant Diffusion Hoogeboom et al. (2022) algorithm
in JAX.

This is the main **topic of this repository** and of the files you can find in [src](https://github.com/YukumoHunter/e3-molecule-diffusion/tree/main/src). 

In order to run the training, the steps are the same as in the original repository:

* Training the EDM

```python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999```

> NOTE: the code will start running but the results aren't correct. Some debug still needs to be done. 

4. Experiments

Even though we haven't finish our implementation in Jax yet (progess can be checked in [src](e3-molecule-diffusion\src) ), we expect a significant improvement in **training time** by using JAX.

A first draft of our blogpost can be fount at [blogpost.md](e3-molecule-diffusion\blogpost.md). 


## Structure
```
repo
.
|-- readme.md                  # Description of the repo with relevant getting started info
|-- blogpost.md                # Main deliverable: a blogpost style report
|-- src                        # Contains the main project files
|   |-- configs                # Configuration files
|   |   |..
|   |-- data                   # Data handling and preprocessing scripts
|   |   |..
|   |-- egnn                   # Equivariant Graph Neural Network implementation
|   |   |..
|   |-- equivariant_diffusion  # Diffusion model implementation
|   |   |..
|   |-- generated_samples      # Generated molecular samples
|   |   |..
|   |-- qm9                    # QM9 dataset handling
|   |   |..
|   |-- main_qm9.py                    # Main script for running the model
|   |-- ..
```

# Getting Started

## Prerequisites

Ensure you have the following software installed:
 - Python 3.8 or higher
 - Git

## Setting Up the Environment

1. Clone this repo
```
git clone https://github.com/YukumoHunter/e3-molecule-diffusion
cd repo
```

2. Create and activate an environment:
```
python -m venv .venv
. .venv/bin/activate
```

3. Install dependencies:
All dependencies are listed in the requirements.txt file. Install them using pip:
```
pip install -r requirements.txt
```

# Usage
To run the main model on the QM9 dataset:
```
python3 src/main_qm9.py
```

Our blogpost can be fount at [blogpost.md](e3-molecule-diffusion\blogpost.md). 
