# Equivariant Diffusion for Molecule Generation in 3D
## Archelaos Chalkiadakis, Harold Ruiter, Marina Orozco Gonzalez, Ricardo Chavez Torres, Robin Reitsma

This repository contains the implementation of the Equivariant Diffusion Model (EDM) for generating 3D molecular structures, based on the work by Hoogeboom et al. (2022). The project aims to reproduce and extend the original work using JAX/FLAX, comparing its performance to the PyTorch implementation.

The main objectives of our project are:
1. Do a thorough review of the current literature on geometric deep learning
for molecule generation.
2. Reproduce the results of Equivariant Diffusion Hoogeboom et al. (2022)
3. Implement the Equivariant Diffusion Hoogeboom et al. (2022) algorithm
in JAX.
4. Measure efficiency and performance in JAX in comparison to to Pytorch
implementation

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
