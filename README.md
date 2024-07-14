<img src="logo.png" alt="" width="200"/>
Selected Non-Orthogonal Configuration Interaction with Singles and Doubles.

Author: Chong Sun [email](sunchong137@gmail.com)
# Features
1. NOCI with FED and ResHF using ADAM optimization provided in the Jax package.
2. Compression of CISD with non-orthognal Slater determinants (NOSDs).
3. Selecting NOSDs based on metric and energy contributions.

# Installation
## Required packages
1. The package is written with Python3.9. It is advisable to create a conda environment by
```bash
conda create -n env_name python=3.9
```
2. PySCF, interface to quantum chemistry.
3. Jax, Jaxlib, Optax, optimization and fast array algebra.

## Installation from GitHub
First get the Python package from GitHub by
```bash
git clone https://github.com/sunchong137/noci_jax.git
```
Then go the to the source directory
```bash
cd noci_jax
```
Lastly, install with `pip`
```bash 
pip install -e .
```
# Running SNOCISD
In the `examples` folder, various examples are provided. 

# Citing SNOCISD
If you use this package, please cite our [paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00240).
```
@article{sun2024snocisd,
author = {Sun, Chong and Gao, Fei and Scuseria, Gustavo E.},
title = {Selected Nonorthogonal Configuration Interaction with Compressed Single and Double Excitations},
journal = {Journal of Chemical Theory and Computation},
volume = {20},
number = {9},
pages = {3741-3748},
year = {2024},
doi = {10.1021/acs.jctc.4c00240},
}
```

