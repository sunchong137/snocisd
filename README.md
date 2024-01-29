<img src="logo.png" alt=" " width="200"/>

SNOCISD
=======
Selected Non-Orthogonal Configuration Interaction with Singles and Doubles.

Author: Chong Sun [email](sunchong137@gmail.com)
# Features
1. NOCI with FED and ResHF using ADAM optimization provided in the Jax package.
2. Compression of CISD with non-orthognal Slater determinants (NOSDs).
3. Selecting NOSDs based on metric and energy contributions.

# Installation
## Required packages
1. PySCF, interface to quantum chemistry.
2. Jax, Jaxlib, Optax, optimization and fast array algebra.

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
To be added.

