'''
Generate reference data for Hubbard model.
'''
import numpy as np
from noci_jax.misc import hubbard_ba
import os

res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

def gen_ba():
    # pbc
    # rows: lattice size
    # columns: U value
    nsites = np.arange(6, 42, 2, dtype=int)
    Uvals = np.arange(2, 10, 2)
    with open(res_dir + "/hubbard1d_ba.txt", "w") as fin:
        fin.write("# 1D Hubbard model w PBC and half-filling\n")
        fin.write("# Bethe Ansatz ground state energies\n")
        fin.write("# rows: number of sites, columns: U value, t = -1.\n")
        fin.write(" 0 ")
        for U in Uvals:
            fin.write("             {}".format(U))
        fin.write("\n")
        for n in nsites:
            ne_a = n//2
            fin.write("{:2d}".format(n))
            for U in Uvals:
                E = hubbard_ba.lieb_wu(n, ne_a, ne_a, U)
                fin.write("  {:2.10f}".format(E/n))
            fin.write("\n")


def gen_dmrg():
    pass