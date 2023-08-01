'''
Optmizing the rotation matrices.
'''

import numpy as np
from pyscf import gto, scf, fci, cc
import sys


def gen_geom_hchain(n, bond=0.8):
    # generate geometry for hydrogen chain
    # H2 bond length = 0.74 Angstrom
    geom = []
    for i in range(n):
        geom.append(['H', .0, .0, i*bond])
    return geom

nH = 6
a = 1.5

# construct molecule
mol = gto.Mole()
mol.atom = gen_geom_hchain(nH, a)
mol.unit='angstrom'
mol.basis = "sto3g"
mol.build()

mf = scf.UHF(mol)
norb = mol.nao 

ao_ovlp = mol.intor_symmetric ('int1e_ovlp')

h1e = mf.get_hcore()
h2e = mol.intor('int2e')


# Hartree-Fock
init_guess = mf.get_init_guess()
init_guess[0][0,0] = 10
mf.init_guess = init_guess
mf.kernel()
occ = mf.get_occ()
nocc = int(np.sum(occ[0]))
nvir = norb - nocc

# energy
elec_energy = mf.energy_elec()[0]
e_nuc = mf.energy_nuc()
e_hf = mf.energy_tot()
mo_coeff = np.asarray(mf.mo_coeff)

# check UHF
dm = mf.make_rdm1()
# print(dm[0] - dm[1])
diff = np.linalg.norm(dm[0] - dm[1])
if diff < 1e-5:
    print("WARNING: converged to RHF solution.")

# CISE
mycc = cc.UCCSD(mf)
mycc.run()
de = mycc.e_corr
e_cc = e_hf + de

# FCI
myci = fci.FCI(mf)
e_fci, c = myci.kernel()
