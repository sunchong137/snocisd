'''
Example from Carlos' paper J. Chem. Phys. 139, 204102 (2013)
RHF: -108.9547
CCSD: -109.2740
CCSD(T): -109.2863
NOTE:  Carlos used Cartessian basis with Gaussian -> mol.cart = True
'''

from pyscf import gto, scf, cc
import numpy as np
import sys 
from noci_jax import thouless, optrbm_all, pyscf_helpers

# set up the system with pyscf
bond_length = 1.09768
mol = gto.Mole()
mol.atom = '''
N   0   0   0
N   0   0   {}
'''.format(bond_length)
mol.unit = "angstrom"
mol.basis = "ccpvdz"
mol.cart=True
mol.build()

break_symm = True

# Mean-field calculation
mf = scf.UHF(mol)
mf.kernel()
if break_symm:
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init) 


# # CCSD 
# mycc = cc.CCSD(mf).run()  
# e_ccsd = mycc.e_tot
# print("CCSD: ", e_ccsd)

# # CCSD(T)
# et = mycc.ccsd_t()
# e_ccsdt = e_ccsd + et
# print("CCSD(T): ", e_ccsdt)

# NOCI res HF

h1e, h2e, e_nuc = pyscf_helpers.get_integrals(mf) 
norb, nocc, nvir, ao_ovlp, mo_coeff = pyscf_helpers.get_mos(mf)


# generate initial guess for thouless rotations
n_dets = 2
niter = 8000
print_step = 1000
tol = 1e-6

t0 = thouless.gen_init_singles(nocc, nvir, max_nt=n_dets, zmax=2, zmin=0.1)[:n_dets]
# t0 += thouless.gen_thouless_random(nocc, nvir, max_nt=n_dets) * 0.1 # better to add noise

nvecs = len(t0)
t0 = t0.reshape(nvecs, -1)
# RES HF
er, vecs = optrbm_all.rbm_all(h1e, h2e, mo_coeff, nocc, nvecs, 
                              init_params=t0, hiddens=[0,1], MaxIter=niter, print_step=print_step)
e_rbm = er + e_nuc
print("E: ", e_rbm)


