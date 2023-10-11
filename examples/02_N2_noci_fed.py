'''
Example from Carlos' paper J. Chem. Phys. 139, 204102 (2013)
RHF: -108.9547
CCSD: -109.2740
CCSD(T): -109.2863
'''

from pyscf import gto, scf, cc
import numpy as np
import sys 
from noci_jax import thouless, pyscf_helper
from noci_jax import opt_fed as optdets

# set up the system with pyscf
bond_length = 1.09768
mol = gto.Mole()
mol.atom = '''
N   0   0   0
N   0   0   {}
'''.format(bond_length)
mol.unit = "angstrom"
mol.basis = "sto6g"
mol.symmetry=1
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

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf) 
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
 

n_dets = 2
niter = 5000
nsweep = 0
print_step=500
t0 = thouless.gen_init_singles(nocc, nvir, max_nt=n_dets, zmax=2, zmin=0.1)[:n_dets]
t_noise = thouless.gen_thouless_random(nocc, nvir, max_nt=n_dets)
# t0 = t0 + t_noise * 0.5
t0 = t0.reshape(n_dets, -1)
# RES HF
E0, tn0 = optdets.optimize_fed(h1e, h2e, mo_coeff, nocc, nvecs=n_dets, 
                               init_tvecs=t0, MaxIter=niter, print_step=print_step)
E, tn = optdets.optimize_sweep(h1e, h2e, mo_coeff, nocc, tn0, MaxIter=niter, 
                               nsweep=nsweep, E0=E0, print_step=print_step)
e_noci = E + e_nuc
print("E: ", e_noci)


