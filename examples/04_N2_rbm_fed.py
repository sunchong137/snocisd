'''
Example from Carlos' paper J. Chem. Phys. 139, 204102 (2013)
RHF: -108.9547
CCSD: -109.2740
CCSD(T): -109.2863
TODO: PySCF and Gaussian gave different results...
'''

from pyscf import gto, scf, cc
import numpy as np
import time
import sys 
sys.path.append("..")
sys.path.append(".")
import rbm, noci

# set up the system with pyscf
bond_length = 1.09768
mol = gto.Mole()
mol.atom = '''
N   0   0   0
N   0   0   {}
'''.format(bond_length)
mol.unit = "angstrom"
mol.basis = "6-31g"
mol.symmetry=1
mol.build()

# Mean-field calculation
mf = scf.UHF(mol)
mf.conv_tol = 1e-10
# break symmetry
init_guess = mf.get_init_guess()
init_guess[0][0, 0] = 2
mf.kernel(init_guess)
e_hf = mf.energy_tot()
print("UHF: ", e_hf)

# # CCSD 
# mycc = cc.CCSD(mf).run()  
# e_ccsd = mycc.e_tot
# print("CCSD: ", e_ccsd)

# # CCSD(T)
# et = mycc.ccsd_t()
# e_ccsdt = e_ccsd + et
# print("CCSD(T): ", e_ccsdt)

# NOCI res HF

# First get values that will be used for NOCI
norb = mol.nao # number of orbitals
occ = mf.get_occ()
nocc = int(np.sum(occ[0])) # number of occupied orbitals for spin up
nvir = norb - nocc
ao_ovlp = mol.intor_symmetric ('int1e_ovlp') # overlap matrix of AO orbitals
h1e = mf.get_hcore()
h2e = mol.intor('int2e')
mo_coeff = np.asarray(mf.mo_coeff)
e_nuc = mf.energy_nuc()


# generate initial guess for thouless rotations
n_dets = 2
t0 = noci.gen_thouless_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)
t0 = t0[:n_dets]
nvecs = len(t0)
t0 = t0.reshape(nvecs, -1)
# RES HF
t1 = time.time()
er, vecs = rbm.rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_rbms=t0, ao_ovlp=ao_ovlp, nsweep=0, tol=1e-5, MaxIter=10)
t2 = time.time()
print("Time used:", t2-t1)
e_rbm = er + e_nuc
print("E: ", e_rbm)

