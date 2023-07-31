import numpy as np
#from pyscf import gto, scf
import time
import sys
sys.path.append('../')
import molecules, rbm, noci
from pyscf import gto, scf, cc
nH = 10
a = 1.5
# construct molecule
mol = gto.Mole()
mol.atom = molecules.gen_geom_hchain(nH, a)
mol.basis = "6-31g"
mol.build()
norb = mol.nao
ao_ovlp = mol.intor_symmetric ('int1e_ovlp')

# UHF
mf = scf.UHF(mol)
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

# CCSD 
mycc = cc.CCSD(mf).run()  
e_ccsd = mycc.e_tot
print("CCSD: ", e_ccsd)

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
n_dets = 3
nr = int((n_dets-1e-3)/2) + 1 
t0 = noci.gen_thouless_singles(nocc, nvir, max_nt=nr, zmax=10, zmin=0.1)[:n_dets]
nvecs = len(t0)
t0 = t0.reshape(nvecs, -1)
# RES HF
nsweep=3
niter=20
t1 = time.time()
er, vecs = rbm.rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_rbms=t0, ao_ovlp=ao_ovlp, nsweep=nsweep, tol=1e-5, MaxIter=niter)
t2 = time.time()
print("Time used:", t2-t1)
e_rbm = er + e_nuc
print("Energy HF: ", e_hf)
print("Energy rbm: ", e_rbm)
print("Energy CCSD: ", e_ccsd)