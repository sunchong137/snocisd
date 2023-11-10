from pyscf import gto, scf
import numpy as np
import time
import sys 
sys.path.append("../..")
import thouless
import time
import reshf as optdets

mol = gto.Mole()
a = 2.0
mol.atom = "Cr 0 0 0; Cr 0 0 {}".format(a)
mol.basis = "ccpvdz"
mol.spin = 0
mol.unit ="angstrom"
mol.symmetry = True 
mol.build() 

mf = scf.UHF(mol)

# Frozen occupancy
# mf.irrep_nelec = {'A1g':5, 'A1u':5} 
ehf = mf.kernel()

# nao = mol.nao
# norb = mol.nao # number of orbitals
# occ = mf.get_occ()
# nocc = int(np.sum(occ[0]))

norb = mol.nao # number of orbitals
occ = mf.get_occ()
nocc = int(np.sum(occ[0])) # number of occupied orbitals for spin up
nvir = norb - nocc
# ao_ovlp = mol.intor_symmetric ('int1e_ovlp') # overlap matrix of AO orbitals
h1e = mf.get_hcore()
h2e = mol.intor('int2e')
mo_coeff = np.asarray(mf.mo_coeff)
e_nuc = mf.energy_nuc()

# generate initial guess for thouless rotations
n_dets = 1
niter = 500
t0 = thouless.gen_thouless_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)[:n_dets]
t0 = t0.reshape(n_dets, -1)
# RES HF
t1 = time.time()
E, rn = optdets.optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=n_dets, init_tvecs=t0, MaxIter=niter)
t2 = time.time()
print(f"Time used: {t2-t1}.")
e_noci = E + e_nuc
print("Energy noci: ", e_noci)