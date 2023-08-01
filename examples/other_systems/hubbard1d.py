import numpy as np
from pyscf import gto, scf, ao2mo

mymol = gto.M() 
norb = 6
nelec = norb
U = 4
mymol.nelectron = nelec

nocc = nelec // 2
nvir = norb - nocc


h1 = np.zeros((norb, norb))
for i in range(norb-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[norb-1,0] = h1[0,norb-1] = -1.0  # PBC
eri = np.zeros((norb,norb,norb,norb))
for i in range(norb):
    eri[i,i,i,i] = U 


mf = scf.UHF(mymol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(norb)
# ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
# ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
mf._eri = ao2mo.restore(8, eri, norb)


# break symmetry
init_guess = mf.get_init_guess()
init_guess[0][0,0] = 2
init_guess[1][0,0] = 0

mf.init_guess = init_guess
mf.kernel()

# values used for rbm
e_hf = mf.energy_tot()
mo_coeff = mf.mo_coeff
h1e = h1
h2e = eri
ao_ovlp = None
e_nuc = 0