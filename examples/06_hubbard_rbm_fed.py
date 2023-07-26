import numpy as np
import time
import sys
sys.path.append("../")
import molecules, noci, rbm
from pyscf import gto, scf, ao2mo, fci, cc, ci

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
h1e = h1
h2e = eri
ao_ovlp = None
e_nuc = 0

# break symmetry
init_guess = mf.get_init_guess()
init_guess[0][0,0] = 2
init_guess[1][0,0] = 0

mf.init_guess = init_guess
mf.kernel()

e_hf = mf.energy_tot()

mo_coeff = mf.mo_coeff
rdm = mf.make_rdm1()


mycc = cc.CCSD(mf).run()  
e_ccsd = mycc.e_tot
print("CCSD: ", e_ccsd)

t1 = time.time()
myci = ci.CISD(mf).run()
de = myci.e_corr
e_cisd = e_hf + de
t2 = time.time()




myci = fci.direct_spin0
e_fci, c = myci.kernel(h1, eri, norb, norb)
print("E FCI: ", e_fci)

# generate initial guess for thouless rotations
n_dets = 3
nr = int((n_dets-1e-3)/2) + 1 
t0 = noci.gen_thouless_singles(nocc, nvir, max_nt=nr, zmax=2, zmin=0.5)[:n_dets]
#t0 += noci.gen_thouless_random(nocc, nvir, max_nt=n_dets) * 0.8

nvecs = len(t0)
t0 = t0.reshape(nvecs, -1)
# RES HF
nsweep=2
niter=300
ftol=1e-5

er, vecs = rbm.rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_rbms=t0, ao_ovlp=ao_ovlp, nsweep=nsweep, tol=ftol, MaxIter=niter, disp=True)
e_rbm = er + e_nuc

print("Energy HF: ", e_hf)
print("Energy rbm: ", e_rbm)
print("Energy CISD: ", e_cisd)
print("Energy CCSD: ", e_ccsd)
print("Energy FCI: ", e_fci)