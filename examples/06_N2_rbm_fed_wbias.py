'''
Example from Carlos' paper J. Chem. Phys. 139, 204102 (2013)
RHF: -108.9547
CCSD: -109.2740
CCSD(T): -109.2863
'''

from pyscf import gto, scf, cc
import numpy as np
from jax import numpy as jnp
import time
from noci_jax import thouless, reshf, opt_rbm_fed_wbias, pyscf_helpers

# set up the system with pyscf
bond_length = 1.09768
mol = gto.Mole()
mol.atom = '''
N   0   0   0
N   0   0   {}
'''.format(bond_length)
mol.unit = "angstrom"
mol.basis = "sto3g"
mol.cart = True
mol.build()

break_symm = True

# Mean-field calculation
mf = scf.UHF(mol)
mf.kernel()
if break_symm:
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init) 

h1e, h2e, e_nuc = pyscf_helpers.get_integrals(mf) 
norb, nocc, nvir, ao_ovlp, mo_coeff = pyscf_helpers.get_mos(mf)

# # CCSD 
# mycc = cc.CCSD(mf).run()  
# e_ccsd = mycc.e_tot
# print("CCSD: ", e_ccsd)

# # CCSD(T)
# et = mycc.ccsd_t()
# e_ccsdt = e_ccsd + et
# print("CCSD(T): ", e_ccsdt)

# NOCI res HF


# generate initial guess for thouless rotations
n_dets = 2
MaxIter = 5000
print_step = 1000
tol=1e-10
nsweep=1
t0 = thouless.gen_init_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)[:n_dets]
t0 += -thouless.gen_thouless_random(nocc, nvir, max_nt=n_dets) * 0.5

nvecs = len(t0)
t0 = t0.reshape(nvecs, -1)
bias = np.random.rand(n_dets)
# RES HF
# t1 = time.time()
E0, vecs0, bias0 = opt_rbm_fed_wbias.rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, 
                                             init_params=t0, bias=bias, MaxIter=MaxIter, print_step=print_step)
E, vecs, bias_n = opt_rbm_fed_wbias.rbm_sweep(h1e, h2e, mo_coeff, nocc, vecs0, bias0, 
                                              E0=E0, nsweep=nsweep, MaxIter=MaxIter, print_step=print_step)
# t2 = time.time()
# print("Time used:", t2-t1)
hidden_coeffs = reshf.hiddens_to_coeffs([0, 1], n_dets)
lc_coeffs = jnp.exp(hidden_coeffs.dot(bias_n))
print(lc_coeffs)
e_rbm = E + e_nuc
print("E: ", e_rbm)


