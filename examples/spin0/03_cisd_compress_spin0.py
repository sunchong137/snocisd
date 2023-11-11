import numpy as np
from pyscf import gto, scf, ci, fci, ao2mo
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd_spin0
from noci_jax import slater, pyscf_helper


bl = 1.0
mol = gto.Mole()
mol.atom = f'''
H   0   0   0
H   0   0   {bl}
H   0   0   {2*bl}
H   0   0   {3*bl}
'''
mol.unit = "angstrom"
mol.basis = "sto3g"
mol.cart = True
mol.build()

mf = scf.RHF(mol)
mf.kernel()

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron
myci = ci.RCISD(mf)
e_corr, civec = myci.kernel()
e_cisd = e_hf + e_corr 

# fcivec = myci.to_fcivec(civec)
# myfci = fci.FCI(mf)
# e, v = myfci.kernel()

# h1_mo, h2_mo = pyscf_helper.rotate_ham_spin0(mf)
# E = myfci.energy(h1_mo, h2_mo, fcivec, norb, nelec) + e_nuc
# print(E)
# exit()
# c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)
# c2ab = c2
# c2aa = c2 - c2.transpose(1, 0, 2, 3)
# print(c2aa)
# exit()

dt = 0.1
tmats, coeffs = nocisd_spin0.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-6)
nvir, nocc = tmats.shape[2:]
rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)

umo_coeff = np.array([mo_coeff, mo_coeff])
E = slater.noci_energy(rmats, umo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
print(E)