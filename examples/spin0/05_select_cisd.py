import numpy as np
from pyscf import gto, scf, ci
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd_spin0
from noci_jax import slater, pyscf_helper, select_ci



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

mf = scf.RHF(mol)
mf.kernel()

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron
myci = ci.RCISD(mf)
e_corr, civec = myci.kernel()
e_cisd = e_hf + e_corr 

dt = 0.1
tmats, coeffs = nocisd_spin0.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-5)
rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
rmats_fix = rmats[0][None, :]
rmats_new = rmats[1:]

m_tol = 1e-6
e_tol = 1e-8
select_rs = select_ci.select_rmats_ovlp(rmats_fix, rmats_new, m_tol=m_tol)
n_tvecs = len(select_rs)

E = slater.noci_energy(select_rs, mo_coeff, h1e, h2e, e_nuc=e_nuc)
print(E)