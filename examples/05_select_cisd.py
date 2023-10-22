import numpy as np
from pyscf import gto, scf, ci
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd
from noci_jax import slater, pyscf_helper, select_ci
import jax
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


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

mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]                                                             
init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
mf.kernel(init) 

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron
myci = ci.UCISD(mf)
e_corr, civec = myci.kernel()
e_cisd = e_hf + e_corr 

dt = 0.1
tmats, coeffs = nocisd.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-5)
# rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
tmats_fix = tmats[0][None, :]
tmats_new = tmats[1:]
m_tol = 1e-8
e_tol = 1e-8
select_ts = select_ci.select_tvecs(tmats_fix, tmats_new, mo_coeff, h1e, h2e, nocc, nvir, m_tol=m_tol, e_tol=e_tol)
n_tvecs = len(select_ts)
np.save("results/snoci_N2_nvec{}.npy".format(n_tvecs), select_ts)

rmats = slater.tvecs_to_rmats(select_ts, nvir, nocc)

E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, e_nuc=e_nuc)
print(E)