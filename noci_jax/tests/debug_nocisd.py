import numpy as np
from pyscf import gto, scf, ci, fci
from pyscf.lib import numpy_helper
import scipy
np.set_printoptions(edgeitems=30, linewidth=100000, precision=6)
from noci_jax import nocisd
from noci_jax import slater, pyscf_helper

# System set up
nH = 4
bl = 1.5
geom = []
for i in range(nH):
    geom.append(['H', 0.0, 0.0, i*bl])

# construct molecule
mol = gto.Mole()
mol.atom = geom
mol.unit='angstrom'
mol.basis = "sto3g"
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
_, civec = myci.kernel()
_, loc = pyscf_helper.sep_cisdvec(norb, nelec)

def ci_singles():
    dt = 0.1
    vec = np.copy(civec)
    vec[loc[2]:] = 0
    vec /= np.linalg.norm(vec)
    e_ci = pyscf_helper.cisd_energy_from_vec(vec, mf)
    c0, c1, c2 = nocisd.ucisd_amplitudes(mf, civec=vec)
    t1 = nocisd.c2t_singles(c1, dt=dt)
    coeffs = np.array([c0] + [1/dt,]*4)
    coeffs /= np.linalg.norm(coeffs)
    t = slater.add_tvec_hf(t1)
    rmats = slater.tvecs_to_rmats(t, nvir, nocc)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    print(e_ci)
    print(E)

def ci_doubles_ab():
    dt = 0.05
    vec = np.copy(civec)
    vec[loc[0]:loc[2]] = 0
    vec[loc[3]:] = 0
    vec /= np.linalg.norm(vec)
    e_ci = pyscf_helper.cisd_energy_from_vec(vec, mf)
    c0, c1, c2 = nocisd.ucisd_amplitudes(mf, civec=vec)
    t2, lam2s = nocisd.c2t_doubles(c2, dt=dt)
    tab = np.asarray(t2[1])
    print(tab)
    cab = np.concatenate([lam2s[1],]*4)/dt**2
    t = slater.add_tvec_hf(tab)
    rmats = slater.tvecs_to_rmats(t, nvir, nocc)
    coeffs = np.concatenate([[c0], cab])
    coeffs /= np.linalg.norm(coeffs)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
    print(E)

ci_doubles_ab()
