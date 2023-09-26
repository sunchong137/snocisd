import numpy as np
from pyscf import gto, scf, ci
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd
from noci_jax import slater, pyscf_helper
import logging

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
mol.basis = "631g"
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


def test_get_ci_coeff():

    _, _, c2 = nocisd.ucisd_amplitudes(myci, flatten_c2=True)
    for i in [0,2]:
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-10 
    # C_aabb is not symmetric


def test_compress():
    dt = 0.2
    tmats, coeffs = nocisd.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-5)
    nvir, nocc = tmats.shape[2:]
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
    # print("compress: ", E)
    assert np.allclose(E, e_cisd)

