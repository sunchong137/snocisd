import numpy as np
from pyscf import gto, scf, ci, fci
from pyscf.lib import numpy_helper
import scipy
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax.cisd import compress
from noci_jax import slater, pyscf_helpers

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

h1e, h2e, e_nuc = pyscf_helpers.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helpers.get_mos(mf)
e_hf = mf.energy_tot()

def test_get_ci_coeff():

    c1, c2 = compress.get_cisd_coeffs_uhf(mf, flatten_c2=True)
    for i in range(3):
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-6  # change to a larger number if basis larger
    

def test_singles_c2t():
    c1, c2 = compress.get_cisd_coeffs_uhf(mf)
    nvir, nocc = c1[0].shape
    dt = 0.05
    tmats = compress.c2t_singles(c1, dt)
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    ovlp = slater.metric_rmats(rmats[0], rmats[1])

    r0 = np.zeros((norb, nocc))
    r0[:nocc, :nocc] = np.eye(nocc)
    r_hf = np.array([[r0, r0]])
    r_all = np.vstack([r_hf, rmats])
    
    E = slater.noci_energy(r_all, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    assert E <= e_hf


def test_doubles_c2t():

    # occ_ref = np.random.rand(nocc, nocc)
    # occ_ref += occ_ref.T
    c1, c2 = compress.get_cisd_coeffs_uhf(mf)
    dt = 0.1
    t1 = compress.c2t_singles(c1, dt)
    t2 = compress.c2t_doubles(c2, dt=dt, tol=8e-2)
    t_hf = np.zeros((1, 2, nvir, nocc))
    tmats = t2
    # tmats = np.vstack([t1, t2])
    tmats = np.vstack([t_hf, tmats])

    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc, occ_mat=occ_ref)
    # ovlp = slater.metric_rmats(rmats[0], rmats[1])


    h, s = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True, lc_coeffs=None, e_nuc=e_nuc)
    print(np.linalg.det(s))
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    print(E)

test_doubles_c2t()