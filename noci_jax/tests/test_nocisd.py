import numpy as np
from pyscf import gto, scf, ci, fci
from pyscf.lib import numpy_helper
import scipy
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
# mo1 = mf.stability()[0]                                                             
# init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
# mf.kernel(init) 

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()


def test_get_ci_coeff():

    _, _, c2 = nocisd.get_cisd_coeffs_uhf(mf, flatten_c2=True)
    for i in [0,2]:
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-10 
    # C_aabb is not symmetric

def test_singles_c2t():
    _, c1, _ = nocisd.get_cisd_coeffs_uhf(mf)
    nvir, nocc = c1[0].shape
    dt = 0.05
    tmats = nocisd.c2t_singles(c1, dt)
    r0 = np.zeros((2, nvir, nocc))
    t_all = np.vstack(r0 + tmats)
    rmats = slater.tvecs_to_rmats(t_all, nvir, nocc)
    ovlp = slater.metric_rmats(rmats[0], rmats[1])
    
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    assert E <= e_hf


def test_doubles_c2t():

    # occ_ref = np.random.rand(nocc, nocc)
    # occ_ref += occ_ref.T
    occ_ref = None
    _, c1, c2 = nocisd.get_cisd_coeffs_uhf(mf)
    dt = 0.1
    t1= nocisd.c2t_singles(c1, dt)
    _t2, lams = nocisd.c2t_doubles(c2, dt=dt, tol=8e-2)
    t2 = np.vstack(_t2)
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


def test_compress():

    tmats, coeffs = nocisd.compress(mf,dt1=0.1, dt2=0.01, tol2=1e-5)
    nvir, nocc = tmats.shape[2:]
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
    print(E)


def test_c2_symm():
    norm = np.linalg.norm
    # myci = ci.UCISD(mf)
    # _, civec = myci.kernel()
    c0, c1, c2 = nocisd.get_cisd_coeffs_uhf(mf, flatten_c2=False)
    print(c2[0][0,1,0,1])
    # print(np.diag(c2[1]))
    # print(norm(c2[0]-c2[0].T.conj()))
    # print(norm(c2[1]-c2[1].T.conj()))
    # print(norm(c2[2]-c2[2].T.conj()))


def test_cisd():
    '''
    Try out PySCF cisd.
    '''
    myci = ci.UCISD(mf)  
    e_hf = mf.e_tot
    eris = myci.ao2mo(mf.mo_coeff)
    e_corr, civec = myci.kernel()
    print(e_corr)
    ci_n = myci.contract(civec, eris)
    e_diff = np.dot(civec.conj().T, ci_n)
    e_cisd = e_hf + e_diff
    print(e_cisd)

test_cisd()