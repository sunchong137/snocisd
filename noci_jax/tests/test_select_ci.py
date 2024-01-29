import numpy as np
from noci_jax import select_ci, slater 
from noci_jax.misc import pyscf_helper
from pyscf import gto, scf


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
t_vecs = np.load("./data/h4_R1.5_sto3g_ndet1.npy")
t_vecs = slater.add_tvec_hf(t_vecs)
rmats = slater.tvecs_to_rmats(t_vecs, nvir, nocc)
t_new = np.random.rand(3, 2, nvir, nocc)
t_new[1] = t_vecs[1]
t_new[1] = t_new[1] + np.random.rand(2, nvir, nocc)*0.001
r_n = slater.tvecs_to_rmats(t_new, nvir, nocc)

def test_select_together():    
    # select with metric
    m_tol = 1e-6
    e_tol = 1e-4
    r_select = select_ci.select_rmats(rmats, r_n, mo_coeff, h1e, h2e, m_tol=m_tol, 
                                        e_tol=e_tol, max_ndets=None)
    e_old = slater.noci_energy(rmats, mo_coeff, h1e, h2e)
    e_new = slater.noci_energy(r_select, mo_coeff, h1e, h2e)
    # print(e_new, e_old)
    assert e_new <= e_old

def test_select_metric():
    m_tol = 1e-5 
    r_select1 = select_ci.select_rmats_ovlp(rmats, r_n, m_tol=m_tol, max_ndets=None, return_indices=False)
    r_select2 = select_ci.select_rmats(rmats, r_n, mo_coeff, h1e, h2e, m_tol=m_tol, 
                                        e_tol=None, max_ndets=None)
    assert np.allclose(r_select1, r_select2)

def test_select_energy():
    e_tol = 1e-4
    r_select = select_ci.select_rmats_energy(rmats, r_n, mo_coeff, h1e, h2e, 
                                        e_tol=e_tol, max_ndets=None)
    e_old = slater.noci_energy(rmats, mo_coeff, h1e, h2e)
    e_new = slater.noci_energy(r_select, mo_coeff, h1e, h2e)
    # print(e_new, e_old)
    assert e_new <= e_old

def test_select_slow():
    m_tol = 1e-6
    e_tol = 1e-4
    r_select = select_ci.select_rmats_slow(rmats, r_n, mo_coeff, h1e, h2e, m_tol=m_tol, 
                                        e_tol=e_tol, max_ndets=None)

    e_old = slater.noci_energy(rmats, mo_coeff, h1e, h2e)
    e_new = slater.noci_energy(r_select, mo_coeff, h1e, h2e)
    assert e_new <= e_old
