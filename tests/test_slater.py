import numpy as np
import time
from numpy import linalg as nl 
from scipy import linalg as sl
#import unittest
from pyscf import gto, scf
import sys
sys.path.append('../')
sys.path.append('./')
import slater, noci


def test_rotation():
    pass

def test_gen_dets():
    nvir = 3
    nocc = 2
    norb = nvir + nocc 
    mo_coeff = np.random.rand(2, norb, norb)


    I = np.array([np.eye(nocc), np.eye(nocc)])
    t1 = np.random.rand(2, nvir, nocc) 
    r1 = np.concatenate([I, t1], axis=1)
    t2 = np.random.rand(2, nvir, nocc) 
    r2 = np.concatenate([I, t2], axis=1)

    tmats = [t1, t2]
    rmats = [r1, r2]

    sdets1 = slater.gen_determinants(mo_coeff, rmats)
    sdets2 = slater.gen_dets_thouless(mo_coeff, tmats)

    assert np.allclose(sdets1, sdets2)


def test_t2r():
    nvir = 3
    nocc = 2

    # two spins
    a = np.random.rand(2, nvir, nocc)
    I = np.eye(nocc)
    r = slater.thouless_to_rotation(a)
    ref = np.zeros((2, nvir+nocc, nocc))
    ref[0][:nocc] = I 
    ref[1][:nocc] = I
    ref[0][nocc:] = a[0]
    ref[1][nocc:] = a[1]

    assert np.allclose(r, ref)

    # one spin
    a = np.random.rand(nvir, nocc)
    r = slater.thouless_to_rotation(a)
    ref = np.zeros((nvir+nocc, nocc))
    ref[:nocc] = I
    ref[nocc:] = a 

    assert np.allclose(r, ref)

def test_t2r_all():
    nvir = 3
    nocc = 2
    norb = nvir + nocc
    mo_coeff = np.random.rand(2, norb, norb)
    max_nt = None
    doubles = noci.gen_thouless_doubles(nocc, nvir, max_nt=max_nt)
    rmats = slater.thouless_to_rotation_all(doubles)
    dets1 = slater.gen_determinants(mo_coeff, rmats, normalize=True)
    dets2 = slater.gen_dets_thouless(mo_coeff, doubles, normalize=True)
    assert np.allclose(dets1, dets2)


def test_normalize():
    # cholesky
    nvir = 8
    nocc = 6
    tmat = np.random.rand(nvir, nocc) 
    rmat = np.zeros((nvir + nocc, nocc))
    rmat[:nocc] = np.eye(nocc)
    rmat[nocc:] = tmat 
    t1 = time.time()
    rmat_n = slater.orthonormal_rotmat_spinless(rmat)
    t2 = time.time()
    r = slater.normalize_rotmat_spinless_old(rmat)
    #print(r.T @ r)
    t3 = time.time() 
    #print("Time diff: ", t2-t1, t3-t2)
    w = rmat_n.T @ rmat_n 
    assert np.allclose(w, np.eye(nocc))

def test_norm_time():
    nvir = 10
    nocc = 8
    tmat = np.random.rand(nvir, nocc) 
    rmat = np.zeros((nvir + nocc, nocc))
    rmat[:nocc] = np.eye(nocc)
    rmat[nocc:] = tmat 

    t1 = time.time()
    rnorm = np.linalg.norm(rmat, axis=0)
    rmat_n = np.divide(rmat, rnorm)
    t2 = time.time()


    rmat_n2 = np.zeros_like(rmat)
    nocc = rmat.shape[-1]
    for i in range(nocc):
        rmat_n2[:, i] = rmat[:, i]/np.linalg.norm(rmat[:, i])

    t3 = time.time() 
    print(t2-t1, t3-t2)
    assert np.allclose(rmat_n, rmat_n2)



a = 1.0
mol = gto.Mole(
    verbose = 0,
    atom = '''
    H 0  0  0
    H 0  0  {}
    H 0  0  {}
    H 0  0  {}'''.format(a, a*2, a*3),
    basis = 'sto-6g'
)

mol.build()
mf = scf.UHF(mol)
mf.conv_tol = 1e-10
mf.kernel()
norb = mol.nao
nocc = mol.nelec[0]
nvir = norb - nocc
mo_coeff = mf.mo_coeff
ao_ovlp = mf.get_ovlp()

h1e = mf.get_hcore()
eri = mol.intor('int2e')
h2e = eri #ao2mo.restore(1, eri, mol.nao_nr())

dm = mf.make_rdm1()

h1e = mf.get_hcore()
vhf = mf.get_veff(mol, dm)
e1 = np.einsum('ij,ji->', h1e, (dm[0]+dm[1])).real
e_coul = np.einsum('ij,ji->', vhf[0], dm[0]).real * 0.5
e_coul += np.einsum('ij,ji->', vhf[1], dm[1]).real * 0.5

def test_get_jk():
    vhf = slater.get_jk_pyscf(mf, dm)
    jk = slater.get_jk(h2e, dm)
    assert np.allclose(vhf, jk)

    # test trans_density_matrix()         
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    rotmat2 = np.concatenate([np.eye(nocc), np.random.rand(nvir, nocc)], axis=0) 
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = slater.normalize_rotmat(rotmat2) 
    mo_coeff_sd1 = slater.rotation(mo_coeff, rotmat1)
    mo_coeff_sd2 = slater.rotation(mo_coeff, rotmat2)
    dm1, _ = slater.make_trans_rdm1(mo_coeff_sd1, mo_coeff_sd2, ao_ovlp=ao_ovlp)
    vhf = slater.get_jk_pyscf(mf, dm1)
    jk = slater.get_jk(h2e, dm1)
    assert np.allclose(vhf, jk)


def test_ovlp_matrix():

    rotmat0 = np.zeros((norb, nocc))
    rotmat0[:nocc] = np.eye(nocc)
    det0 = [mo_coeff[0][:, :nocc], mo_coeff[1][:, :nocc]]
    rotmat = np.zeros((norb, nocc))
    rotmat[:nocc] = np.eye(nocc)
    rotmat[nocc:] = np.random.rand(nvir, nocc)*2
    rotmat = slater.normalize_rotmat(rotmat)
    det1 = slater.rotation(mo_coeff, rotmat)

    omat1 = slater.metric_sdet(det0, det1, ao_ovlp=ao_ovlp)
    omat3 = slater.metric_rotmat(rotmat0, rotmat)
    assert np.allclose(omat1, omat3)

def test_overlap():
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    rotmat2 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = slater.normalize_rotmat(rotmat2)

    mo_coeff1 = slater.rotation(mo_coeff, rotmat1)
    mo_coeff2 = slater.rotation(mo_coeff, rotmat2)
    overlap1 = slater.ovlp_sdet(mo_coeff1, mo_coeff2, ao_ovlp=ao_ovlp)
    overlap2 = slater.ovlp_rotmat(rotmat1, rotmat2)
    assert np.allclose(overlap1, overlap2)

def test_rdms():
    # test trans_density_matrix()         
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    rotmat2 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0) 
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = slater.normalize_rotmat(rotmat2) 
    mo_coeff_sd1 = slater.rotation(mo_coeff, rotmat1)
    mo_coeff_sd2 = slater.rotation(mo_coeff, rotmat2)
    ovlp = slater.ovlp_sdet(mo_coeff_sd1, mo_coeff_sd2, ao_ovlp=ao_ovlp)
    trans_rdm_1, _ = slater.make_trans_rdm1(mo_coeff_sd1, mo_coeff_sd2, ao_ovlp=ao_ovlp)
    # trans_rdm_1 /= ovlp
    # reference 
    rdm = mf.make_rdm1()

    # reference 2
    mo_coeff_occ = [mo_coeff[0][:, :nocc], mo_coeff[1][:, :nocc]]
    rdm2 = [np.dot(mo_coeff_occ[0], mo_coeff_occ[0].T.conj()),  np.dot(mo_coeff_occ[1], mo_coeff_occ[1].T.conj())]

    assert np.allclose(rdm, rdm2)
    assert np.allclose(trans_rdm_1, rdm)

def test_trans_rdm():
    # test trans_density_matrix()         
    tol = 1e-3
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    shift = np.concatenate([np.eye(nocc), np.ones((nvir, nocc))], axis=0) 
    rotmat2 = rotmat1 + shift * tol
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = slater.normalize_rotmat(rotmat2) 

    mo_coeff_sd1 = slater.rotation(mo_coeff, rotmat1)
    mo_coeff_sd2 = slater.rotation(mo_coeff, rotmat2)
    trans_rdm_1, _ = slater.make_trans_rdm1(mo_coeff_sd1, mo_coeff_sd2, ao_ovlp=ao_ovlp)
    

    # rho^2 = rho
    rho = trans_rdm_1[0]
    rho_rho = rho@ao_ovlp@rho
    assert np.allclose(rho, rho_rho)

    # trace(rho) = N
    N = np.trace((trans_rdm_1[0] + trans_rdm_1[1])@ao_ovlp)
    assert np.allclose(N, nocc*2)

def test_energy():
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    rotmat2 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0) 
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = slater.normalize_rotmat(rotmat2)   
    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)
    trans_rdm, _ = slater.make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp)
    ovlp = slater.ovlp_rotmat(rotmat1, rotmat2)
    energy_1 = slater.trans_hamilt_dm(trans_rdm, ovlp, h1e, h2e)
    energy_ref = slater.trans_hamilt_pyscf(mf, trans_rdm/ovlp, ovlp=1)
    assert np.allclose(energy_1, energy_ref)

def test_trans_hamilt():
    tol = 1e-3 
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)

    i = np.random.randint(nocc)
    a = np.random.randint(nvir)
    rotmat2 = np.zeros((norb, nocc))
    rotmat2[:nocc] = np.eye(nocc)
    rotmat2[i,i] = 0.01
    rotmat2[nocc+a, i] = 1
    rotmat2 = slater.normalize_rotmat(rotmat2)
    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)
    trans_rdm, _ = slater.make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp)

    ovlp = slater.ovlp_rotmat(rotmat1, rotmat2)
    energy = slater.trans_hamilt_dm(trans_rdm, ovlp, h1e, h2e)
    energy_2 = slater.trans_hamilt_pyscf(mf,trans_rdm, ovlp)
    assert np.allclose(energy, energy_2)

def test_trans_hamilt_sdet():
    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    i = np.random.randint(nocc)
    a = np.random.randint(nvir)
    rotmat2 = np.zeros((norb, nocc))
    rotmat2[:nocc] = np.eye(nocc)
    rotmat2[i,i] = 0.01
    rotmat2[nocc+a, i] = 1
    rotmat2 = slater.normalize_rotmat(rotmat2)

    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)

    e1 = slater.trans_hamilt(sdet1, sdet2, h1e, h2e, ao_ovlp=ao_ovlp)
    
    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)
    trans_rdm, _ = slater.make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp)
    ovlp = slater.ovlp_rotmat(rotmat1, rotmat2)
    e2 = slater.trans_hamilt_dm(trans_rdm, ovlp, h1e, h2e)

    assert np.allclose(e1, e2)

def test_trans_hamilt_all():

    rotmat1 = np.concatenate([np.eye(nocc), np.zeros((nvir, nocc))], axis=0)
    i = np.random.randint(nocc)
    a = np.random.randint(nvir)
    rotmat2 = np.zeros((norb, nocc))
    rotmat2[:nocc] = np.eye(nocc)
    rotmat2[i,i] = 0.01
    rotmat2[nocc+a, i] = 1
    rotmat2 = slater.normalize_rotmat(rotmat2)

    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)

    e1 = slater.trans_hamilt(sdet1, sdet2, h1e, h2e, ao_ovlp=ao_ovlp)
    
    sdet1 = slater.rotation(mo_coeff, rotmat1)
    sdet2 = slater.rotation(mo_coeff, rotmat2)
    trans_rdm, _ = slater.make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp)
    ovlp = slater.ovlp_rotmat(rotmat1, rotmat2)
    e2 = slater.trans_hamilt_dm(trans_rdm, ovlp, h1e, h2e)

    # e3, ovlp3, grad = slater.trans_hamilt_all(rotmat1, rotmat2, h1e, h2e, mo_coeff, ao_ovlp=ao_ovlp, get_grad=True)

    # assert np.allclose(e2, e3)
    # assert np.allclose(ovlp, ovlp3)

test_trans_hamilt_all()

def test_overlap():
    rotmat = np.concatenate([np.eye(nocc), np.random.rand(nvir, nocc)], axis=0)
    rotmat =slater.normalize_rotmat(rotmat)
    assert np.allclose(slater.ovlp_rotmat(rotmat, rotmat), 1)

def test_normalize():
    rotmat = np.concatenate([np.eye(nocc), np.random.rand(nvir, nocc)], axis=0)
    rotmat = slater.normalize_rotmat(rotmat)
    sd = slater.rotation(mo_coeff, rotmat)
    ovlp_rot = slater.ovlp_rotmat(rotmat, rotmat)
    ovlp = slater.ovlp_sdet(sd, sd, ao_ovlp=ao_ovlp)
    
    assert np.allclose(ovlp, 1)
    assert np.allclose(ovlp_rot, 1)

def test_numpy():
    '''
    Test the built-in functions I use.
    '''
    # inverse 
    rotmat1 = np.concatenate([np.eye(nocc), np.random.rand(nvir, nocc)], axis=0)
    rotmat1 = slater.normalize_rotmat(rotmat1)
    rotmat2 = np.concatenate([np.eye(nocc), np.random.rand(nvir, nocc)], axis=0)
    rotmat2 = slater.normalize_rotmat(rotmat2)
    ovlp_mat = slater.metric_rotmat(rotmat1, rotmat2)
    inv_ovlp_mat = np.linalg.inv(ovlp_mat)
    mat = np.dot(ovlp_mat, inv_ovlp_mat)
    mat2 = np.dot(inv_ovlp_mat, ovlp_mat)
    assert np.allclose(mat, np.eye(nocc))
    assert np.allclose(mat2, np.eye(nocc))

    rand_mat = np.array([[1,2],[3,2]])
    det_ref = -4 
    det = np.linalg.det(rand_mat)
    assert np.allclose(det_ref, det)


