import numpy as np
from noci_jax import sphf, slater
from noci_jax.misc import pyscf_helper
from pyscf import gto, scf

def test_wignerd():
    beta = np.random.uniform(0, np.pi)
    # l=1/2
    d1 = sphf.wignerd(beta, 1/2, 1/2, 1/2)
    assert np.allclose(d1, np.cos(beta/2))
    d2 = sphf.wignerd(beta, 1/2, 1/2, -1/2)
    assert np.allclose(d2, -np.sin(beta/2))
    # l=1
    d3 = sphf.wignerd(beta, 1, 1, 1)
    assert np.allclose(d3, 1/2 * (1 + np.cos(beta)))
    d4 = sphf.wignerd(beta, 1, 1, 0)
    assert np.allclose(d4, -np.sin(beta) / np.sqrt(2))
    d5 = sphf.wignerd(beta, 1, 1, -1)
    assert np.allclose(d5, 1/2 * (1 - np.cos(beta)))
    d6 = sphf.wignerd(beta, 1, 0, 0)
    assert np.allclose(d6, np.cos(beta))

def test_root_weight():
    '''
    Taking the output from Tom Henderson's code as references.
    '''
    ngrid = 4
    j = 2
    m = 1
    r, w = sphf.gen_roots_weights(ngrid, j, m)
    r1 = sphf.gen_roots(ngrid)
    r_ref = np.array([0.21812657, 1.03675535, 2.1048373,  2.92346608])
    w_ref = np.array([ 0.11130528,  0.01199616, -0.43682783, -0.00413635])
    assert np.allclose(r, r_ref)
    assert np.allclose(r, r1)
    assert np.allclose(w, w_ref)


def test_rotation():
    beta = 0.5321710512913808
    norb = 2
    r = sphf.gen_rotations_ao(beta, norb)
    r_ref = np.array([[ 0.96480762,  0.,          0.26295675,  0. ],
            [ 0.,          0.96480762,  0.,          0.26295675  ],
            [-0.26295675,  0.,          0.96480762, 0.          ],
            [ 0.,         -0.26295675,  0.,          0.96480762  ]])
    beta = np.pi 
    # r1 = sphf.gen_rotations_ao(beta, norb)
    # print(r1)
    assert np.allclose(r, r_ref)

def test_transmat():
    norb = 4
    ngrid = 2
    H = np.random.rand(norb, norb)
    H += H.T 
    e, v = np.linalg.eigh(H)
    H = np.random.rand(norb, norb)
    H += H.T 
    e, v1 = np.linalg.eigh(H)
    mo = np.array([v, v1]) 
    U = sphf.gen_transmat_sphf(mo, ngrid)

def test_bshf_energy():
    from pyscf import fci
    bl = 1.4
    mol = gto.Mole()
    mol.atom = f'''
    H   0   0   0
    H   0   0   {bl}
    H   0   0   {bl*2}
    H   0   0   {bl*3}
    '''
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
    myci = fci.FCI(mf)
    e, v = myci.kernel()
    print("FCI energy: ", e)

    ngrid = 4
    U = sphf.gen_transmat_sphf(mo_coeff, ngrid)
    R = U[:,:,:,:nocc]
    R_hf = np.zeros((1, 2, norb, nocc))
    R_hf[0, 0, :nocc] = np.eye(nocc)
    R_hf[0, 1, :nocc] = np.eye(nocc)
    rmats = np.vstack([R_hf, R])
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    
    # compare to half_spin 
    r_hsp = slater.half_spin(R_hf, mo_coeffs=mo_coeff)
    rmats_hsp = np.vstack([R_hf, r_hsp])
    E2 = slater.noci_energy(rmats_hsp, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    print(E, E2)

test_bshf_energy()