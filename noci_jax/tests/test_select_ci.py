import numpy as np
from noci_jax import select_ci, slater, pyscf_helper
from pyscf import gto, scf


def test_linear_dependency():

    n = 100
    m = np.random.rand(n, n)
    q, r = np.linalg.qr(m)
    num_ind = np.random.randint(n)
    num_ld = n - num_ind
    v = np.zeros_like(q)
    v_ind = q[:, :num_ind]
    v[:, :num_ind] = v_ind
    rot_mat = np.random.rand(num_ind, num_ld)
    v_ld = v_ind@rot_mat 
    v[:, num_ind:] = v_ld
    
    ovlp_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ov = v[:, i].T@v[:, j]
            ovlp_mat[i, j] = ov 
            ovlp_mat[j, i] = ov 
    
    n_nz = select_ci.check_linear_depend(ovlp_mat, 1e-10)
    assert np.allclose(n_nz, num_ind)

def test_metric():
    nocc = 2
    nvir = 2
    norb = nocc + nvir
    t_vecs = np.load("./data/h4_R1.5_sto3g_ndet1.npy")
    rmats = slater.tvecs_to_rmats(t_vecs, nvir, nocc)
    r_n = np.random.rand(2, 2, norb, nocc)
    r_n[0,0,:nocc] = np.eye(nocc)
    r_n[0,1,:nocc] = np.eye(nocc)
    r_n[1] = rmats[1]
    
    mr = select_ci.criteria_ovlp(rmats, r_n)
    assert mr[1] < 1e-10


def test_criteria():
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

    m, e = select_ci.criteria_all(rmats, r_n, mo_coeff, h1e, h2e)
    print(m, e)
    # single det
    m1, e1 = select_ci.criteria_all_single_det(rmats, r_n[0], mo_coeff, h1e, h2e)
    print(m1, e1)

    t_select = select_ci.select_tvecs(t_vecs, t_new, mo_coeff, h1e, h2e, nocc, nvir)
    print(len(t_select))

test_criteria()