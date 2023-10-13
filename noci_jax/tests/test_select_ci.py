import numpy as np
from noci_jax import select_ci, slater


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
    
    mr = select_ci.metric_residual(rmats, r_n)
    assert mr[1] < 1e-10
