import numpy as np
from noci_jax import math_helper


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
    
    n_nz = math_helper.check_linear_depend(ovlp_mat, 1e-10)
    assert np.allclose(n_nz, num_ind)
