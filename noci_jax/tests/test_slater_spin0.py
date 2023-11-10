import numpy as np
from noci_jax import slater_spin0, pyscf_helper, slater
from pyscf import gto, scf

def test_tvecs_to_rmats():
    
    nocc = 2
    nvir = 3
    occ_mat = np.eye(2)/2
    occ_mat[0,1] = -0.1
    occ_mat += occ_mat.T

    tvecs = np.arange(nocc*nvir)

    rmats = slater_spin0.tvecs_to_rmats(tvecs, nvir, nocc, occ_mat=occ_mat)
    ref = np.array([[[ 1.,  -0.1], [-0.1,  1. ], [ 0.,1. ],[ 2., 3. ], [ 4.,5. ]]])
    assert np.allclose(rmats, ref)

    ovlp = slater_spin0.metric_rmats(rmats[0], rmats[0])
    print(ovlp)


test_tvecs_to_rmats()
