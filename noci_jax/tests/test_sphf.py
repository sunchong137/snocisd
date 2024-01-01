import numpy as np
from noci_jax import sphf 

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
    r_ref = np.array([0.21812657, 1.03675535, 2.1048373,  2.92346608])
    w_ref = np.array([ 0.11130528,  0.01199616, -0.43682783, -0.00413635])
    assert np.allclose(r, r_ref)
    assert np.allclose(w, w_ref)

def test_rotation():
    beta = 0.5321710512913808
    norb = 2
    r = sphf.gen_rotations_ao(beta, norb)
    r_ref = np.array([[ 0.96480762,  0.,          0.26295675,  0. ],
            [ 0.,          0.96480762,  0.,          0.26295675  ],
            [-0.26295675,  0.,          0.96480762, 0.          ],
            [ 0.,         -0.26295675,  0.,          0.96480762  ]])
    
    assert np.allclose(r, r_ref)

test_rotation()