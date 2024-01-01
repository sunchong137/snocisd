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