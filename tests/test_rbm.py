import jax.numpy as jnp
import numpy as np
from scipy import linalg as sla
import sys
sys.path.append("../")
import rbm 


def test_solve_lc():
    n = 10
    hmat = np.random.rand(n, n)
    hmat = hmat + hmat.T 
    smat = np.random.rand(n, n) * 0.1 
    smat = smat + smat.T 
    smat += np.eye(n)

    # use scipy
    e0, v0 = sla.eigh(hmat, b=smat)
    esla = e0[0] 
    vsla = v0[:, 0]

    hmat = jnp.array(hmat)
    smat = jnp.array(smat)

    e, v = rbm.solve_lc_coeffs(hmat, smat, return_vec=True)
    v = np.array(v)

    h = v.conj().T.dot(hmat).dot(v)
    s = v.conj().T.dot(smat).dot(v)
    e2 = h/s
    assert np.allclose(e, esla)
    assert np.allclose(e, e2)
    assert np.allclose(v, vsla)

def test_hiddens():
    nvecs = 4
    print(rbm.trucated_hidden_coeffs(nvecs))

