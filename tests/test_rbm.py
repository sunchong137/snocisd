import numpy as np
from pyscf import gto, scf, fci
import sys
sys.path.append('../')
import rbm
import jax.numpy as jnp

def test_hiddens_to_coeffs():
    h = [0, 1]
    rep = 3 
    coeff = rbm.hiddens_to_coeffs(h, rep)
    print(coeff)
test_hiddens_to_coeffs()


def test_expand_vectors():

    # test empty vecs
    rbm_all = rbm.expand_vecs([])
    assert np.allclose(rbm_all, [])

    vec = np.random.rand(4)
    rbm_all = rbm.expand_vecs([vec])
    assert np.allclose(rbm_all[0], vec*0)
    
def test_add_vec():
    vec = np.random.rand(4) 
    vec_list = [np.zeros(4)]
    new_vec = rbm.add_vec(vec, vec_list)
    assert np.allclose(new_vec[0], vec)

def test_metrics_all():
    n = 10 
    norb = 20
    nocc = 8

    rmats = jnp.array(np.random.rand(n, 2, norb, nocc))
    m0 = jnp.dot(rmats[0,0].T.conj(), rmats[0,0])
    metrics = rbm.metrics_all(rmats)
    print(m0 - metrics[0,0,0])




test_metrics_all()