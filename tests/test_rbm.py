import numpy as np
from pyscf import gto, scf, fci
import sys
sys.path.append('../')
import rbm, noci, slater
import jax.numpy as jnp

def test_hiddens_to_coeffs():
    h = [0, 1]
    rep = 3 
    coeff = rbm.hiddens_to_coeffs(h, rep)
    print(coeff)



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



def test_einsum():
    
    # test rdm1
    nvecs = 10
    norb = 14
    nocc = 6
    rmats = np.random.rand(nvecs, 2, norb, nocc)
    mo_coeff = np.random.rand(2, norb, norb)
    # first calculate metric and thus overlap
    metrics_all = jnp.einsum('nsji, msjk -> nmsik', rmats.conj(), rmats)
    smat = jnp.prod(jnp.linalg.det(metrics_all), axis=-1)
    smat_ref = jnp.array(noci.full_ovlp_w_rotmat(rmats))
    assert jnp.allclose(smat, smat_ref)

    # transition density matrices
    inv_metrics = jnp.linalg.inv(metrics_all)
    assert jnp.allclose(inv_metrics[0, 1, 0], jnp.linalg.inv(metrics_all[0, 1, 0]))
    sdets = jnp.einsum("sij, nsjk -> nsik", mo_coeff, rmats)
    sdets_ref = slater.gen_determinants(mo_coeff, rmats) 
    assert jnp.allclose(sdets, sdets_ref)
    trdms = jnp.einsum("msij, nmsjk, nslk -> nmsil", sdets, inv_metrics, sdets.conj())
    
    i = 2; j = 1
    rdm01 = slater.make_trans_rdm1(sdets[i], sdets[j], omat=metrics_all[i, j], return_ovlp=False)
    print(jnp.linalg.norm(rdm01 - trdms[i, j]))

test_einsum()