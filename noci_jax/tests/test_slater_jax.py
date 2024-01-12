# Copyright 2023 by NOCI_Jax developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import numpy as np
from scipy import linalg as sla
from noci_jax import slater_jax
from noci_jax.misc import pyscf_helper
from pyscf import gto, scf

def test_tvecs_to_rmats():
    
    nocc = 2
    nvir = 3
    occ_mat = np.eye(2)/2
    occ_mat[0,1] = -0.1
    occ_mat += occ_mat.T

    tvecs = np.arange(nocc*nvir*2)

    rmats = slater_jax.tvecs_to_rmats(tvecs, nvir, nocc, occ_mat=occ_mat)
    ref = np.array([[[[ 1.,  -0.1], [-0.1,  1. ], [ 0.,1. ],[ 2., 3. ], [ 4.,5. ]],
                     [[ 1.,-0.1],[-0.1,  1. ],[ 6., 7. ],[ 8., 9. ],[10.,  11. ]]]])
    assert np.allclose(rmats, ref)


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

    e, v = slater_jax.solve_lc_coeffs(hmat, smat, return_vec=True)
    v = np.array(v)

    h = v.conj().T.dot(hmat).dot(v)
    s = v.conj().T.dot(smat).dot(v)
    e2 = h/s
    assert np.allclose(e, esla)
    assert np.allclose(e, e2)
    assert np.allclose(v, vsla)


def test_make_rdm1():
    # construct molecule
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
    mol.unit='angstrom'
    mol.basis = "sto6g"

    # mol.symmetry = True
    mol.build()

    mf = scf.UHF(mol)

    # Hartree-Fock
    mf.kernel(verbose=0, tol=1e-10)
    e_ref = mf.e_tot
    
    # orthogonalize ao overlap matrix
    h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=True)
    mf.kernel(verbose=0, tol=1e-10)
    
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)

    t_vecs = np.load("./data/h4_R1.5_sto3g_ndet1.npy")

    rmats = slater_jax.tvecs_to_rmats(t_vecs, nvir, nocc)
    hmat, smat = slater_jax.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    energy, c = slater_jax.solve_lc_coeffs(hmat, smat, return_vec=True)
    rdm1 = slater_jax.make_rdm1(rmats, mo_coeff, c)
    ne_a = np.sum(np.diag(rdm1[0]))
    ne_b = np.sum(np.diag(rdm1[1]))

    assert np.allclose(ne_a, 2)
    assert np.allclose(ne_b, 2)


def test_make_rdm12s():
    # construct molecule
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
    mol.unit='angstrom'
    mol.basis = "sto6g"

    # mol.symmetry = True
    mol.build()

    mf = scf.UHF(mol)

    # Hartree-Fock
    mf.kernel(verbose=0, tol=1e-10)
    e_ref = mf.e_tot
    
    # orthogonalize ao overlap matrix
    h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=True)
    mf.kernel(verbose=0, tol=1e-10)
    
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)

    t_vecs = np.random.rand(1, 2*nvir*nocc)-0.5
    t_vecs[0] = 0

    rmats = slater_jax.tvecs_to_rmats(t_vecs, nvir, nocc)
    hmat, smat = slater_jax.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    energy, c = slater_jax.solve_lc_coeffs(hmat, smat, return_vec=True)
    rmats = rmats.at[0].set(rmats[0]/np.sqrt(smat[0,0]))
    rdm1s, rdm2s = slater_jax.make_rdm12(rmats, mo_coeff, c)
    rdm2s_ud_diag = np.zeros((norb, norb))
    for i in range(norb):
        for j in range(norb):
            rdm2s_ud_diag[i,j] = rdm2s[1][i,i,j,j]
    rdm1s_n, rdm2s_ud = slater_jax.make_rdm12_diag(rmats, mo_coeff, c)
    # rdm1 = slater_jax.make_rdm1(rmats, mo_coeff, c)
    ne_a = np.sum(np.diag(rdm1s[0]))
    ne_b = np.sum(np.diag(rdm1s[1]))

    dm2_hf = mf.make_rdm2()

    assert np.allclose(rdm1s_n[0], np.diag(rdm1s[0]))
    assert np.allclose(dm2_hf[0], rdm2s[0])
    assert np.allclose(dm2_hf[1], rdm2s[1]) 
    assert np.allclose(dm2_hf[2], rdm2s[3])
    assert np.allclose(rdm2s_ud, rdm2s_ud_diag)

    E1 = jnp.einsum("ij, sji ->", h1e, rdm1s)
    E2 = jnp.einsum("ijkl, sjilk ->", h2e, rdm2s)
    E = E1 + 0.5*E2

    assert np.allclose(E, mf.energy_elec()[0])


def test_orthonormal():

    nvir = 5
    nocc = 3
    norb = nvir + nocc
    tmats = np.random.rand(nvir, nocc)
    r = slater_jax.orthonormal_mos(tmats)
    I = np.eye(norb)
    assert np.allclose(np.dot(r.T, r), I)


    nt = 3
    spin = 2
    tmats = np.random.rand(nt, spin, nvir, nocc)
    rotm = slater_jax.orthonormal_mos(tmats)
    x = np.moveaxis(rotm, -2, -1).conj() @ rotm
    I_all = np.tile(I, nt*spin).T.reshape(nt, spin, norb, norb)

    assert np.allclose(x, I_all)


