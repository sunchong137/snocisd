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
from noci_jax import reshf, pyscf_helpers 
from pyscf import gto, scf


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

    e, v = reshf.solve_lc_coeffs(hmat, smat, return_vec=True)
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
    h1e, h2e, e_nuc = pyscf_helpers.get_integrals(mf, ortho_ao=True)
    mf.kernel(verbose=0, tol=1e-10)
    
    norb, nocc, nvir, mo_coeff = pyscf_helpers.get_mos(mf)

    t_vecs = np.random.rand(2, 2*nvir*nocc)-0.5
    t_vecs[0] = 0

    rmats = reshf.tvecs_to_rmats(t_vecs, nvir, nocc)
    hmat, smat = reshf.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    energy, c = reshf.solve_lc_coeffs(hmat, smat, return_vec=True)
    rmats = rmats.at[1].set(rmats[1]/np.sqrt(smat[0,0]))
    rdm1 = reshf.make_rdm1(rmats, mo_coeff, c)
    ne_a = np.sum(np.diag(rdm1[0]))
    ne_b = np.sum(np.diag(rdm1[1]))

    assert np.allclose(ne_a, 2)
    assert np.allclose(ne_b, 2)

