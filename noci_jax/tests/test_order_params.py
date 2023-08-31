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
from noci_jax import slater, pyscf_helpers, analysis
from pyscf import gto, scf


def test_corr_spin():
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

    # t_vecs = np.random.rand(3, 2*nvir*nocc)-0.5
    # t_vecs[0] = 0
    t_vecs = np.load("./data/h4_tvec5.npy")

    rmats = slater.tvecs_to_rmats(t_vecs, nvir, nocc)
    hmat, smat = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    energy, c = slater.solve_lc_coeffs(hmat, smat, return_vec=True)

    c_spin_det = analysis.corr_spin_state(rmats, mo_coeff, c)
    dm1s, rdm2s = slater.make_rdm12(rmats, mo_coeff, c)

    c_spin_dms = analysis.corr_spin_dms(dm1s, rdm2s[1])

    assert np.allclose(c_spin_det, c_spin_dms)

