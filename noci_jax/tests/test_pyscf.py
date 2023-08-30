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
from noci_jax import pyscf_helpers
from pyscf import gto, scf, ao2mo

def test_get_integrals():

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
    e_hf = mf.e_tot
    
    rdm = mf.make_rdm1()
    ne_a = np.sum(np.diag(rdm[0]))
    ne_b = np.sum(np.diag(rdm[1]))

    assert abs(e_ref - e_hf) < 1e-10
    assert np.allclose(ne_a, 2)
    assert np.allclose(ne_b, 2)

    

