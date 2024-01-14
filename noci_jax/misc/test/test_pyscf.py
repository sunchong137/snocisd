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


import numpy as np
from noci_jax.misc import pyscf_helper
from pyscf import gto, scf, ci




def test_cisd_energy():

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
    mol.unit='angstrom'
    mol.basis = "sto6g"

    # mol.symmetry = True
    mol.build()
    mf = scf.UHF(mol)
    # Hartree-Fock
    mf.kernel(verbose=0, tol=1e-10)
    myci = ci.UCISD(mf)  
    eris = myci.ao2mo(mf.mo_coeff)
    e_corr, civec = myci.kernel()
    ci_n = myci.contract(civec, eris)
    e_diff = np.dot(civec.conj().T, ci_n)
    assert np.allclose(e_diff, e_corr)

def test_civec_size():
    lci = 199
    size, loc = pyscf_helper.sep_cisdvec(8, 4)
    assert np.allclose(np.sum(size), lci)
    assert np.allclose(loc[-1], lci)

def test_ortho():
    pass