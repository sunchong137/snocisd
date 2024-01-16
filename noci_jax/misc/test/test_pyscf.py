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
from noci_jax import hamiltonians
from pyscf import gto, scf, ci

mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
mol.unit='angstrom'
mol.basis = "sto6g"

# mol.symmetry = True
mol.build()

def test_cisd_energy():


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
    mf = scf.UHF(mol)
    # mf.kernel()
    # Step 3: Get attributes needed for the NOCI calculations
    h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=True)
    mf.kernel()
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
    e_hf = mf.energy_tot()
    nelec = mol.nelectron
    CC = mo_coeff[0].T @ mo_coeff[0]
    assert np.allclose(CC, np.eye(norb))

    dm1 = mf.make_rdm1(mo_coeff)
    assert np.allclose(np.sum(np.diag(dm1[0])), nelec//2)

def test_compare_ortho_ao():
    '''
    check if the OAO and AO give the same energy.
    '''
    # no orthogonalization
    mf = scf.UHF(mol)
    mf.kernel() 
    e1 = mf.energy_tot() 

    mf2 = pyscf_helper.mf_with_ortho_ao(mol)
    mf2.kernel()
    e2 = mf2.energy_tot() 

    mf3 = scf.UHF(mol)
    h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf3, ortho_ao=True)
    mf3.kernel() 
    e3 = mf3.energy_tot()

    assert np.allclose(e1, e2)
    assert np.allclose(e1, e3)



