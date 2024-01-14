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
from noci_jax.misc import solvers
from pyscf import gto, scf, fci 

def test_dmrg():
    atom = ["H 0 0 0", "H 0 0 1.1"]
    # mol = gto.M(atom="H 0 0 0; H 0 0 1.1", basis="sto3g", verbose=0)
    mol = gto.M(atom=atom, basis="sto3g", verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1E-14)
    myfci = fci.FCI(mf)
    e_fci, v = myfci.kernel()
    dm1_ci, dm2_ci = myfci.make_rdm12s(v, 2, 2)
    e_dmrg, _, dm1, dm2 = solvers.run_block2(mf, init_bdim=100, max_bdim=200, 
                                                      spin_symm=False, return_pdms=True)
    assert abs(e_dmrg-e_fci) < 1e-10
    assert np.allclose(np.asarray(dm1_ci), np.asarray(dm1))
    assert np.allclose(np.asarray(dm2_ci), np.asarray(dm2))


def test_sci_solver():

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
    mol.unit='angstrom'
    mol.basis = "sto6g"

    # mol.symmetry = True
    mol.build()
    mf = scf.UHF(mol)
    # Hartree-Fock
    mf.kernel(verbose=0, tol=1e-10)
    fcisolver = fci.FCI(mf)
    e, c = fcisolver.kernel()
    e1, c1 = solvers.run_shci(mol)
    assert np.allclose(e, e1)