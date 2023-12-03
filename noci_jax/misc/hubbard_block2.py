# Copyright 2023 HubBench developers. All Rights Reserved.
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
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
# from pyblock2.algebra.io import MPSTools, MPOTools


def hubbard1d_dmrg(nsite, U, nelec=None, pbc=False, filling=1.0, init_bdim=50, 
                   max_bdim=200, nsweeps=8, cutoff=1e-8, max_noise=1e-5):
    # set system
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec/nsite - filling) > 1e-5:
            print("WARNING: The filling is changed to {:1.2f}".format(nelec/nsite))
        if nelec % 2 == 0:
            spin = 0
        else:
            spin = 1
    else:
        try:
            neleca, nelecb = nelec
            spin = abs(neleca - nelecb)
            nelec = neleca + nelecb 
        except:
            spin = 0  
    
    
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec, spin=spin)

    # build Hamiltonian
    # c - creation spin up, d - annihilation spin up
    # C - creation spin dn, D - annihilation spin dn
    ham_str = driver.expr_builder() 
    ham_str.add_term("cd", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -1)
    ham_str.add_term("CD", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -1)
    if pbc:
        ham_str.add_term("cd", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), -1)
        ham_str.add_term("CD", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), -1)

    ham_str.add_term("cdCD", np.array([[i, ] * 4 for i in range(nsite)]).flatten(), U)
    ham_mpo = driver.get_mpo(ham_str.finalize(), iprint=0)

    # Schedule, using the linearly growing bond dimonsion
    bdims = list(np.linspace(init_bdim, max_bdim, nsweeps//2, endpoint=True, dtype=int))
    if max_noise < 1e-16:
        noises = [0.0]
    else:
        noises = list(np.logspace(np.log(max_noise), -16, nsweeps//2, endpoint=True)) + [0.0]

    ket = driver.get_random_mps(tag="KET", bond_dim=init_bdim, nroots=1)
    thrds = [1e-10] * nsweeps
    energy = driver.dmrg(ham_mpo, ket, n_sweeps=nsweeps, bond_dims=bdims, noises=noises,
             thrds=thrds, cutoff=cutoff, iprint=0)

    print('DMRG total energy = {:2.6f}, energy per site = {:2.6f}'.format(energy, energy/nsite))
    return energy, ket, ham_mpo