# Copyright 2023 NOCI_Jax developers. All Rights Reserved.
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
'''
Interface to PySCF.
Only support unrestricted spin symmetry.
'''
import numpy as np
from scipy import linalg as sla
from pyscf import ao2mo

def get_integrals(mf, ortho_ao=False):
    '''
    Return essential values needed for NOCI calculations.
    '''
    h1e = mf.get_hcore()
    h2e = mf.mol.intor('int2e')
    e_nuc = mf.energy_nuc()
    

    if ortho_ao:
        norb = mf.mol.nao
        ao_ovlp = mf.mol.intor_symmetric ('int1e_ovlp')
        trans_m = sla.inv(sla.sqrtm(ao_ovlp))
        h1e = trans_m @ h1e @ trans_m # trans_m.T = trans_m 
        h2e = ao2mo.incore.full(h2e, trans_m)
        # update the values 
        mf.get_hcore = lambda *args: h1e     
        mf._eri = ao2mo.restore(8, h2e, norb)                             
        mf.get_ovlp = lambda *args: np.eye(norb)                        

    return h1e, h2e, e_nuc    


def get_mos(mf):

    norb = mf.mol.nao # number of orbitals
    occ = mf.get_occ()
    nocc = int(np.sum(occ[0])) # number of occupied orbitals for spin up
    nvir = norb - nocc
    mo_coeff = np.asarray(mf.mo_coeff)

    return norb, nocc, nvir, mo_coeff


def run_stab_scf(mf):
    '''
    Stabalized HF.
    '''
    mf.kernel()
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init) 