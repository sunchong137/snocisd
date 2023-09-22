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
Functions to better use PySCF.
Only support functions for unrestricted spin symmetry.

Includes:
1. mean-field helpers.
2. CISD helpers.
'''
import numpy as np
from scipy import linalg as sla
from pyscf import ao2mo, ci
import logging 

# mean-field helpers
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
    print("**********System information***********")
    print("Number of orbitals: {}".format(norb))
    print("Number of occupied orbitals: {}".format(nocc))
    print("Number of virtual orbitals: {}".format(nvir))
    return norb, nocc, nvir, mo_coeff


def run_stab_scf(mf):
    '''
    Stabalized HF.
    '''
    mf.kernel()
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init) 


def rotate_ham(mf):
    '''
    Rotate the Hamiltonian from AO to MO.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    eri = mf._eri
    mo_coeff = mf.mo_coeff
    # aaaa, aabb, bbbb
    Ca, Cb = mo_coeff[0], mo_coeff[1]
    aaaa = (Ca,)*4
    bbbb = (Cb,)*4
    aabb = (Ca, Ca, Cb, Cb)

    h1_mo = np.array([Ca.T @ h1e @ Ca, Cb.T @ h1e @ Cb])
    h2e_aaaa = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
    h2e_bbbb = ao2mo.incore.general(eri, bbbb, compact=False).reshape(norb, norb, norb, norb)
    h2e_aabb = ao2mo.incore.general(eri, aabb, compact=False).reshape(norb, norb, norb, norb)
    h2_mo = np.array([h2e_aaaa, h2e_aabb, h2e_bbbb])

    return h1_mo, h2_mo

# CISD helpers
def cisd_energy_from_vec(vec, mf):
    '''
    Given a CISD-vector-like vector, return the energy.
    NOTE: the energy can be higher than the HF energy.
    Args:
        vec: 1D array, coefficients of HF, S and D.
        mf: PySCF mean-field object.
    Returns:
        double, the energy corresponding to the vec.
    '''
    myci = ci.UCISD(mf)  
    e_hf = mf.e_tot
    eris = myci.ao2mo(mf.mo_coeff)
    ci_n = myci.contract(vec, eris)
    e_corr = np.dot(vec.conj().T, ci_n)
    e_cisd = e_hf + e_corr
    return e_cisd

def sep_cisdvec(norb, nelec):
    '''
    Give the length of the 0th, one- and two- body excitations, respectively.
    Args:
        norb: int, number of orbitals
        nelec: int or tuple, number of electrons
    Returns:
        list of [l0, l1_u, l1_d, l1_ab, l1_aa, l1_bb]
    '''
    try:
        na, nb = nelec 
    except:
        na = nelec//2
        nb = nelec - na

    noa = na 
    nob = nb
    nva = norb - na 
    nvb = norb - nb

    nooa = noa * (noa - 1) // 2
    nvva = nva * (nva - 1) // 2
    noob = nob * (nob - 1) // 2
    nvvb = nvb * (nvb - 1) // 2

    size = [1, noa*nva, nob*nvb, noa*nob*nva*nvb,
            nooa*nvva, noob*nvvb]
    loc = np.cumsum(size)
    
    return size, loc