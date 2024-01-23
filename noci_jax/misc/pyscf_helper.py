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
from pyscf import ao2mo, ci, scf, lo
from noci_jax.misc import basis_transform

# mean-field helpers
def mf_with_ortho_ao(mol, spin_symm=False, method="lowdin"):
    '''
    Construct an SCF object with orthogonal AO basis.
    '''
    if spin_symm:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    norb = mf.mol.nao
    trans_m = basis_transform.get_C_ortho(mf.mol, method='lowdin')
    h1e = mf.get_hcore()
    h2e = mf.mol.intor('int2e')
    h1e = trans_m @ h1e @ trans_m # trans_m.T = trans_m 
    h2e = ao2mo.incore.full(h2e, trans_m)
    # update the values 
    mf.get_hcore = lambda *args: h1e     
    mf._eri = ao2mo.restore(8, h2e, norb)                             
    mf.get_ovlp = lambda *args: np.eye(norb)   
    return mf

def get_integrals_lo(mf, ortho_ao=False):
    '''
    Return essential values needed for NOCI calculations.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    if mf._eri is None:
        mf._eri = mf.mol.intor('int2e')
    h2e =  ao2mo.restore(1, mf._eri, norb)
    e_nuc = mf.energy_nuc()

    if ortho_ao:
        print("INFO: the AOs are being orthogonalized by the Lowdin method.")
        norb = mf.mol.nao
        C = lo.orth_ao(mf.mol, 'meta_lowdin') 
        ao_ovlp = mf.mol.intor_symmetric('int1e_ovlp')
        trans_m = C.T @ ao_ovlp
        h1e = trans_m @ h1e @ trans_m # trans_m.T = trans_m 
        h2e = ao2mo.incore.full(h2e, trans_m)
        # update the values 
        mf.get_hcore = lambda *args: h1e     
        mf._eri = ao2mo.restore(8, h2e, norb)                           
        mf.get_ovlp = lambda *args: np.eye(norb)                      
    return h1e, h2e, e_nuc

def get_integrals(mf, ortho_ao=False):
    '''
    Return essential values needed for NOCI calculations.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    if mf._eri is None:
        mf._eri = mf.mol.intor('int2e')
    h2e =  ao2mo.restore(1, mf._eri, norb)
    e_nuc = mf.energy_nuc()

    if ortho_ao:
        print("INFO: the AOs are being orthogonalized by diagonalizing AO ovlp!")
        norb = mf.mol.nao
        trans_m = basis_transform.get_C_ortho(mf.mol, method='lowdin')
        h1e = trans_m @ h1e @ trans_m # trans_m.T = trans_m 
        h2e = ao2mo.incore.full(h2e, trans_m)
        # update the values 
        mf.get_hcore = lambda *args: h1e     
        mf._eri = ao2mo.restore(8, h2e, norb)                             
        mf.get_ovlp = lambda *args: np.eye(norb)                      
    return h1e, h2e, e_nuc


def get_mos(mf):
    '''
    This is wrong because the ortho_ao is not considered.
    '''
    norb = mf.mol.nao # number of orbitals
    occ = mf.get_occ()
    ndim = occ.ndim
    if ndim > 1:
        nocc = int(np.sum(occ[0])) # number of occupied orbitals for spin up
    else:
        nocc = int(np.sum(occ)/2 + 1e-5)
    nvir = norb - nocc
    mo_coeff = np.asarray(mf.mo_coeff)
    try:
        na, nb = mf.mol.nelectron
        nelec = na + nb 
    except:
        nelec = mf.mol.nelectron
    spin = mf.mol.spin 
    print("########## System Information ##########")
    print("# Number of orbitals: {}".format(norb))
    print("# Number of occupied orbitals: {}".format(nocc))
    print("# Number of virtual orbitals: {}".format(nvir))
    print("# Number of electrons: {}; |Na - Nb| = {}".format(nelec, spin))
    print("#--------------------------------------#")
    # print("#"*40)
    return norb, nocc, nvir, mo_coeff


def run_stab_scf(mf, tol=1E-14, max_iter=100, chkfname=None):
    '''
    Stabalized HF.
    '''
    mf.kernel()
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ) 
    mf.max_cycle = max_iter        
    mf.conv_tol = tol      
    if chkfname is not None: # save mf information
        mf.chkfile = chkfname                                   
    mf.kernel(dm0=init) 

def run_stab_scf_breaksymm(mf, tol=1E-14, max_iter=100, chkfname=None):
    init_guess = mf.get_init_guess()
    init_guess[0][0, 0] = 1
    init_guess[1][0, 0] = 0
    mf.kernel(init_guess)
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)
    mf.max_cycle = max_iter    
    mf.conv_tol = tol     
    if chkfname is not None: # save mf information
        mf.chkfile = chkfname                                     
    mf.kernel(dm0=init) 

def restart_scf_from_check(mf, chkname, tol=1E-14, max_iter=100, save_chk=None, stab=False):
    '''
    Restart SCF from a given chkfile.
    '''
    mf.conv_tol = tol 
    mf.max_cycle = max_iter
    if save_chk is not None:
        mf.chkfile = save_chk
    mf.init_guess = mf.from_chk(chkname) 
    print("# Restarting SCF from {}".format(chkname))
    mf.kernel()
    if stab:
        # stablize
        mo1 = mf.stability()[0]                                                             
        init = mf.make_rdm1(mo1, mf.mo_occ)
        mf.kernel(dm0=init) 

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


def spin_correlation(dm1_diag, dm2_diag, i, j):
    '''
    Spin correlation between ith and jth AO.
    <n_iu n_jd> - <n_iu><n_jd>
    Args:
        dm1_diag: 2D array of size (2, norb), diagonal term of dm1s
        dm2_diag: 2D array of size (norb, norb), <up+ up dn+ dn>
        i: int, ith AO
        j: int, jth AO
    Returns:
        float
    '''
    return dm2_diag[i, j] - dm1_diag[0][i]*dm1_diag[1][j]