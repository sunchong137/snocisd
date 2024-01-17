# Copyright 2023 NOCI_JAX developers. All Rights Reserved.
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
from scipy import linalg as sla
from pyscf import ao2mo


def get_C_ortho(mol, method='lowdin'):
    '''
    Evaluate the transformation matrix.
    Args:
        mol: PySCF gto.Mole() object.
    Kwargs:
        method: string, lowdin or schmidt.
    Returns:
        2D array, transformation matrix from AO to OAO (orthogonal AO)
    '''
    ao_ovlp = mol.intor_symmetric('int1e_ovlp')
    if method == 'lowdin':
        return C_ortho_lowdin(ao_ovlp) 
    elif method == 'schmidt':
        return C_ortho_schmidt(ao_ovlp)
    else:
        print("Warning: method can only be 'lowdin' or 'schmidt'! Using Lowdin method.")
        return C_ortho_lowdin(ao_ovlp) 

def C_ortho_lowdin(ao_ovlp):
    '''
    Evaluate the transform matrix via directly multiply S^{-1/2}.

    '''
    e, v = np.linalg.eigh(ao_ovlp)
    idx = e > 1e-15 # avoid singularity
    return np.dot(v[:, idx]/np.sqrt(e[idx]), v[:, idx].conj().T)

def C_ortho_schmidt(ao_ovlp):
    '''
    Schmidt orthogonalization.
    '''
    c = np.linalg.cholesky(ao_ovlp)
    return sla.solve_triangular(c, np.eye(c.shape[1]), lower=True, overwrite_b=False).conj().T


def basis_trans_mat(mat, C):
    '''
    Transform a matrix from one basis to another.
    The basis are transformed in the way:
         |new>_i = \sum_{ij} |old>_j C_{ji}
    Args:
        mat: 2D array.
        C: 2D array or list of 2D array.
    '''
    try:
        ndim_c = C.ndim
    except:
        ndim_c = 3 

    if ndim_c == 3: # spin-up and spin-down separately 
        len_c = len(C)
        assert len_c == 2 
        return np.array([C[0].conj().T @ mat @ C[0], C[1].conj().T @ mat @ C[1]])
    else:
        return C.conj().T @ mat @ C


def basis_trans_tensor(tnsr, C):
    '''
    Transform a tensor from one basis to another.
    The basis are transformed in the way:
         |new>_i = \sum_{ij} |old>_j C_{ji}
    Args:
        tnsr: 4D array or a list of 3 4D arrays. 
        C: 2D array or a list of 2 2D arrays.
    Returns:
        Rotated tensor.
    '''
    try:
        ndim_c = C.ndim
    except:
        ndim_c = 3 
    try:
        ndim_t = tnsr.ndim 
    except:
        ndim_t = 5

    if ndim_c == 2: # one rotation matrix.
        if ndim_t == 4: 
            nao = tnsr.shape[-1]
            return ao2mo.incore.general(tnsr, (C,)*4, compact=False).reshape(nao, nao, nao, nao)
        else:
            assert len(tnsr) == 3
            t_aa = ao2mo.incore.general(tnsr[0], (C,)*4, compact=False).reshape(nao, nao, nao, nao)
            t_bb = ao2mo.incore.general(tnsr[2], (C,)*4, compact=False).reshape(nao, nao, nao, nao)
            t_ab = ao2mo.incore.general(tnsr[1], (C,)*4, compact=False).reshape(nao, nao, nao, nao)
            return np.array([t_aa, t_ab, t_bb])
    else: # two spins
        aa = (C[0],)*4
        bb = (C[1],)*4
        ab = (C[0], C[0], C[1], C[1])
        if ndim_t == 4: 
            nao = tnsr.shape[-1]
            t_aa = ao2mo.incore.general(tnsr, aa, compact=False).reshape(nao, nao, nao, nao)
            t_bb = ao2mo.incore.general(tnsr, bb, compact=False).reshape(nao, nao, nao, nao)
            t_ab = ao2mo.incore.general(tnsr, ab, compact=False).reshape(nao, nao, nao, nao)
            return np.array([t_aa, t_ab, t_bb])
        else:
            assert len(tnsr) == 3
            nao = tnsr[0].shape[-1]
            t_aa = ao2mo.incore.general(tnsr[0], aa, compact=False).reshape(nao, nao, nao, nao)
            t_bb = ao2mo.incore.general(tnsr[2], bb, compact=False).reshape(nao, nao, nao, nao)
            t_ab = ao2mo.incore.general(tnsr[1], ab, compact=False).reshape(nao, nao, nao, nao)
            return np.array([t_aa, t_ab, t_bb])


# def ao2mo_ham(mf):
#     '''
#     Rotate the Hamiltonian from AO to MO.
#     '''
#     h1e = mf.get_hcore()
#     norb = mf.mol.nao
#     eri = mf._eri
#     mo_coeff = mf.mo_coeff
#     # aaaa, aabb, bbbb
#     Ca, Cb = mo_coeff[0], mo_coeff[1]
#     aaaa = (Ca,)*4
#     bbbb = (Cb,)*4
#     aabb = (Ca, Ca, Cb, Cb)

#     h1_mo = np.array([Ca.T @ h1e @ Ca, Cb.T @ h1e @ Cb])
#     h2e_aaaa = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
#     h2e_bbbb = ao2mo.incore.general(eri, bbbb, compact=False).reshape(norb, norb, norb, norb)
#     h2e_aabb = ao2mo.incore.general(eri, aabb, compact=False).reshape(norb, norb, norb, norb)
#     h2_mo = np.array([h2e_aaaa, h2e_aabb, h2e_bbbb])
#     return h1_mo, h2_mo

# def ao2mo_ham_spin0(mf):
#     '''
#     Rotate the Hamiltonian from AO to MO.
#     '''
#     h1e = mf.get_hcore()
#     norb = mf.mol.nao
#     eri = mf._eri
#     C = mf.mo_coeff
#     # aaaa, aabb, bbbb
#     aaaa = (C,)*4
#     h1_mo = C.T @ h1e @ C
#     h2_mo = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
#     return h1_mo, h2_mo