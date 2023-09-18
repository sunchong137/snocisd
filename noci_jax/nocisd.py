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
Compress the linear combinations of CISD with NOCI.
'''
import numpy as np
from pyscf import ci

def get_cisd_coeffs_uhf(mf, flatten_c2=False):
    '''
    Return the CISD coefficients.
    Returns:
        c1: 3D array
        c2: 
    '''
    myci = ci.UCISD(mf)                                                                    
    _, civec = myci.kernel()
    _, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    nocc, nvir = c1[0].shape

    c1_n = np.transpose(np.array(c1), (0, 2, 1))
    # transpose c2
    c2_n = np.transpose(np.array(c2), (0, 3, 1, 4, 2))

    if flatten_c2:
        nocc, nvir = c1[0].shape
        c2_n = c2_n.reshape(3, nvir*nocc, nvir*nocc)

    return c1_n, c2_n

def c2t_singles(c1, dt=0.1):
    '''
    Given the CIS coefficients, generate Thouless rotation paramters.
    Only for UHF.
    Args:
        c1: 3D array, size (2, nocc, nvir) amplitudes for singly excited states.
        t: float, a small number for the NOSD expansion approximation
    Returns:
        arrays of Thouless for NOSDs
    '''
    t1_p = c1 * dt / 2.
    t1_m = -c1 * dt / 2.

    # coeffs = np.array([1./dt, -1./dt])

    return np.array([t1_p, t1_m])
    

def c2t_doubles(c2, dt=0.1, nvir=None, nocc=None, tol=5e-4):
    '''
    Generate NOSDs corresponding to the doubly excited states.
    Args:
        c2: array.
    '''
    if nvir is None:
        nvir = c2.shape[1]
        nocc = c2.shape[2]
    
    c2 = c2.reshape(3, nvir*nocc, nvir*nocc)
    # TODO make the following more efficient
    e_aa, v_aa = np.linalg.eigh(c2[0])
    idx_aa = np.where(np.abs(e_aa) > tol)
    z_aa = v_aa[:, idx_aa].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_aa = np.zeros_like(z_aa)
    t_aa = np.transpose(np.array([z_aa, pad_aa]), (1,0,2,3))
    c_aa = e_aa[idx_aa]

    e_ab, v_ab = np.linalg.eigh(c2[1])
    idx_ab = np.where(np.abs(e_ab) > tol)
    z_ab = v_ab[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    t_ab = np.transpose(np.array([z_ab, z_ab]), (1,0,2,3))
    c_ab = e_ab[idx_ab]
 
    e_bb, v_bb = np.linalg.eigh(c2[2])
    idx_bb = np.where(np.abs(e_bb) > tol)
    z_bb = v_bb[:, idx_bb].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_bb = np.zeros_like(z_bb)
    t_bb = np.transpose(np.array([pad_bb, z_bb]), (1,0,2,3))
    c_bb = e_bb[idx_bb]

    tmat_aa = np.vstack([t_aa*dt, -t_aa*dt])
    tmat_ab = np.vstack([t_ab*dt, -t_ab*dt])
    tmat_bb = np.vstack([t_bb*dt, -t_bb*dt])

    return [tmat_aa, tmat_ab, tmat_bb], [c_aa, c_ab, c_bb]


def compress():
    pass