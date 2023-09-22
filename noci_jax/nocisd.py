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
NOTE: assumed nocca = noccb.
'''
import numpy as np
from pyscf import ci

def ucisd_amplitudes(mf, civec=None, flatten_c2=False):
    '''
    Return the CISD coefficients.
    Returns:
        c0: float, amplitude for HF ground state
        c1: 3D array, amplitudes for single excitations.
        c2: 3D array or 5D array, amplitudes for double excitations.
    '''
    myci = ci.UCISD(mf)   
    if civec is None:                                                                 
        _, civec = myci.kernel()
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)

    # NOTE assumed alpha and beta same number of electrons
    nocc, nvir = c1[0].shape

    c1_n = np.transpose(np.array(c1), (0, 2, 1))
    # transpose c2
    c2_n = np.transpose(np.array(c2), (0, 3, 1, 4, 2)) 
    # count for the 4-fold degeneracy for same spin excitations.
    c2_n[0] /= 4.
    c2_n[2] /= 4.

    if flatten_c2:
        c2_n = c2_n.reshape(3, nvir*nocc, nvir*nocc)

    return c0, c1_n, c2_n


def c2t_singles(c1, dt=0.1):
    '''
    Given the CIS coefficients, generate Thouless rotation paramters.
    Only for UHF.
    Args:
        c1: 3D array, size (2, nocc, nvir) amplitudes for singly excited states.
        t: float, a small number for the NOSD expansion approximation
    Returns:
        A list of two Thouless matrices (size (2, nvir, nocc))
    '''
    t0 = np.zeros_like(c1[0])
    t1p = c1 * dt / 2.
    t1m = -c1 * dt / 2.
    t1pa = np.array([t1p[0], t0])
    t1pb = np.array([t0, t1p[1]])
    t1ma = np.array([t1m[0], t0])
    t1mb = np.array([t0, t1m[1]])
    # coeffs = np.array([1./dt, -1./dt])
    return [t1pa, t1ma, t1pb, t1mb]
    

def c2t_doubles(c2, dt=0.1, nvir=None, nocc=None, tol=5e-4):
    '''
    Generate NOSDs corresponding to the doubly excited states.
    same spin - 4 fold degeneracy 
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

    u_a, e_ab, v_bt = np.linalg.svd(c2[1])
    v_b = v_bt.conj().T
    idx_ab = np.where(np.abs(e_ab) > tol)
    z_a = u_a[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    z_b = v_b[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    t_ab_p = np.transpose(np.array([z_a, z_b]), (1,0,2,3))
 
    t_ab_m = np.transpose(np.array([z_a, -z_b]), (1,0,2,3))
    c_ab = e_ab[idx_ab]
 
    e_bb, v_bb = np.linalg.eigh(c2[2])
    idx_bb = np.where(np.abs(e_bb) > tol)
    z_bb = v_bb[:, idx_bb].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_bb = np.zeros_like(z_bb)
    t_bb = np.transpose(np.array([pad_bb, z_bb]), (1,0,2,3))
    c_bb = e_bb[idx_bb]

    tmat_aa = np.vstack([t_aa, -t_aa])*dt
    tmat_ab = np.vstack([t_ab_p, -t_ab_p, t_ab_m, -t_ab_m])*dt
    tmat_bb = np.vstack([t_bb, -t_bb])*dt

    return [tmat_aa, tmat_ab, tmat_bb], [c_aa, c_ab, c_bb]


def compress(mf, civec=None, dt1=0.1, dt2=0.1, tol2=1e-5):
    '''
    Return NOSDs and corresponding coefficients.
    TODO: rewrite the up-down part.
    '''
    c0, c1, c2 = ucisd_amplitudes(mf, civec=civec)
    coeff0 = c0
    # get the CIS thouless
    t1s = np.array(c2t_singles(c1, dt=dt1))
    coeff1 = np.array([1/dt1, -1/dt1]*2)

    # get the CID thouless for same spin
    t2s, lam2s = c2t_doubles(c2, dt=dt2, tol=tol2)
    t2s = np.vstack(t2s)
    
    coeff2 = np.concatenate([lam2s[0],]*2 + [lam2s[1],]*2 + [-lam2s[1],]*2 + [ lam2s[2],]*2)
    coeff2 /= (dt2**2)

    # CID also has the contribution of HF GS
    nvir, nocc = t1s.shape[2:]
    t0 = np.zeros((1, 2, nvir, nocc))
    coeff2_0 = np.concatenate([lam2s[0],]*2 + [lam2s[2],]*2)/(dt2**2)
    coeff0 -= 2*np.sum(coeff2_0)
    coeff0 = np.array([coeff0])

    t_all = np.vstack([t0, t1s, t2s])
    coeff_all = np.concatenate([coeff0, coeff1, coeff2])
    coeff_all /= np.linalg.norm(coeff_all)

    return t_all, coeff_all