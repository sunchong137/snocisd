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
Compress the linear combinations of RCISD with NOCI.
'''
import numpy as np
import copy
from pyscf import ci
from noci_jax import slater_spin0, select_ci


def gen_nocisd_multiref(tvecs_ref, mf, nvir=None, nocc=None, dt=0.1, tol2=1e-5):
    '''
    Given a set of non-orthogonal SDs, generate the compressed 
    non-orthogonal CISD expansion from each SD.
    Args:
        tvec_ref: (N, 2, nvir, nocc) array, the reference NOSDs. 
                  The first one is the HF state.
        mf: the converged PySCF scf object.
    Returns: 
        (M, 2, norb, nocc) array, the rotation matrices
        from the CISD expansion from each reference. 
        All the Thouless matrices are based on the HF state.
    '''
    num_ref = len(tvecs_ref)
    if nvir is None:
        nvir, nocc = tvecs_ref.shape[-2:]
    # generate orthonormal MOs for each determinant

    U_on_ref = slater_spin0.orthonormal_mos(tvecs_ref)
    mo_coeff = mf.mo_coeff # HF MO coeffs
    mo_ref = np.einsum("ij, njk -> nik", mo_coeff, U_on_ref)
    
    my_mf = copy.copy(mf)
    # generate the CISD compressed NOSDs
    # first do HF
    my_ci = ci.RCISD(mf)
    _, civec = my_ci.kernel()
    t, _ = compress(my_ci, civec=civec, dt1=dt, dt2=dt, tol2=tol2)
    r = slater_spin0.tvecs_to_rmats(t, nvir, nocc)
    r_cisd = r[1:] # only choose the singles and doubles 

    for i in range(num_ref-1):
        my_mf.mo_coeff = mo_ref[i+1]
        my_ci = ci.RCISD(my_mf)
        _, civec = my_ci.kernel()
        t, _ = compress(my_ci, civec=civec, dt1=dt, dt2=dt, tol2=tol2)
        r = slater_spin0.tvecs_to_rmats(t, nvir, nocc)
        r = slater_spin0.rotate_rmats(r, U_on_ref[i+1])
        r_cisd = np.vstack([r_cisd, r[1:]])

    return r_cisd


def compress(myci, civec=None, dt1=0.1, dt2=0.1, tol2=1e-5):
    '''
    Approximate an orthogonal CISD expansion with the compressed non-orthogonal
    expansion. 

    Args:
        myci: PySCF CISD object.
    Kwargs:
        civec: the coefficients of the CISD expansion
        dt1: difference for the two-point derivative for the singles expansion.
        dt2: difference for the two-point derivative for the doubles expansion.
        tol2: tolerance to truncate the doubles expansion.
    Returns:
        t_all: (N, 2, nvir, nocc) array, where N is the number of NO determinants (including the ground state).
        coeff_all: the corresponding coefficients to recover the CISD wavefunction with NOSDs.
    '''
    c0, c1, c2 = cisd_amplitudes(myci, civec=civec)
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
    num_t = len(t_all) 
    print("Compressed CISD to {} NOSDs.".format(num_t))
    coeff_all = np.concatenate([coeff0, coeff1, coeff2])
    coeff_all /= np.linalg.norm(coeff_all)

    return t_all, coeff_all


def cisd_amplitudes(myci, civec=None):
    '''
    Return the CISD coefficients.
    Args:
        myci: PySCF CISD object.
    Kwargs:
        civec: the linear combination coefficients for orthogonal CISD expansion.
        flatten_c2: boolean, whether to turn the rank-4 tensor doubles coefficients into a matrix.
    Returns:
        c0: float, amplitude for HF ground state
        c1: 2D array, amplitudes for single excitations.
        c2: 2D array or 4D array, amplitudes for double excitations.
    '''

    if civec is None:                                                                 
        _, civec = myci.kernel()
    lci = len(civec)
    print("There are {} CISD dets.".format(lci))
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)

    c1 = c1.T
    # transpose c2
    c2ab = c2
    c2aa = c2 - c2.transpose(1, 0, 2, 3)
    c2aa /= 4. # count for the 4-fold degeneracy for same spin excitations.
    c2aa = c2aa.transpose(2, 0, 3, 1)
    c2ab = c2ab.transpose(2, 0, 3, 1)

    return c0, c1, [c2aa, c2ab]


def c2t_singles(c1, dt=0.1):
    # TODO only return t1p and t1m
    t0 = np.zeros_like(c1)
    t1p = c1 * dt / 2.
    t1m = -c1 * dt / 2.
    t1pa = np.array([t1p, t0])
    t1pb = np.array([t0, t1p])
    t1ma = np.array([t1m, t0])
    t1mb = np.array([t0, t1m])
    return [t1pa, t1ma, t1pb, t1mb]
    

def c2t_doubles(c2, dt=0.1, nvir=None, nocc=None, tol=5e-4):
    # TODO make it more efficient

    if nvir is None:
        nvir, nocc = c2[0].shape[:2]
    
    c2 = c2.reshape(3, nvir*nocc, nvir*nocc)
    # aaaa
    e_aa, v_aa = np.linalg.eigh(c2[0])
    idx_aa = np.where(np.abs(e_aa) > tol)
    z_aa = v_aa[:, idx_aa].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_aa = np.zeros_like(z_aa)
    t_aa = np.transpose(np.array([z_aa, pad_aa]), (1,0,2,3))
    c_aa = e_aa[idx_aa]
    # aabb
    u_a, e_ab, v_bt = np.linalg.svd(c2[1])
    v_b = v_bt.conj().T
    idx_ab = np.where(np.abs(e_ab) > tol)
    z_a = u_a[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    z_b = v_b[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    t_ab_p = np.transpose(np.array([z_a, z_b]), (1,0,2,3))
    t_ab_m = np.transpose(np.array([z_a, -z_b]), (1,0,2,3))
    c_ab = e_ab[idx_ab]
 
    tmat_aa = np.vstack([t_aa, -t_aa])*dt
    tmat_ab = np.vstack([t_ab_p, -t_ab_p, t_ab_m, -t_ab_m])*dt/2.

    return [tmat_aa, tmat_ab, tmat_aa], [c_aa, c_ab, c_aa]



#
# The following is not adapted to RCISD
#

def gen_nocid_two_layers(mf, nocc, nroots1=4, nroots2=2, dt=0.1):
    '''
    Generate NOCI as following:
    |HF> -> CISD -> choose |mu> with largest coeff -> CISD on |mu> -> choose ...
    '''
    # Generate the first layer
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.RCISD(mf)

    # first layer
    c2 = cisd_amplitudes_doubles(my_ci)
    t2 = c2t_doubles_truncate(c2, num_roots=nroots1, dt=dt, nvir=nvir, nocc=nocc)
    num_layer1 = len(t2)
    t_hf = np.zeros((1, nvir, nocc))
    r_all = slater_spin0.tvecs_to_rmats(np.vstack([t_hf, t2]), nvir, nocc) # store all the rmats
    
    U2 = slater_spin0.orthonormal_mos(t2)
    mo_2 = np.einsum("sij, nsjk -> nsik", mo_coeff, U2)

    my_mf = copy.copy(mf)
    # do cisd on the first layer
    for i in range(num_layer1):
        my_mf.mo_coeff = mo_2[i]
        my_ci = ci.UCISD(my_mf)
        c2_n = cisd_amplitudes_doubles(my_ci)
        t2_n = c2t_doubles_truncate(c2_n, num_roots=nroots2, dt=dt, nvir=nvir, nocc=nocc)
        r = slater_spin0.tvecs_to_rmats(t2_n, nvir, nocc)
        r = slater_spin0.rotate_rmats(r, U2[i])
        r_all = np.vstack([r_all, r])

    return r_all


def gen_two_layers_w_selection(mf, nocc, nroots1=4, nroots2=2, dt=0.1, m_tol=1e-6):
    '''
    Generate NOCI as following:
    |HF> -> CISD -> choose |mu> with largest coeff -> CISD on |mu> -> choose ...
    '''
    # Generate the first layer
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)
    t_hf = np.zeros((1, 2, nvir, nocc))
    r_all = slater_spin0.tvecs_to_rmats(t_hf, nvir, nocc)

    # first layer
    c2 = cisd_amplitudes_doubles(my_ci)
    t2 = c2t_doubles_truncate(c2, num_roots=nroots1, dt=dt, nvir=nvir, nocc=nocc)
    r2 = slater_spin0.tvecs_to_rmats(t2, nvir, nocc)
    r_all, idx1 = select_ci.select_rmats_ovlp(r_all, r2, m_tol=m_tol, return_indices=True)
    t2 = t2[idx1]
    num_layer1 = len(t2)    
    
    U2 = slater_spin0.orthonormal_mos(t2)
    mo_2 = np.einsum("sij, nsjk -> nsik", mo_coeff, U2)

    my_mf = copy.copy(mf)
    # do cisd on the first layer
    for i in range(num_layer1):
        my_mf.mo_coeff = mo_2[i]
        my_ci = ci.UCISD(my_mf)
        c2_n = cisd_amplitudes_doubles(my_ci)
        t2_n = c2t_doubles_truncate(c2_n, num_roots=nroots2, dt=dt, nvir=nvir, nocc=nocc)
        r2_n = slater_spin0.tvecs_to_rmats(t2_n, nvir, nocc)
        r2_n = slater_spin0.rotate_rmats(r2_n, U2[i])
        r_all, idx2 = select_ci.select_rmats_ovlp(r_all, r2_n, m_tol=m_tol, return_indices=True)

    return r_all


def gen_nocid_truncate(mf, nocc, nroots=4, dt=0.1):
    '''
    Return the doubly excited states that has largest contribution.
    '''
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)
    c2 = cisd_amplitudes_doubles(my_ci)
    t2 = c2t_doubles_truncate(c2, num_roots=nroots, dt=dt, nvir=nvir, nocc=nocc)
    return t2


def cisd_amplitudes_doubles(myci, civec=None):
    '''
    Return the CISD coefficients for doubles
    Args:
        myci: PySCF CISD object.
    Kwargs:
        civec: the linear combination coefficients for orthogonal CISD expansion.
    Returns:
        c2: 3D array or 5D array, amplitudes for double excitations.
    '''
    # myci = ci.UCISD(mf)   
    if civec is None:                                                                 
        _, civec = myci.kernel()
    
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)

    # transpose c2
    c2_n = np.transpose(np.array(c2), (0, 3, 1, 4, 2)) 
    # count for the 4-fold degeneracy for same spin excitations.
    c2_n[0] /= 4.
    c2_n[2] /= 4.

    return c2_n


def c2t_doubles_truncate(c2, num_roots=4, dt=0.1, nvir=None, nocc=None):
    '''
    Generate NOSDs corresponding to the doubly excited states.
    Pick num_dets determinants with largest impact.
    same spin - 4 fold degeneracy 
    Args:
        c2: array of dimension (3, nvir, nocc, nvir, nocc).
    Kwargs:
        num_roots: number of eigenvalues to choose.
        dt: finite difference to approximate 2nd order derivatives.
        nvir: number of virtual orbitals.
        nocc: number of occupied orbitals.
    Returns:
        a list of Thouless matrices corresponding to aaaa, aabb, bbbb
        the corresponding coefficients
    '''
    # TODO very inefficient
    if nvir is None:
        nvir = c2.shape[1]
        nocc = c2.shape[2]
    
    c2 = c2.reshape(3, nvir*nocc, nvir*nocc)
    e_aa, v_aa = np.linalg.eigh(c2[0])
    u_a, e_ab, v_bt = np.linalg.svd(c2[1])
    v_b = v_bt.conj().T
    e_bb, v_bb = np.linalg.eigh(c2[2])
    n_aa, n_ab = len(e_aa), len(e_ab)

    e_all = np.abs(np.concatenate([e_aa, e_ab, e_bb]))

    idx_all = np.argsort(e_all)[::-1][:num_roots]
    tmats = []
    ns_aa, ns_ab, ns_bb = 0, 0, 0
    # tmats_aa, tmats_ab, tmats_bb = [], [], []
    for i in idx_all:
        if i < n_aa:
            ns_aa += 1
            idx_aa = i
            z_aa = v_aa[:, idx_aa].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            pad_aa = np.zeros_like(z_aa)
            t_a = np.transpose(np.array([z_aa, pad_aa]), (1,0,2,3))[0]
            tmats.append(t_a*dt)
            tmats.append(-t_a*dt)

        elif i < (n_aa + n_ab):
            ns_ab += 1
            idx_ab = i - n_aa
            z_a = u_a[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            z_b = v_b[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            t_ab_p = np.transpose(np.array([z_a, z_b]), (1,0,2,3))[0]
            t_ab_m = np.transpose(np.array([z_a, -z_b]), (1,0,2,3))[0]
            tmats.append(t_ab_p*dt/2.)
            tmats.append(-t_ab_p*dt/2.)
            tmats.append(t_ab_m*dt/2.)
            tmats.append(-t_ab_m*dt/2.)
        else:
            ns_bb += 1
            idx_bb = i - n_aa - n_ab
            z_bb = v_bb[:, idx_bb].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            pad_bb = np.zeros_like(z_bb)
            t_b = np.transpose(np.array([z_bb, pad_bb]), (1,0,2,3))[0]
            tmats.append(t_b*dt)
            tmats.append(-t_b*dt)
    print(f"Selected doubles with {ns_aa*2} aa, {ns_bb*2} bb, and {ns_ab*4} ab.")
    return np.asarray(tmats)




