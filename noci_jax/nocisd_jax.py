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
Jax version of nocisd.py.
Not faster because of array shape changes.
'''
from jax import numpy as jnp
import copy 
from pyscf import ci
from noci_jax import slater_jax
import gc 
import logging

def compress(ci_amps, dt1=0.1, dt2=0.1, tol2=1e-5, silent=True):
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
    c0, c1, c2 = ci_amps
    coeff0 = c0
    # get the CIS thouless
    t1s = jnp.array(c2t_singles(c1, dt=dt1))
    coeff1 = jnp.array([1/dt1, -1/dt1]*2)

    # get the CID thouless for same spin
    t2s, lam2s = c2t_doubles(c2, dt=dt2, tol=tol2)
    t2s = jnp.vstack(t2s)
    
    coeff2 = jnp.concatenate([lam2s[0],]*2 + [lam2s[1],]*2 + [-lam2s[1],]*2 + [ lam2s[2],]*2)
    coeff2 /= (dt2**2)

    # CID also has the contribution of HF GS
    nvir, nocc = t1s.shape[2:]
    t0 = jnp.zeros((1, 2, nvir, nocc))
    coeff2_0 = jnp.concatenate([lam2s[0],]*2 + [lam2s[2],]*2)/(dt2**2)
    coeff0 -= 2*jnp.sum(coeff2_0)
    coeff0 = jnp.array([coeff0])

    t_all = jnp.vstack([t0, t1s, t2s])
    num_t = len(t_all) 
    print("# Compressed CISD to {} NOSDs.".format(num_t))
    coeff_all = jnp.concatenate([coeff0, coeff1, coeff2])
    coeff_all /= jnp.linalg.norm(coeff_all)
    return t_all, coeff_all

def gen_nocisd_multiref(tvecs_ref, mf, nvir=None, nocc=None, dt=0.1, tol2=1e-5, silent=False):
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
    import time 
    print("#"*10 + " Multi-Ref NOCISD " + "#"*10)
    num_ref = len(tvecs_ref)
    if nvir is None:
        nvir, nocc = tvecs_ref.shape[-2:]
    # generate orthonormal MOs for each determinant

    U_on_ref = slater_jax.orthonormal_mos(tvecs_ref)
    mo_coeff = mf.mo_coeff # HF MO coeffs
    mo_ref = jnp.einsum("sij, nsjk -> nsik", mo_coeff, U_on_ref)
    my_mf = copy.copy(mf)
    # generate the CISD compressed NOSDs
    # first do HF
    my_ci = ci.UCISD(mf)
    _, civec = my_ci.kernel()
    ci_amps = ucisd_amplitudes(my_ci, civec=civec, silent=silent)
    t, _ = compress(ci_amps, dt1=dt, dt2=dt, tol2=tol2, silent=silent)
    r = slater_jax.tvecs_to_rmats(t, nvir, nocc)
    r_cisd = r[1:] # only choose the singles and doubles 
    for i in range(num_ref-1):
        my_mf.mo_coeff = mo_ref[i+1]
        my_ci = ci.UCISD(my_mf)
        _, civec = my_ci.kernel()
        ci_amps = ucisd_amplitudes(my_ci, civec=civec, silent=silent)

        r = slater_jax.tvecs_to_rmats(t, nvir, nocc)
        r = slater_jax.rotate_rmats(r, U_on_ref[i+1])
        r_cisd = jnp.vstack([r_cisd, r[1:]])
        gc.collect()
    return r_cisd

def gen_nocisd_onevec(tvec, mf, nvir=None, nocc=None, dt=0.1, tol2=1e-5, silent=False):
    '''
    Given one Thouless vector, generate the compressed 
    non-orthogonal CISD from this SD.
    Args:
        tvec: (1, 2, nvir, nocc) or (2, nvir, nocc)array.
        mf: the converged PySCF scf object.
    Returns: 
        (M, 2, norb, nocc) array, the rotation matrices
        from the CISD expansion from this reference state.
    '''
    # print("#"*10 + " Multi-Ref NOCISD " + "#"*10)
    # num_ref = len(tvecs_ref)
    if nvir is None:
        nvir, nocc = tvec.shape[-2:]
    norb = nvir + nocc
    U = slater_jax.orthonormal_mos(tvec).reshape(2, norb, norb)
    mo_coeff = mf.mo_coeff # HF MO coeffs
    mo_ref = jnp.array([mo_coeff[0] @ U[0], mo_coeff[1] @ U[1]])
    my_mf = copy.copy(mf)
    my_mf.mo_coeff = mo_ref
    my_ci = ci.UCISD(my_mf)
    _, civec = my_ci.kernel()
    t, _ = compress(my_ci, civec=civec, dt1=dt, dt2=dt, tol2=tol2, silent=silent)
    r = slater_jax.tvecs_to_rmats(t, nvir, nocc)
    r = slater_jax.rotate_rmats(r, U)
    return r

def gen_nocisd_multiref_hsp(mf, nvir, nocc, dt=0.1, tol2=1e-5, silent=False):
    '''
    Generate nocisd from half-spin projection.
    Args:
        mf: the converged PySCF scf object.
        nvir: int, number of virtual orbitals.
        nocc: int, number of occupied orbitals.
    Returns: 
        (M, 2, norb, nocc) array, the rotation matrices
        from the CISD expansion from each reference. 
    '''

    mo_coeff = mf.mo_coeff # HF MO coeffs
    Ca = mo_coeff[0]
    Cb = mo_coeff[1]
    len_C = Ca.shape[0]

    my_mf = copy.copy(mf)
    # generate the CISD compressed NOSDs
    # first do HF 
    my_ci = ci.UCISD(mf)
    _, civec = my_ci.kernel()
    t, _ = compress(my_ci, civec=civec, dt1=dt, dt2=dt, tol2=tol2, silent=silent)
    r = slater_jax.tvecs_to_rmats(t, nvir, nocc)
    r_cisd = r[1:] # only choose the singles and doubles 
    r_ref = jnp.array([r[0]])
    gc.collect()
    # check if RHF 
    diff = Ca - Cb 
    if jnp.linalg.norm(diff)/len_C**2 < 1e-10:
        logging.warning("Two spins are degenerate.")
        return r_cisd
    else:
        U = jnp.linalg.inv(Ca) @ Cb
    
    # next do hsp
    my_mf.mo_coeff = jnp.array([Cb, Ca])

    my_ci = ci.UCISD(my_mf)
    _, civec = my_ci.kernel()
    t, _ = compress(my_ci, civec=civec, dt1=dt, dt2=dt, tol2=tol2, silent=silent)
    r = slater_jax.tvecs_to_rmats(t, nvir, nocc)
    r = slater_jax.rotate_rmats(r, jnp.array([U, U.conj().T]))
    r_cisd = jnp.vstack([r_cisd, r[1:]])
    r_ref = jnp.vstack([r_ref, r[0][None, :]])

    return r_cisd, r_ref


def gen_nocid_truncate(mf, nocc, nroots=4, dt=0.1):
    '''
    Return the doubly excited states that has largest contribution.
    '''
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)
    c2 = ucisd_amplitudes_doubles(my_ci)
    t2 = c2t_doubles_truncate(c2, num_roots=nroots, dt=dt, nvir=nvir, nocc=nocc)
    return t2


def ucisd_amplitudes(myci, civec=None, flatten_c2=False, silent=False):
    '''
    Return the CISD coefficients.
    Args:
        myci: PySCF CISD object.
    Kwargs:
        civec: the linear combination coefficients for orthogonal CISD expansion.
        flatten_c2: boolean, whether to turn the rank-4 tensor doubles coefficients into a matrix.
    Returns:
        c0: float, amplitude for HF ground state
        c1: 3D array, amplitudes for single excitations.
        c2: 3D array or 5D array, amplitudes for double excitations.
    '''
    # myci = ci.UCISD(mf)   
    if civec is None:                                                                 
        _, civec = myci.kernel()
    lci = len(civec)
    if not silent:
        print("# There are {} CISD dets.".format(lci))
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)

    # NOTE assumed alpha and beta same number of electrons
    nocc, nvir = c1[0].shape

    c1_n = jnp.transpose(jnp.array(c1), (0, 2, 1))
    # transpose c2
    c2_n = jnp.transpose(jnp.array(c2), (0, 3, 1, 4, 2)) 
    # count for the 4-fold degeneracy for same spin excitations.
    c2_n = c2_n.at[0].set( c2_n[0] / 4)
    c2_n = c2_n.at[2].set( c2_n[2] / 4)

    if flatten_c2:
        c2_n = c2_n.reshape(3, nvir*nocc, nvir*nocc)
    return c0, c1_n, c2_n


def ucisd_amplitudes_doubles(myci, civec=None):
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
    c2_n = jnp.transpose(jnp.array(c2), (0, 3, 1, 4, 2)) 
    # count for the 4-fold degeneracy for same spin excitations.
    c2_n = c2_n.at[0].set(c2_n[0]/4)
    c2_n = c2_n.at[2].set(c2_n[2]/4)
    return c2_n


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
    t0 = jnp.zeros_like(c1[0])
    t1p = c1 * dt / 2.
    t1m = -c1 * dt / 2.
    t1pa = jnp.array([t1p[0], t0])
    t1pb = jnp.array([t0, t1p[1]])
    t1ma = jnp.array([t1m[0], t0])
    t1mb = jnp.array([t0, t1m[1]])
    # coeffs = jnp.array([1./dt, -1./dt])
    return [t1pa, t1ma, t1pb, t1mb]
    

def c2t_doubles(c2, dt=0.1, nvir=None, nocc=None, tol=5e-4):
    '''
    Generate NOSDs corresponding to the doubly excited states.
    same spin - 4 fold degeneracy 
    Args:
        c2: array of dimension (3, nvir, nocc, nvir, nocc).
    Kwargs:
        dt: finite difference to approximate 2nd order derivatives.
        nvir: number of virtual orbitals.
        nocc: number of occupied orbitals.
        tol: threshold to discard eigenvalues.
    Returns:
        a list of Thouless matrices corresponding to aaaa, aabb, bbbb
        the corresponding coefficients
    '''
    if nvir is None:
        nvir = c2.shape[1]
        nocc = c2.shape[2]
    
    c2 = c2.reshape(3, nvir*nocc, nvir*nocc)
    e_aa, v_aa = jnp.linalg.eigh(c2[0])
    idx_aa = jnp.where(jnp.abs(e_aa) > tol)
    z_aa = v_aa[:, idx_aa].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_aa = jnp.zeros_like(z_aa)
    t_aa = jnp.transpose(jnp.array([z_aa, pad_aa]), (1,0,2,3))
    c_aa = e_aa[idx_aa]

    # aabb
    u_a, e_ab, v_bt = jnp.linalg.svd(c2[1])
    v_b = v_bt.conj().T
    idx_ab = jnp.where(jnp.abs(e_ab) > tol)
    z_a = u_a[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    z_b = v_b[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    t_ab_p = jnp.transpose(jnp.array([z_a, z_b]), (1,0,2,3))
    t_ab_m = jnp.transpose(jnp.array([z_a, -z_b]), (1,0,2,3))
    c_ab = e_ab[idx_ab]
 
    # bbbb
    e_bb, v_bb = jnp.linalg.eigh(c2[2])
    idx_bb = jnp.where(jnp.abs(e_bb) > tol)
    z_bb = v_bb[:, idx_bb].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
    pad_bb = jnp.zeros_like(z_bb)
    t_bb = jnp.transpose(jnp.array([pad_bb, z_bb]), (1,0,2,3))
    c_bb = e_bb[idx_bb]

    tmat_aa = jnp.vstack([t_aa, -t_aa])*dt
    tmat_ab = jnp.vstack([t_ab_p, -t_ab_p, t_ab_m, -t_ab_m])*dt/2.
    tmat_bb = jnp.vstack([t_bb, -t_bb])*dt

    return [tmat_aa, tmat_ab, tmat_bb], [c_aa, c_ab, c_bb]


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
    e_aa, v_aa = jnp.linalg.eigh(c2[0])
    u_a, e_ab, v_bt = jnp.linalg.svd(c2[1])
    v_b = v_bt.conj().T
    e_bb, v_bb = jnp.linalg.eigh(c2[2])
    n_aa, n_ab = len(e_aa), len(e_ab)

    e_all = jnp.abs(jnp.concatenate([e_aa, e_ab, e_bb]))

    idx_all = jnp.argsort(e_all)[::-1][:num_roots]
    tmats = []
    ns_aa, ns_ab, ns_bb = 0, 0, 0
    # tmats_aa, tmats_ab, tmats_bb = [], [], []
    for i in idx_all:
        if i < n_aa:
            ns_aa += 1
            idx_aa = i
            z_aa = v_aa[:, idx_aa].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            pad_aa = jnp.zeros_like(z_aa)
            t_a = jnp.transpose(jnp.array([z_aa, pad_aa]), (1,0,2,3))[0]
            tmats.append(t_a*dt)
            tmats.append(-t_a*dt)

        elif i < (n_aa + n_ab):
            ns_ab += 1
            idx_ab = i - n_aa
            z_a = u_a[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            z_b = v_b[:, idx_ab].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            t_ab_p = jnp.transpose(jnp.array([z_a, z_b]), (1,0,2,3))[0]
            t_ab_m = jnp.transpose(jnp.array([z_a, -z_b]), (1,0,2,3))[0]
            tmats.append(t_ab_p*dt/2.)
            tmats.append(-t_ab_p*dt/2.)
            tmats.append(t_ab_m*dt/2.)
            tmats.append(-t_ab_m*dt/2.)
        else:
            ns_bb += 1
            idx_bb = i - n_aa - n_ab
            z_bb = v_bb[:, idx_bb].reshape(nvir*nocc, -1).T.reshape(-1, nvir, nocc)
            pad_bb = jnp.zeros_like(z_bb)
            t_b = jnp.transpose(jnp.array([z_bb, pad_bb]), (1,0,2,3))[0]
            tmats.append(t_b*dt)
            tmats.append(-t_b*dt)
    print(f"Selected doubles with {ns_aa*2} aa, {ns_bb*2} bb, and {ns_ab*4} ab.")
    return jnp.asarray(tmats)




