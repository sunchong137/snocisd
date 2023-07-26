# Functions to valuate values related to Slater determinants
# Many functions are provided with two approaches: 
#   1. from Slater determinants
#   2. from rotation matrices
# Slater determinants must contain two spins, 
# while the rotation matrices can just be for one spin.

import numpy as np
import time
from numpy import einsum 
# from pyscf.lib import einsum


def rotation_spinless(mo_coeff, rotmat):
    '''
    Thouless rotation of a single spin.
    '''
    mo_coeff_new = np.dot(mo_coeff, rotmat)
    return mo_coeff_new 

def rotation(mo_coeff, rotmat):
    '''
    Thouless rotation of a Slater determinant: exp(rotmat E1)|Phi>, where E1 is the one-body excitation matrix.
    ..math:
        C_{occ}^{new} = C_{occ} + C_{vir}Z 

    rotmat = [I Z]^T, so C_{occ}^{new} = C rotmat
    Input:
        mo_coeff: either 2D array or a list of two 2D arrays.
        rotmat: either a 2D array of size (Norb x Nocc) or a list of two 2D arrays.
    Output:
        The MO coefficients for the new determinant.
    '''
    try:
        ndim = mo_coeff.ndim 
    except:
        ndim = 3

    if ndim > 2: # two spins
        try:
            ndim_r = rotmat.ndim
        except:
            ndim_r = 3 

        if ndim_r > 2:
            mo_coeff_new = np.array([mo_coeff[0]@rotmat[0], mo_coeff[1]@rotmat[1]])
        else: # spin-up and down use the same rotation matrix
            mo_coeff_new = np.array([mo_coeff[0]@rotmat, mo_coeff[1]@rotmat])

    else: # one spin 
        mo_coeff_new = np.dot(mo_coeff, rotmat)

    return mo_coeff_new

def normalize_rotmat_spinless(rmat, norm_all=True):
    '''
    Normalize the spinless rotation matrix.
    Normalization is not needed, but we have it in case.
    This function is not costy.
    '''

    rnorm = np.linalg.norm(rmat, axis=0)
    rmat_n = np.divide(rmat, rnorm)
    
    if norm_all:
        nocc = rmat.shape[-1]
        r_norm = norm_rotmat_spinless(rmat_n)
        rmat_n /= (r_norm ** (0.5/nocc))

    return rmat_n


def orthonormal_rotmat_spinless(rmat):
    '''
    Orthogonalization is not required. 
    This function is much slower than normalize_rotmat_spinless
    Carlos thesis Section 3.4
    I + Z^T Z^* = L L^\dag (Cholesky decomp)
    rmat -> rmat L^{-1, T}
    '''
    norb, nocc = rmat.shape
    tmat = rmat[nocc:]
    LLd = np.eye(nocc) +tmat.T @ tmat.conj() 
    L = np.linalg.cholesky(LLd)
    L_invT = np.linalg.inv(L).T
    rmat_n = rmat @ L_invT 
    return rmat_n

def normalize_rotmat(rmat):
    '''
    Normalize the rotation matrix.
    '''
    try:
        ndim = rmat.ndim
    except:
        ndim = 3
    
    if ndim > 2: # two spins
        # normalize column-wise
        rmat_n_up = normalize_rotmat_spinless(rmat[0])
        rmat_n_dn = normalize_rotmat_spinless(rmat[1])
        rmat_n = [rmat_n_up, rmat_n_dn]

    else:
        rmat_n = normalize_rotmat_spinless(rmat)
   
    return rmat_n


def gen_determinants(mo_coeff, rmats, normalize=False):
    '''
    Generate all configurations from rotation.
    Args:
        mo_coeff: 2D numpy array of size (Norb, Norb)
        rotmats:  list of rotation matrices.
        nocc: int, number of occupied orbitals
    Returns:
        a list of MO coefficients.
    '''
    num_rot = len(rmats)
    if normalize:
        for i in range(num_rot):
            rmats[i] = normalize_rotmat(rmats[i])

    sdets = []
    for iter in range(num_rot):
        r = rmats[iter]
        sdets.append(rotation(mo_coeff, r))

    return sdets

def gen_dets_thouless(mo_coeff, tmats, normalize=False):
    '''
    Generate determinants from Thouless rotation matrices.
    '''
    num_tmats = len(tmats)
    sdets = []
    for i in range(num_tmats):
        rmat = thouless_to_rotation(tmats[i])
        if normalize:
            rmat = normalize_rotmat(rmat)
        sdets.append(rotation(mo_coeff, rmat))

    return sdets
                
def thouless_to_rotation(tmat, normalize=False):
    '''
    Turn a Thouless matrix to a rotation matrix.
    NOTE: the rotation matrix is not normalized.
    '''
    tmat = np.asarray(tmat)
    ndim = tmat.ndim
    nocc = tmat.shape[-1]
    if ndim > 2: # two spins
        rmat = _t2r_uhf(tmat, nocc)
    else:
        rmat = _t2r_rhf(tmat, nocc)
    if normalize: 
        rmat = normalize_rotmat(rmat)
    return rmat  


def thouless_to_rotation_all(tmats, normalize=False):

    tmats = np.asarray(tmats)
    ndim = tmats[0].ndim
    tshape = tmats.shape
    nt = tshape[0]
    nocc = tshape[-1]
    rmats = []
    if ndim > 2: # two spins
        for i in range(nt):
            rmats.append(_t2r_uhf(tmats[i], nocc, normalize=normalize))
    else:
        for i in range(nt):
            rmats.append(_t2r_rhf(tmats[i], nocc, normalize=normalize))
    return rmats


def _t2r_uhf(t, nocc, normalize=False):
    t = np.asarray(t)
    Imat = np.eye(nocc)
    id = np.array([Imat, Imat])
    rmat = np.concatenate([id, t], axis=1)
    if normalize:
        rmat = normalize_rotmat(rmat)
    return rmat


def _t2r_rhf(t, nocc, normalize=False):
    Imat = np.eye(nocc)   
    rmat = np.concatenate([Imat, t], axis=0)
    if normalize:
        rmat = normalize_rotmat(rmat)
    return rmat


def ovlp_sdet(sdet1, sdet2, ao_ovlp=None, tol=1e-10):
    '''
    Evaluate the overlap between two Slater determinants.
    Args:
        sdet1: a list of two arrays
        sdet2: a list of two arrays
    Kwargs:
        ao_ovlp: 2D array, the overlap matrix of the basis if the basis is non-orthogonal.
    Returns:
        A scalar: the overlap between the two Slater determinants.
    '''

    ovlp_mat = metric_sdet(sdet1, sdet2, ao_ovlp=ao_ovlp)
    ovlp = np.linalg.det(ovlp_mat[0]) * np.linalg.det(ovlp_mat[1])

    if (abs(ovlp) < tol):
        print("WARNING: ovlp_sdet() overlap is too small: {:.2e}".format(ovlp))

    return ovlp 

def norm_rotmat_spinless(rmat):
    ovlp_mat = np.dot(rmat.T.conj(), rmat)
    ovlp = np.linalg.det(ovlp_mat) 
    return ovlp

def ovlp_rotmat(rmat1, rmat2, spin=True, tol=1e-10):
    '''
    Evaluate the overlap between two Slater determinants from the rotation matrix.
    Only correct when mo_coeff is unitary.
    Returns:
        A scalar: the overlap between the two Slater determinants.
    '''
    
    ovlp_mat = metric_rotmat(rmat1, rmat2)
    try:
        ndim = rmat1.ndim
    except:
        ndim = 3

    if ndim > 2: # two spins
        ovlp = np.linalg.det(ovlp_mat[0]) * np.linalg.det(ovlp_mat[1])
    else: # one spin
        ovlp = np.linalg.det(ovlp_mat) 
        if spin:
            ovlp = ovlp ** 2

    if (abs(ovlp) < tol):
        print("WARNING: ovlp_rotmat overlap is too small: {:.2e}".format(ovlp))

    return ovlp 

def metric_sdet(sdet1, sdet2, ao_ovlp=None):
    '''
    Get the overlap matrix or metric between two Slater determinants.
    Output:
        A 2D array of size (Nocc x Nocc).
    '''
    if ao_ovlp is None:
        ovlp_mat = np.array([sdet1[0].T.conj()@sdet2[0], sdet1[1].T.conj()@sdet2[1]])
    else:
        ovlp_mat = np.array([sdet1[0].T.conj()@ao_ovlp@sdet2[0], sdet1[1].T.conj()@ao_ovlp@sdet2[1]])


    return ovlp_mat

def metric_rotmat(rmat1, rmat2):
    '''
    No need for the AO overlap matrix.
    '''
    try:
        ndim = rmat1.ndim
    except:
        ndim = 3
        
    if ndim > 2:
        ovlp_mat = np.array([rmat1[0].T.conj()@rmat2[0], rmat1[1].T.conj()@rmat2[1]])
    else:
        omat = np.dot(rmat1.T.conj(), rmat2)
        ovlp_mat = omat

    return ovlp_mat

def make_trans_rdm1(sdet1, sdet2, ao_ovlp=None, omat=None, tol=1e-10, return_ovlp=True):
    '''
    Evaluate the transition density matrix rho_{ij} = <Phi_1|a^\dag_i a_j|Phi_2>
    rho_{12} = C_2 M^-1 C_1^\dag
    where M = C_1^\dag C_2
    NOTE: didn't consider the orthogonal case.
    Output:
        A list of 2D arrays.
        float: reduced overlap
    '''
    if omat is None:
        omat = metric_sdet(sdet1, sdet2, ao_ovlp=ao_ovlp)
    dm_u, ou = make_trans_rdm1_spinless(sdet1[0], sdet2[0], omat[0])
    dm_d, od = make_trans_rdm1_spinless(sdet1[1], sdet2[1], omat[1])
    dm = [dm_u, dm_d]
    ovlp = ou * od

    if return_ovlp:
        return dm, ovlp
    else:
        return dm

def make_trans_rdm1_spinless(s1, s2, omat, tol=1e-10, s_tol=1e-8):
    '''
    Make transition density matrix. <s1 |a^dag a| s2>
    Args:
        s1: MO coefficient for determinant 1
        s2: MO coefficient for determinant 2
        omat: overlap matrix between s1 and s2
    Returns:
        spinless density matrix
        reduced overlap
    '''
    ovlp = np.linalg.det(omat)
    inv = np.linalg.inv(omat)
    dm = s2 @ inv @ s1.T.conj()
    rovlp = ovlp

    return dm, rovlp
    
def rdm1_to_rdm2(dml, dmr=None):
    '''
    Effective two body transition rdm2
    # TODO: evaluate J and K terms separately.
    # TODO: this is very inefficient.
    '''
    if dmr is None:
        dmr = dml
       
    dm_all_l = dml[0] + dml[1]
    dm_all_r = dmr[0] + dmr[1]

    dm2 = einsum("ji, lk -> jilk", dm_all_l, dm_all_r) # J term
    dm2 -= einsum("li, jk -> jilk", dml[0], dmr[0]) 
    dm2 -= einsum("li, jk -> jilk", dml[1], dmr[1])  # K term

    return dm2

def get_j(h2e, dm):
    '''
    J_ij  =  sum_k,l  (i j | k l) rho_lk
    '''
    try:
        ndim = dm.ndim
    except:
        ndim = 3

    if ndim > 2:
        dm_all = dm[0] + dm[1]
        # print(dm_all.shape)
        # print(h2e.shape)
        jab = np.einsum('ijkl, lk -> ij', h2e, dm_all)
        return np.array([jab, jab])
    else:
        return np.einsum('ijkl, lk -> ij', h2e, dm)

def get_k(h2e, dm):
    '''
    K_il  =  sum_j,k  (i j | k l) P_jk
    '''
    try:
        ndim = dm.ndim
    except:
        ndim = 3

    if ndim > 2:    
        return np.einsum('ijkl, njk -> nil', h2e, dm)
    else:
        return np.einsum('ijkl, jk -> il', h2e, dm)
    
def get_jk(h2e, dm):
    try:
        ndim = dm.ndim
    except:
        ndim = 3
    J = get_j(h2e, dm)
    K = get_k(h2e, dm)
    if ndim > 2:
        return J - K 
    else:
        return J - 0.5*K
    
def get_jk_pyscf(mf, dm):
    vhf = mf.get_veff(mf.mol, dm, hermi=0)
    return vhf

def trans_hamilt_all(dm, h1e, h2e, mo_coeff=None, get_grad=False):
    '''
    Evaluate <sdet1|H|sdet2> / ovlp. 
    If needed, return gradient and overlap.
    Args:
        dm: transition density matrix
        h1e: 1-body Hamiltonian
        h2e: 2-body Hamiltonian
        
    Kwargs:
        mo_coeff: Orbital coefficients of HF solutions (molecular orbitals)
        get_grad: if True, return the gradient d <sdet1|H|sdet2>/ d(Z2)

    Returns:
        Float of <sdet1|H|sdet2>
        Optional: matrix of d <sdet1|H|sdet2>/ d(Z2)
                  
    '''

    norb = h1e.shape[0]
    
    # evaluate energy
    jk = get_jk(h2e, dm)
    hval1 = einsum('ij, nji -> ', h1e, dm)
    hval2 = np.einsum('nij, nji -> ', jk, dm)
    hval = (hval1 + 0.5 * hval2) 

    # evaluate gradient
    if get_grad:
        I = np.eye(norb)
        heff = np.array([h1e, h1e]) + jk
        g_up =  (I - dm[0]) @ heff[0] @ dm[0]
        g_dn =  (I - dm[1]) @ heff[1] @ dm[1]
        grad = np.array([g_up, g_dn]) 

        return hval, grad
    else:
        return hval
    

def trans_hamilt(sdet1, sdet2, h1e, h2e, ao_ovlp=None, 
                          tol=1e-10, diag_tol=1e-8, mf=None):
    '''
    Evaluate <sdet1|H|sdet2>.
    Deals with singularity cases.
    Used the generalized Slater-Condon rules.
    Koch and Dalgaard, Chem. Phys. Lett., 212, 193 (1993)
    Thom and Head-Gordon, J. Chem. Phys. 131, 124113 (2009)
    TODO test singularity cases
    TODO move each spin into a function, return rdm1, dml, dmr, ovlp.
    '''

    # evaluate overlap and check singularity
    ovlp_mat = metric_sdet(sdet1, sdet2, ao_ovlp=ao_ovlp)
    ovlp_u, ovlp_d = np.linalg.det(ovlp_mat)

    if abs(ovlp_u) > tol and abs(ovlp_d) > tol:

        dm, ovlp = make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp, omat=ovlp_mat)
        if mf is None:
            E = trans_hamilt_dm(dm, ovlp, h1e, h2e) 
        else:
            E = trans_hamilt_pyscf(mf, dm, ovlp)

    else:
        omat_u, omat_d = ovlp_mat[0], ovlp_mat[1]
        sd1_u, sd2_u = sdet1[0], sdet2[0]
        sd1_d, sd2_d = sdet1[1], sdet2[1]
        # examine the two spins separately
        # SPIN UP
        if abs(ovlp_u) > tol:
            inv = np.linalg.inv(omat_u)
            rdm1_u = sd2_u @ inv @ sd1_u.T.conj()
            rovlp_u = ovlp_u
            dml_u = rdm1_u 
            dmr_u = rdm1_u 
        else:
            print("WARNING: overlap is too small: {:.2e}".format(ovlp_u))
            u, s, vh = np.linalg.svd(omat_u)
            v = vh.conj().T
            nker = int(np.sum(s < diag_tol))

            if nker == 0: # no singularity
                inv = np.linalg.inv(omat_u)
                rdm1_u = sd2_u @ inv @ sd1_u.T.conj()
                rovlp_u = ovlp_u
                dml_u = rdm1_u 
                dmr_u = rdm1_u 
            elif nker == 1: 
                sd1_n = np.dot(sd1_u, u[:, -1])
                sd2_n = np.dot(sd2_u, v[:, -1])
                rdm1_u = sd2_n @ sd1_n.conj().T
                rovlp_u = np.product(s[:-1])
                inv_red = np.dot(np.dot(u[:, :-1], np.diag(1./s[:-1])), vh[:-1])
                dml_u = sd2_u @ inv_red @ sd1_u.conj().T # TODO check against eq (9) Thompson
                dmr_u = rdm1_u 
            elif nker == 2:
                sd1_n = np.dot(sd1_u, u[:, -1])
                sd2_n = np.dot(sd2_u, v[:, -1])
                rdm1_u = 0
                dml_u = sd2_n @ sd1_n.conj().T
                dmr_u = dml_u

        # SPIN DOWN
        if abs(ovlp_d) > tol:
            inv = np.linalg.inv(omat_d)
            rdm1_d = sd2_d @ inv @ sd1_d.T.conj()
            rovlp_d = ovlp_d
            dml_d = rdm1_d 
            dmr_d = rdm1_d 
        else:
            print("WARNING: overlap is too small: {:.2e}".format(ovlp_u))
            u, s, vh = np.linalg.svd(omat_d)
            v = vh.conj().T
            nker = int(np.sum(s < diag_tol))
            if nker == 0: # no singularity
                inv = np.linalg.inv(omat_d)
                rdm1_d = sd2_d @ inv @ sd1_d.T.conj()
                rovlp_d = ovlp_d
                dml_d = rdm1_d 
                dmr_d = rdm1_d 
            elif nker == 1: 
                sd1_n = np.dot(sd1_d, u[:, -1])
                sd2_n = np.dot(sd2_d, v[:, -1])
                rdm1_d = sd2_n @ sd1_n.conj().T
                rovlp_d = np.product(s[:-1])
                inv_red = np.dot(np.dot(u[:, :-1], np.diag(1./s[:-1])), vh[:-1])
                dml_d = sd2_d @ inv_red @ sd1_d.conj().T # TODO check against eq (9) Thompson
                dmr_d = rdm1_d 
            elif nker == 2:
                sd1_n = np.dot(sd1_d, u[:, -1])
                sd2_n = np.dot(sd2_d, v[:, -1])
                rdm1_d = 0
                dml_d = sd2_n @ sd1_n.conj().T
                dmr_d = dml_d
                rovlp_d = np.product(s[:-2])

        rdm1_tot = rdm1_u + rdm1_d 
        rdm2 = rdm1_to_rdm2([dml_u, dml_d], [dmr_u, dmr_d])
        E1 = einsum('ij, ji -> ', h1e, rdm1_tot)
        E2 = einsum('ijkl, jilk ->', h2e, rdm2)
        E =  (E1 + 0.5 * E2) * rovlp_u * rovlp_d

    return E

def trans_hamilt_dm(dm, ovlp, h1e, h2e): 
    '''
    Evaluate the value of <Phi_1|H|Phi_2>
    Input:
        dm: 2D numpy array of size (Norb x Norb), the transition density matrix between Phi_1 and Phi_2.
        ovlp: the overlap between Phi_1 and Phi_2.
        h1e: one-body Hamiltonian 
        h2e: two-body Hamiltonian
    Return:
        A scalar <Phi_1|H|Phi_2> (not the energy)
    '''
    try: 
        ndim = dm.ndim
    except:
        ndim = 3
 
    jk = get_jk(h2e, dm)
    if ndim > 2:
        E1 = einsum('ij, nji -> ', h1e, dm)
        E2 = np.einsum('nij, nji -> ', jk, dm)
    else:
        E1 = einsum('ij, ji -> ', h1e, dm)
        E2 = np.einsum('ij, ji -> ', jk, dm)

    E = (E1 + 0.5 * E2) * ovlp

    return E

def trans_hamilt_pyscf(mf, dm, ovlp):
    '''
    Using PySCF for electronic energy. 
        ... math::
            
            E = \sum_{ij}h_{ij} \gamma_{ji} 
              + \frac{1}{2}\sum_{ijkl} \gamma_{ji}\gamma_{lk} \langle ik||jl\rangle   

    Input:
        mf: scf object in pyscf
        dm: the density matrix.
        ovlp: the overlap between the two Slaters.
    Output:
        A scalar, the energy corresponding to dm.
    '''
    vhf = get_jk_pyscf(mf, dm) #hermi is for J and K, still doesnt work
    return mf.energy_elec(dm=dm, vhf=vhf)[0] * ovlp 

