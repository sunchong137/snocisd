'''
Non-orthogonal configuration interaction.
'''
import time
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize
import slater
import jax 
import jax.numpy as jnp
from jax.scipy import linalg as jsla


def gen_thouless_random(nocc, nvir, max_nt):

    tmats = []
    tshape = (2, nvir, nocc)
    for i in range(max_nt):
        t = np.random.rand(2, nvir, nocc)
        #t = np.random.normal(size=tshape)
        tmats.append(t)

    return np.asarray(tmats)

def gen_thouless_singles(nocc, nvir, max_nt=None, zmax=10, zmin=0.1):
    '''
    Generate rotations for near singly excited state for spinless systems.
    Input:
        nocc: number of occupied orbitals.
        nvir: number of virtual orbitals.
    Kwargs:
        max_nrot: maximum number of matrices to generate.
    Returns:
        A list of unnormalized Thouless parameters.
    '''

    if max_nt is None:
        max_nt = nvir * nocc

    # pick the excitations closest to the Fermi level    
    sqrt_nt = int(np.sqrt(max_nt)) + 1
    if nocc < nvir:
        if nocc < sqrt_nt: 
            d_occ = nocc 
            d_vir = nvir  
        else:
            d_occ = sqrt_nt 
            d_vir = sqrt_nt
    else:
        if nvir < sqrt_nt:
            d_occ = nocc 
            d_vir = nvir 
        else:
            d_occ = sqrt_nt 
            d_vir = sqrt_nt

    tmats = []
    t0 = np.zeros((nvir, nocc))
    k = 0
    for i in range(d_occ): # occupied
        for j in range(d_vir): # virtual
            if k == max_nt:
                break
            tm = np.ones((nvir, nocc)) * zmin 
            tm[j, nocc-i-1] = zmax
            tmats.append(np.array([tm, t0]))
            tmats.append(np.array([t0, tm]))
            k += 1
    tmats = np.asarray(tmats)
    return tmats


def gen_thouless_doubles(nocc, nvir, max_nt=None, zmax=10, zmin=0.1):
    '''
    Generate rotations for near doubly excited state for spinless systems.
    Since (i -> a, j -> b) and (j -> a, i -> b) corresponds to the same determinant,
    we do not allow cross excitation, i.e., for (i -> a, j -> b), i < j and a < b.

    '''
    if max_nt is None:
        max_nt = int(nvir*(nvir-1)/2) * int(nocc*(nocc-1)/2)
    max_nt = min(max_nt, int(nvir*(nvir-1)/2) * int(nocc*(nocc-1)/2))

    tmats = []
    k = 0
    t0 = np.zeros((nvir, nocc))
    for i in range(nocc-1): # top e occ
        for j in range(i+1, nocc): # bot e occ
            for a in range(1, nvir): # top e vir
                for b in range(a): # bot e occ
                    if k == max_nt:
                        break
                    tm = np.ones((nvir, nocc)) * zmin 
                    tm[a, nocc-i-1] = zmax
                    tm[b, nocc-j-1] = zmax # HOMO electron is further excited
                    tmats.append([tm, t0])
                    tmats.append([t0, tm])
                    k += 1
    tmats = np.asarray(tmats)
    return tmats


def noci_energy(rmats, mo_coeff, h1e, h2e, ao_ovlp=None, include_hf=False, grad_idx=None, rmats2=None, **kwargs):
    '''
    Get the ground state energy with noci.
    Args:
        rmats: a list of arrays, rotation matrices.
        mo_coeff: Hartree-Fock solution
        h1e: one-body Hamiltonian
        h2e: two-body Hamiltonian
    Kwargs:
        ao_ovlp: overlap matrix for AO basis
        include_hf: if the rmats include the HF state (no rotation).
    Returns:
        float, energy of noci
    '''

    norb, nocc = rmats[0][0].shape
    mo_coeff = np.asarray(mo_coeff)

    if not include_hf:
        rot0_u = np.zeros((norb, nocc))
        rot0_u[:nocc, :nocc] = np.eye(nocc)
        rot_hf = np.array([rot0_u, rot0_u]) # the HF state
        rmats_n = list(rmats) + [rot_hf]
    else:
        rmats_n = rmats
    
    if rmats2 is None:
        rmats2_n = rmats_n
    else:
        if not include_hf:
            rmats2_n =  list(rmats2) + [rot_hf]
        else:
            rmats2_n = rmats2

    nt = len(rmats_n)

    hmat = np.zeros((nt, nt))
    smat = np.zeros((nt, nt))

    if grad_idx is not None:
        grads = []
        mo_vir = mo_coeff[:, :, nocc:]
        mo_occ = mo_coeff[:, :, :nocc]

    for i in range(nt):
        for j in range(nt):
            sdet1 = slater.rotation(mo_coeff, rmats_n[i])
            sdet2 = slater.rotation(mo_coeff, rmats2_n[j])
            dm, ovlp = slater.make_trans_rdm1(sdet1, sdet2, ao_ovlp=ao_ovlp)
            if j == grad_idx: # evaluate gradient
                h, _g = slater.trans_hamilt_all(dm, h1e, h2e, get_grad=True)
                g_up = mo_vir[0].T @ _g[0] @ mo_occ[0].conj() * ovlp
                g_dn = mo_vir[1].T @ _g[1] @ mo_occ[1].conj() * ovlp
                grads.append(np.array([g_up, g_dn]))
            else:
                h = slater.trans_hamilt_all(dm, h1e, h2e, get_grad=False)
            h *= ovlp
            hmat[i, j] = h 
            smat[i, j] = ovlp

    energy, lc = solve_lc_coeffs(hmat, smat, return_vec=True)

    if grad_idx is not None:
        grad = np.zeros_like(grads[0])
        for i in range(nt):
            grad += grads[i] * lc[i].conj()
        grad *= lc[grad_idx]
        
        return energy, grad 
    else:
        return energy
    

def full_ovlp_w_rotmat(rmats, rmats2=None):
    '''
    Evaluate the overlaps between all Slater determinants.
    Args:
        rmats: 3D numpy array of size (n**d, nspin, Norb, Nocc)
        rmats2: another set of rotation matrices
    '''

    if rmats2 is None:
        rmats2 = rmats

    num_rot = len(rmats)
    ovlp_all = jnp.zeros((num_rot, num_rot))
    for i in range(num_rot):
        for j in range(i+1):
            ovlp = slater.ovlp_rotmat(rmats[i], rmats2[j])
            ovlp_all = ovlp_all.at[i,j].set(ovlp)
            ovlp_all = ovlp_all.at[j,i].set(ovlp.conj())

    return ovlp_all

def full_ovlp_w_sdets(sdets, ao_ovlp=None, sdets2=None):
    '''
    Evaluate the overlaps between each Slater determinant.
    Args:
        dets: 3D numpy array of size (n**d, nspin, Norb, Nocc), MO coefficients of the Slater determinants.
    '''
    #num_rot = sdets.shape[0]
    if sdets2 is None:
        sdets2 = sdets

    num_rot = len(sdets)
    ovlp_all = np.zeros((num_rot, num_rot))
    for i in range(num_rot):
        for j in range(i+1):
            ovlp = slater.ovlp_sdet(sdets[i], sdets2[j], ao_ovlp=ao_ovlp)
            ovlp_all[i,j] = ovlp
            ovlp_all[j,i] = ovlp.conj()

    return ovlp_all

def full_hamilt_w_sdets(dets, h1e=None, h2e=None, ao_ovlp=None, mf=None, sdets2=None):
    '''
    Evaluate the full hamiltonian matrix:
        H_ij = <det_i | H | det_j>
    '''
    if mf is None:
        return full_hamilt_w_sdets_direct(dets=dets, h1e=h1e, h2e=h2e, ao_ovlp=ao_ovlp, dets2=sdets2)
    else:
        return full_hamilt_w_sdets_pyscf(dets=dets, mf=mf)

def full_hamilt_w_sdets_direct(dets, h1e, h2e, ao_ovlp=None, dets2=None):
    
    if dets2 is None:
        dets2 = dets

    num_det = len(dets)
    # evaluate Hamiltonian
    ham_mat = jnp.zeros((num_det, num_det))
    for i in range(num_det):
        for j in range(i+1):
            det1 = dets[i]
            det2 = dets2[j] 
            h = slater.trans_hamilt(det1, det2, h1e, h2e, ao_ovlp=ao_ovlp)
            ham_mat = ham_mat.at[i, j].set(h)
            ham_mat = ham_mat.at[j, i].set(h.conj())
    return ham_mat

def full_hamilt_w_sdets_pyscf(dets, mf):

    '''
    Slower than full_hamilt_w_sdets_direct

    '''
    ao_ovlp = mf.mol.intor_symmetric ('int1e_ovlp')
    num_det = len(dets)
    # evaluate Hamiltonian
    ham_mat = np.zeros((num_det, num_det))
    for i in range(num_det):
        for j in range(i+1):
            det1 = dets[i]
            det2 = dets[j] 
            dm, ovlp = slater.make_trans_rdm1(det1, det2, ao_ovlp=ao_ovlp, return_ovlp=True)
            h = slater.trans_hamilt_pyscf(mf, dm, ovlp)
            ham_mat[i, j] = h
            ham_mat[j, i] = h.conj()

    return ham_mat

def generalized_eigh(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    A_redo = L_inv.dot(A).dot(L_inv.T)
    return jnp.linalg.eigh(A_redo)

def solve_lc_coeffs(hmat, smat, return_vec=False):
    '''
    Solve the eigenvalue problem Hc = ESc. 
    Using scipy function, 4x times faster.
    Before: First solve S^-1/2 H S^-1/2 -> v, then c = S^-1/2 v
    Args:
        hmat: 2D numpy array of size (n**d, n**d)
        smat: 2D numpy array of size (n**d, n**d)
    Kwargs:
        return_vec: whether to return the LC coefficients.
    Returns:
        double, ground state energy
        A 1D numpy array of size (n**d,), linear combination coefficient

    '''
    e, v = generalized_eigh(hmat, smat)

    energy = e[0]
    c = v[:, 0]

    if return_vec:
        return energy, c
    else:
        return energy


def num_grad_two_points(tmats, idx, h1e, h2e, mo_coeff, ao_ovlp=None, delt=0.01, include_hf=False):
    '''
    Numerical gradient. Used to test the analytic gradient.
    '''
    grad = np.zeros_like(np.array(tmats[0]))
    nvir, nocc = grad[0].shape
    rmats = slater.thouless_to_rotation_all(tmats, normalize=False)

    for s in range(2): # spin
        for i in range(nvir):
            for j in range(nocc):
                tmats_p = np.copy(tmats)
                tmats_m = np.copy(tmats)
                tmats_p[idx][s][i, j] += delt 
                tmats_m[idx][s][i, j] -= delt 
                rmats_p = slater.thouless_to_rotation_all(tmats_p, normalize=False)
                rmats_m = slater.thouless_to_rotation_all(tmats_m, normalize=False)
                Ep = noci_energy(rmats_p, mo_coeff=mo_coeff, h1e=h1e, h2e=h2e, ao_ovlp=ao_ovlp, include_hf=include_hf, rmats2=rmats)
                Em = noci_energy(rmats_m, mo_coeff=mo_coeff, h1e=h1e, h2e=h2e, ao_ovlp=ao_ovlp, include_hf=include_hf, rmats2=rmats)
                grad[s][i, j] = (Ep - Em)/(2*delt)
    return grad


def noci_gradient_one_det(tmats, idx, h1e, h2e, mo_coeff, ao_ovlp=None, include_hf=False):
    '''
    First assume we do not have the energy evaluation.
    '''

    # generate slater determinants
    mo_coeff = np.asarray(mo_coeff)
    # add hartree-fock
    if not include_hf: # the HF state is not included in tmats
        t_hf = np.zeros_like(tmats[0])
        tmats = list(tmats) + [t_hf]

    rmats = slater.thouless_to_rotation_all(tmats, normalize=False)
    sdets = slater.gen_determinants(mo_coeff, rmats)

    # TODO the following is also evaluating energy
    hmat = full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)  
    smat = full_ovlp_w_rotmat(rmats)
    E0, lc = solve_lc_coeffs(hmat, smat, return_vec=True)

    det0 = sdets[idx]
    c0 = lc[idx]
    grad = np.zeros_like(np.array(tmats[0]))
    
    nocc = grad.shape[-1]
    mo_occ = mo_coeff[:, :, :nocc]
    mo_vir = mo_coeff[:, :, nocc:]
    nt = len(tmats)
    for i in range(nt):
        Eij = hmat[i, idx]
        dm, ovlp = slater.make_trans_rdm1(sdets[i], det0, ao_ovlp=ao_ovlp)
        heff = np.array([h1e, h1e]) + slater.get_jk(h2e, dm)
        g = _gradient_element(dm, mo_vir, mo_occ, Eij, E0, heff)
        grad += g * lc[i].conj() * c0 * ovlp

    # denominator
    denom = lc.conj().T @ smat @ lc  # = 1
    grad /= denom

    return grad

def _gradient_element(dm, mo_vir, mo_occ, Eij, E0, heff):
    '''
    Evaluate <det_fix | (H - E0) b^\dag_p b_h |det0 > / <det_fix|det0>   
    Args:
        r0: rotation matrix for the Thouless parameters to be optimized
        rmat_fix: rotation matrix for the fixed determinant
        E0: the energy with the current NOCIs
        Eij: float, <det_fix| H | det0>
        heff: h1e - get_ik NOTE: no 1/2 factor
    Kwargs:
        ao_ovlp
      
    Returns:
        nd array of size (spin, nvir, nocc)
    '''
    
    norb = mo_occ[0].shape[0]

    rho_c_up =  dm[0] @ mo_occ[0].conj()
    rho_c_dn =  dm[1] @ mo_occ[1].conj()

    # # phi_mu | b^\dag_p b_h | phi_nu> /  S_{mu nu}
    part1_up = mo_vir[0].T @ rho_c_up 
    part1_dn = mo_vir[1].T @ rho_c_dn 

    # exhange term
    I = np.eye(norb)
    part2_up = mo_vir[0].T @ (I - dm[0]) @ heff[0] @ rho_c_up 
    part2_dn = mo_vir[1].T @ (I - dm[1]) @ heff[1] @ rho_c_dn 


    # grad_up = (Eij - E0) * part1_up + part2_up
    # grad_dn = (Eij - E0) * part1_dn + part2_dn
    grad_up = part2_up
    grad_dn = part2_dn

    return np.array([grad_up, grad_dn])
