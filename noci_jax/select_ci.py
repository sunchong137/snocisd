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
Selected NOCI.
Following Dutta et. al., J. Chem. Phys. 154, 114112 (2021)
'''
import numpy as np
from noci_jax import slater


def select_rmats(rmats_fix, rmats_new, mo_coeff, h1e, h2e, m_tol=1e-5, 
                 e_tol=5e-8, max_ndets=None):
    '''
    First select non-linearly dependant determinants, then energy contribution.
    Note: the returned determinants include rmats_fix.
    '''
    n_fix = len(rmats_fix)
    n_new = len(rmats_new)

    if max_ndets is None:
        max_ndets = n_new

    r_select = select_rmats_ovlp(rmats_fix, rmats_new, m_tol=m_tol, max_ndets=n_new)
    r_select = select_rmats_energy(rmats_fix, r_select[n_fix:], mo_coeff, h1e, h2e, 
                                      e_tol=e_tol, max_ndets=max_ndets)
    return r_select

def select_rmats_slow(rmats_fix, rmats_new, mo_coeff, h1e, h2e, m_tol=1e-5, 
                 e_tol=5e-8, max_ndets=None):
    '''
    Similar to select_tvecs(), this function selects rotation 
    matrices of size (N, 2, norb, nocc). One should use this function
    when there is multi-reference, i.e. more than one set of MO 
    coefficients.
    TODO: Rewrite to save time. First evaluate overlap, then hmat, and energy.
    '''
    print("***Selecting determinants based on overlap and energy contribution.")
    hmat_fix, smat_fix = slater.noci_matrices(rmats_fix, mo_coeff, h1e, h2e)
    n_new = len(rmats_new)
    # selected_indices = []

    if max_ndets is None:
        max_ndets = n_new
        
    count = 0
    for i in range(n_new):
        if count == max_ndets:
            print("Maximum number of determinant exceeded, please try to increase the overlap threshold.")
            break

        r_new = rmats_new[i][None, :]
        m, e, h, s = criteria_all_single_det(rmats_fix, r_new[0], mo_coeff, h1e, h2e, 
                                         smat_fix=smat_fix, hmat_fix=hmat_fix)
        # print(m, e)
        if m > m_tol and abs(e) > e_tol:
            hmat_fix = h
            smat_fix = s
            rmats_fix = np.vstack([rmats_fix, r_new])
            # selected_indices.append(i)
            count += 1
        else:
            continue

    # num_added = len(selected_indices)
    print("***Selected CI Summary:***")
    print("Metric Threshold: {:.1e}".format(m_tol))
    print("Energy Threshold: {:.1e}".format(e_tol))
    print("Reduced {} determinants to {} determinants.".format(n_new, count))
    return rmats_fix


def select_rmats_ovlp(rmats_fix, rmats_new, m_tol=1e-5, max_ndets=None, return_indices=False):
    '''
    Only consider the overlap criteria.
    Much faster than the select_rmat, and serve the same purpose.
    '''
    smat_fix = slater.get_smat(rmats_fix)
    n_new = len(rmats_new)
    if max_ndets is None:
        max_ndets = n_new
    # selected_indices = []
    count = 0
    idx_select = []

    # print information
    print("#"*40)
    print("# Selecting determinants based on overlap.")
    print("# overlap threshold: {:1.2e}".format(m_tol))

    for i in range(n_new):
        if count == max_ndets:
            print("# Maximum number of determinant exceeded, please try to increase the overlap threshold.")
            break
        r_new = rmats_new[i][None, :]
        m, s = criterial_ovlp_single_det(rmats_fix, r_new[0], smat_fix=smat_fix, m_tol=m_tol)
        # print(m)
        if m > m_tol:
            smat_fix = s
            rmats_fix = np.vstack([rmats_fix, r_new])
            idx_select.append(i)
            count += 1
        else:
            continue

    # num_added = len(selected_indices)
    print("# Reduced {} determinants to {} determinants.".format(n_new, count))
    if return_indices:
        return rmats_fix, idx_select
    else:
        return rmats_fix


def select_rmats_energy(rmats_fix, rmats_new, mo_coeff, h1e, h2e, 
                        e_tol=1e-5, max_ndets=None):
    '''
    Select via energy.
    '''
    hmat_fix, smat_fix = slater.noci_matrices(rmats_fix, mo_coeff, h1e, h2e)
    n_new = len(rmats_new)
    if max_ndets is None:
        max_ndets = n_new
    
    # print information
    print("#"*40)
    print("# Selecting determinants based on energy contribution.")
    print("# overlap threshold: {:1.2e}".format(e_tol))

    count = 0
    # selected_indices = []
    for i in range(n_new):
        if count == max_ndets:
            print("Maximum number of determinant exceeded, please try to increase the overlap threshold.")
            break
        r_new = rmats_new[i][None, :]
        e, h, s = criteria_energy_single_det(rmats_fix, r_new[0], mo_coeff, h1e, h2e, 
                                         smat_fix=smat_fix, hmat_fix=hmat_fix)
        # print(m, e)
        if abs(e) > e_tol:
            hmat_fix = h
            smat_fix = s
            rmats_fix = np.vstack([rmats_fix, r_new])
            # selected_indices.append(i)
            count += 1
        else:
            continue
    print("# Reduced {} determinants to {} determinants.".format(n_new, count))
    return rmats_fix


def check_linear_depend(rmats, ovlp_mat=None, tol=1e-6):
    '''
    Check the linear dependancy of a set of vectors.
    Args:
        ovlp_mat: 2D array, overlap matrix
    Kwargs:
        tol: zero tolerace
    Returns:
        Dimension of the space spanned by the vectors, i.e., number of linearly-independent vectors.
    '''
    if ovlp_mat is None:
        metrics_all = np.einsum('nsji, msjk -> nmsik', rmats.conj(), rmats)
        ovlp_mat = np.prod(np.linalg.det(metrics_all), axis=-1)

    _, s, _ = np.linalg.svd(ovlp_mat)
    num_ind = np.sum(s > tol) # S is positive semi definite
    
    return num_ind

def criteria_ovlp(rmats_fix, rmats_new, smat_fix=None):
    '''
    Check if a new vector is not linearly dependent to the existing NOCI pool.
    '''
    nr0 = len(rmats_fix)

    if smat_fix is None:
        rmats_all = np.vstack([rmats_fix, rmats_new])
        smat_all = slater.get_smat(rmats_all)
        smat_fix = smat_all[:nr0, :nr0]
    else:
        smat_all = slater.expand_smat(smat_fix, rmats_fix, rmats_new)

    smat_mix_l = smat_all[:nr0, nr0:]
    smat_new = smat_all[nr0:, nr0:]

    # calculate the residual 
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = np.einsum("np, pq, qn -> n", smat_mix_l.T.conj(), inv_fix, smat_mix_l)
    norm_new_proj = 1.0 - proj_old/np.diag(smat_new)

    return norm_new_proj


def criterial_ovlp_single_det(rmats_fix, r_new, smat_fix=None, m_tol=1e-5):
    '''
    Linear independence criterial for one determinant.
    '''
    if smat_fix is None:
        rmats_all = np.vstack([rmats_fix, r_new[None, :]])
        smat_all = slater.get_smat(rmats_all)
        smat_fix = smat_all[:-1, :-1]
        smat_left = smat_all[:-1, -1] # (nr0, 1)
        s_new = smat_all[-1, -1]
    else:
        metrics_mix = np.einsum('nsji, sjk -> nsik', rmats_fix.conj(), r_new)
        smat_left = np.prod(np.linalg.det(metrics_mix), axis=-1)
        s_new = np.einsum("sji, sjk -> sik", r_new.conj(), r_new)
        s_new = np.prod(np.linalg.det(s_new), axis=-1)


    # calculate the residual 
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = smat_left.T.conj() @ inv_fix @ smat_left
    proj_new = 1.0 - proj_old/s_new

    if proj_new > m_tol:
        nr = len(rmats_fix) + 1
        smat_all = np.zeros((nr, nr))
        smat_all[:-1, :-1] = smat_fix
        smat_all[:-1, -1]  = smat_left
        smat_all[-1, :-1]  = smat_left.conj().T
        smat_all[-1, -1]   = s_new
        return proj_new, smat_all
    
    else:
        return proj_new, None


def criteria_all(rmats_fix, rmats_new, mo_coeff, h1e, h2e, E_fix=None, 
                   noci_vec=None, smat_fix=None, hmat_fix=None):
    '''
    Evaluate the linear dependency and energy contribution of the new vectors.
    '''

    nr0 = len(rmats_fix)

    if smat_fix is None or hmat_fix is None:
        rmats_all = np.vstack([rmats_fix, rmats_new])
        hmat_all, smat_all = slater.noci_matrices(rmats_all, mo_coeff, h1e, h2e)
        smat_fix = smat_all[:nr0, :nr0]
        hmat_fix = hmat_all[:nr0, :nr0]
    else:
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, rmats_new, rmats_fix, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    smat_mix_l = smat_all[:nr0, nr0:]
    smat_new = smat_all[nr0:, nr0:]
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = np.einsum("np, pq, qn -> n", smat_mix_l.T.conj(), inv_fix, smat_mix_l)
    norm_new = np.diag(smat_new)
    proj_new = 1.0 - proj_old/norm_new
    sdiag_new = norm_new - proj_old

    # calculate the energy contribution
    if E_fix is None:
        E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
    H_fix = E_fix * norm_fix
    alpha = inv_fix @ smat_mix_l  # (nr0, nr) array 
    hmat_mix_l = hmat_all[:nr0, nr0:]
    hmat_new = hmat_all[nr0:, nr0:]
    T_part = noci_vec.conj() @ (hmat_mix_l - hmat_fix @ alpha) # (nr,) array
    H_new = np.diag(hmat_new) - 2*np.real(np.einsum("pn, pn -> n", alpha.conj(), hmat_mix_l))
    H_new = H_new + np.einsum("pn, pq, qn -> n", alpha.conj(), hmat_fix, alpha)
    E_new = H_new / sdiag_new
    R_term = np.sqrt((H_new*norm_fix - H_fix*sdiag_new)**2 + 4*sdiag_new*norm_fix*(np.abs(T_part))**2) 
    de_ratio = (E_new - E_fix - R_term/(sdiag_new * norm_fix)) / (2 * E_fix)

    return proj_new, de_ratio


def criteria_all_single_det(rmats_fix, r_new, mo_coeff, h1e, h2e, E_fix=None, 
                   noci_vec=None, smat_fix=None, hmat_fix=None, metric_tol=1e-6):
    '''
    Evaluate the linear dependency and energy contribution of one new vector.
    '''

    if smat_fix is None or hmat_fix is None:
        rmats_all = np.vstack([rmats_fix, r_new[None, :]])
        hmat_all, smat_all = slater.noci_matrices(rmats_all, mo_coeff, h1e, h2e)
        smat_fix = smat_all[:-1, :-1]
        hmat_fix = hmat_all[:-1, :-1]
    else:
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, r_new[None, :], rmats_fix, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    smat_mix_l = smat_all[:-1, -1] # (nr0, 1)
    s_new = smat_all[-1, -1]
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = smat_mix_l.T.conj()@inv_fix@smat_mix_l
    norm_new = s_new
    proj_new = 1.0 - proj_old/norm_new
    sdiag_new = norm_new - proj_old

    if proj_new < metric_tol:
        de_ratio = 0
    else:
        # calculate the energy contribution
        if E_fix is None:
            E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

        norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
        H_fix = E_fix * norm_fix
        alpha = inv_fix @ smat_mix_l  # (nr0, 1) array 
        hmat_mix_l = hmat_all[:-1, -1]
        T_part = noci_vec.conj() @ (hmat_mix_l - hmat_fix @ alpha) # (nr,) array
        H_new = hmat_all[-1, -1] - 2*np.real(alpha.conj().T@hmat_mix_l)
        H_new = H_new + alpha.conj().T@hmat_fix@alpha
        E_new = H_new / sdiag_new
        R_term = np.sqrt((H_new*norm_fix - H_fix*sdiag_new)**2 + 4*sdiag_new*norm_fix*(np.abs(T_part))**2) 
        de_ratio = (E_new - E_fix - R_term/(sdiag_new * norm_fix)) / (2 * E_fix)

    return proj_new, de_ratio, hmat_all, smat_all


def criteria_energy_single_det(rmats_fix, r_new, mo_coeff, h1e, h2e, E_fix=None, 
                   noci_vec=None, smat_fix=None, hmat_fix=None):
    '''
    Evaluate the energy contribution of one new vector.
    '''

    if smat_fix is None or hmat_fix is None:
        rmats_all = np.vstack([rmats_fix, r_new[None, :]])
        hmat_all, smat_all = slater.noci_matrices(rmats_all, mo_coeff, h1e, h2e)
        smat_fix = smat_all[:-1, :-1]
        hmat_fix = hmat_all[:-1, :-1]
    else:
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, r_new[None, :], rmats_fix, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    smat_mix_l = smat_all[:-1, -1] # (nr0, 1)
    s_new = smat_all[-1, -1]
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = smat_mix_l.T.conj()@inv_fix@smat_mix_l
    norm_new = s_new
    sdiag_new = norm_new - proj_old

    if E_fix is None:
        E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
    H_fix = E_fix * norm_fix
    alpha = inv_fix @ smat_mix_l  # (nr0, 1) array 
    hmat_mix_l = hmat_all[:-1, -1]
    T_part = noci_vec.conj() @ (hmat_mix_l - hmat_fix @ alpha) # (nr,) array
    H_new = hmat_all[-1, -1] - 2*np.real(alpha.conj().T@hmat_mix_l)
    H_new = H_new + alpha.conj().T@hmat_fix@alpha
    E_new = H_new / sdiag_new
    R_term = np.sqrt((H_new*norm_fix - H_fix*sdiag_new)**2 + 4*sdiag_new*norm_fix*(np.abs(T_part))**2) 
    epsilon = (E_new - R_term/(sdiag_new * norm_fix))
    # print("Check", epsilon, E_fix)
    de_ratio = (E_fix - epsilon) / abs(E_fix)

    return de_ratio, hmat_all, smat_all

def eval_epsilon(rmats_fix, r_new, mo_coeff, h1e, h2e, E_fix=None, 
                   noci_vec=None, smat_fix=None, hmat_fix=None):
    '''
    Evaluate epsilon in the energy criteria.
    |epsilon - E0|/|E_0| > h_0
    '''

    if smat_fix is None or hmat_fix is None:
        rmats_all = np.vstack([rmats_fix, r_new[None, :]])
        hmat_all, smat_all = slater.noci_matrices(rmats_all, mo_coeff, h1e, h2e)
        smat_fix = smat_all[:-1, :-1]
        hmat_fix = hmat_all[:-1, :-1]
    else:
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, r_new[None, :], rmats_fix, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    smat_mix_l = smat_all[:-1, -1] # (nr0, 1)
    s_new = smat_all[-1, -1]
    inv_fix = np.linalg.inv(smat_fix)
    proj_old = smat_mix_l.T.conj()@inv_fix@smat_mix_l
    norm_new = s_new
    sdiag_new = norm_new - proj_old

    E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
    H_fix = E_fix * norm_fix
    alpha = inv_fix @ smat_mix_l  # (nr0, 1) array 
    hmat_mix_l = hmat_all[:-1, -1]
    T_part = noci_vec.conj() @ (hmat_mix_l - hmat_fix @ alpha) # (nr,) array
    H_new = hmat_all[-1, -1] - 2*np.real(alpha.conj().T@hmat_mix_l)
    H_new = H_new + alpha.conj().T@hmat_fix@alpha
    E_new = H_new / sdiag_new
    R_term = np.sqrt((H_new*norm_fix - H_fix*sdiag_new)**2 + 4*sdiag_new*norm_fix*(np.abs(T_part))**2) 
    epsilon = (E_new - R_term/(sdiag_new * norm_fix))
    return epsilon