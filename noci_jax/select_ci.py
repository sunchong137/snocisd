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
import gc

def select_rmats(rmats_fix, rmats_new, mo_coeff, h1e, h2e, m_tol=1e-5, 
                 e_tol=None, max_ndets=None):
    '''
    Select based on overlap and energy. 
    '''
    print("#Selecting determinants based on overlap and energy contribution.")
    hmat_fix, smat_fix = slater.noci_matrices(rmats_fix, mo_coeff, h1e, h2e)
    if e_tol is None:
        print("# Only using metric criterion.")
        print("# Metric Threshold: {:.1e}".format(m_tol))
    else:
        print("# Using both metric and hamiltonian criteria.")
        print("# Metric Threshold: {:.1e}".format(m_tol))
        print("# Energy Threshold: {:.1e}".format(e_tol))
        E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    n_new = len(rmats_new)
    n_fix = len(rmats_fix)
    n_ref = n_fix

    if max_ndets is None:
        max_ndets = n_new
        
    count = 0
    count_m = 0 # count the number that passed m_tol
    for i in range(n_new):
        if count == max_ndets:
            print("# Warning: maximum number of determinant exceeded!")
            break
        r_new = rmats_new[i]
        # first compute necessary values
        metrics_mix = np.einsum('nsji, sjk -> nsik', rmats_fix.conj(), r_new)
        inv_fix = np.linalg.inv(smat_fix)
        if e_tol is None:
            smat_left = np.prod(np.linalg.det(metrics_mix), axis=-1)
            s_new = np.einsum("sji, sjk -> sik", r_new.conj(), r_new)
            s_new = np.prod(np.linalg.det(s_new), axis=-1)
        else:
            hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, r_new[None, :], rmats_fix, h1e, h2e, mo_coeff)
            smat_left = smat_all[:-1, -1]
            s_new = smat_all[-1, -1] 

        proj_old = smat_left.T.conj() @ inv_fix @ smat_left
        proj_new = 1.0 - proj_old/s_new

        if proj_new > m_tol:
            if e_tol is None: # only consider overlap
                rmats_fix = np.vstack([rmats_fix, r_new[None, :]])
                n_fix += 1
                smat_all = np.zeros((n_fix, n_fix))
                smat_all[:-1, :-1] = smat_fix
                smat_all[:-1, -1]  = smat_left
                smat_all[-1, :-1]  = smat_left.conj().T
                smat_all[-1, -1]   = s_new
                smat_fix = smat_all
                count += 1
            else: # consider hamitonian criterion 
                count_m += 1
                norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
                H_fix = E_fix * norm_fix 
                b_p = inv_fix @ smat_left
                hmat_left = hmat_all[:-1, -1]
                T = noci_vec.conj() @ (hmat_left - hmat_fix @ b_p)
                H_new = hmat_all[-1, -1] - 2*np.real(b_p.conj().T @ hmat_left) 
                H_new = H_new + b_p.conj().T @ hmat_fix @ b_p
                norm_new = proj_new * s_new
                H_22 = np.zeros((2, 2))
                S_22 = np.zeros((2, 2))
                H_22[0, 0] = H_fix
                H_22[0, 1] = T
                H_22[1, 0] = T.conj()
                H_22[1, 1] = H_new
                S_22[0, 0] = norm_fix
                S_22[1, 1] = norm_new 
                epsilon, vec = slater.solve_lc_coeffs(H_22, S_22, return_vec=True)
                ratio = (E_fix - epsilon) / abs(E_fix)

                if ratio > e_tol:
                    rmats_fix = np.vstack([rmats_fix, r_new[None, :]])
                    n_fix += 1
                    smat_fix = smat_all 
                    hmat_fix = hmat_all
                    E_fix = epsilon 
                    c = np.zeros(n_fix)
                    c[:-1] = vec[0] * noci_vec - vec[1] * b_p
                    c[-1] = vec[1]
                    noci_vec = c
                    count += 1
        else:
            count += 0
        gc.collect()

    # num_added = len(selected_indices)
    print("###### Selected CI Summary ######")
    if e_tol is None:
        print("**Metric threshold**: Reduced {} determinants to {} determinants.".format(n_new, count))
        print("Total number of determinants after Metric threshold: ", n_ref + count)
    else:
        print("**Metric threshold**: Reduced {} determinants to {} determinants.".format(n_new, count_m))
        print("**Metric + Energy threshold**: Reduced {} determinants to {} determinants.".format(n_new, count))
        print("Total number of determinants after Metric threshold: ", n_ref + count_m)
        print("Total number of determinants after Metric and Energy threshold: ", n_ref + count)
    return rmats_fix

def select_rmats_slow(rmats_fix, rmats_new, mo_coeff, h1e, h2e, m_tol=1e-5, 
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
        m, s = _criterial_ovlp_single_det(rmats_fix, r_new[0], smat_fix=smat_fix, m_tol=m_tol)
        # print(m)
        if m > m_tol:
            smat_fix = s
            rmats_fix = np.vstack([rmats_fix, r_new])
            idx_select.append(i)
            count += 1
        else:
            count += 0
        gc.collect()

    # num_added = len(selected_indices)
    print("# Reduced {} determinants to {} determinants.".format(n_new, count))
    if return_indices:
        return rmats_fix, idx_select
    else:
        return rmats_fix

def select_rmats_energy(rmats_fix, rmats_new, mo_coeff, h1e, h2e, e_tol=None, max_ndets=None):
    '''
    Select based energy. 
    '''
    print("# Selecting determinants based on energy contribution.")
    print("# Energy Threshold: {:.1e}".format(e_tol))
    hmat_fix, smat_fix = slater.noci_matrices(rmats_fix, mo_coeff, h1e, h2e)
    E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    n_new = len(rmats_new)
    n_fix = len(rmats_fix)
    n_ref = n_fix

    if max_ndets is None:
        max_ndets = n_new
        
    count = 0
    for i in range(n_new):
        if count == max_ndets:
            print("# Warning: maximum number of determinant exceeded!")
            break
        r_new = rmats_new[i]
        # first compute necessary values
        inv_fix = np.linalg.inv(smat_fix)
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, r_new[None, :], rmats_fix, h1e, h2e, mo_coeff)
        smat_left = smat_all[:-1, -1]
        s_new = smat_all[-1, -1] 
        proj_old = smat_left.T.conj() @ inv_fix @ smat_left
        norm_fix = np.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
        H_fix = E_fix * norm_fix 
        b_p = inv_fix @ smat_left
        hmat_left = hmat_all[:-1, -1]
        T = noci_vec.conj() @ (hmat_left - hmat_fix @ b_p)
        H_new = hmat_all[-1, -1] - 2*np.real(b_p.conj().T @ hmat_left) 
        H_new = H_new + b_p.conj().T @ hmat_fix @ b_p
        norm_new = s_new - proj_old # proj_new * s_new
        H_22 = np.zeros((2, 2))
        S_22 = np.zeros((2, 2))
        H_22[0, 0] = H_fix
        H_22[0, 1] = T
        H_22[1, 0] = T.conj()
        H_22[1, 1] = H_new
        S_22[0, 0] = norm_fix
        S_22[1, 1] = norm_new 
        epsilon, vec = slater.solve_lc_coeffs(H_22, S_22, return_vec=True)
        ratio = (E_fix - epsilon) / abs(E_fix)

        if ratio > e_tol:
            rmats_fix = np.vstack([rmats_fix, r_new[None, :]])
            n_fix += 1
            smat_fix = smat_all 
            hmat_fix = hmat_all
            E_fix = epsilon 
            c = np.zeros(n_fix)
            c[:-1] = vec[0] * noci_vec - vec[1] * b_p
            c[-1] = vec[1]
            noci_vec = c
            count += 1
        else:
            continue
    # num_added = len(selected_indices)
    print("###### Selected CI Summary ######")
    print("**Energy threshold**: Reduced {} determinants to {} determinants.".format(n_new, count))
    print("Total number of determinants after Energy threshold: ", n_ref + count)
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


def _criterial_ovlp_single_det(rmats_fix, r_new, smat_fix=None, m_tol=1e-5):
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


def _criteria_all_single_det(rmats_fix, r_new, mo_coeff, h1e, h2e, E_fix=None, 
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