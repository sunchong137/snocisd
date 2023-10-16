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
from jax import numpy as jnp
from noci_jax import slater
import jax
from jax.config import config
config.update("jax_enable_x64", True)

def check_linear_depend(ovlp_mat, tol=1e-10):
    '''
    Check the linear dependancy of a set of vectors.
    Args:
        ovlp_mat: 2D array, overlap matrix
    Kwargs:
        tol: zero tolerace
    Returns:
        Dimension of the space spanned by the vectors, i.e., number of linearly-independent vectors.
    '''
    _, s, _ = np.linalg.svd(ovlp_mat)
    num_ind = np.sum(s > tol) # S is positive semi definite
    
    return num_ind

def metric_residual(rmats_fix, rmats_new, smat_fix=None):
    '''
    Check if a new vector is not linearly dependent to the existing NOCI pool.
    '''
    nr0 = len(rmats_fix)
    nr = nr0 + len(rmats_new)

    if smat_fix is None:
        rmats_all = jnp.vstack([rmats_fix, rmats_new])
        metrics_all = jnp.einsum('nsji, msjk -> nmsik', rmats_all.conj(), rmats_all)
        smat_all = jnp.prod(jnp.linalg.det(metrics_all), axis=-1)
        smat_mix_l = smat_all[:nr0, nr0:]
        smat_new = smat_all[nr0:, nr0:]
        smat_fix = smat_all[:nr0, :nr0]
    else:
        smat_all = jnp.zeros((nr, nr))
        smat_all = smat_all.at[:nr0, :nr0].set(smat_fix) 
        metrics_mix = jnp.einsum('nsji, msjk -> nmsik', rmats_fix.conj(), rmats_new)
        smat_mix_l = jnp.prod(jnp.linalg.det(metrics_mix), axis=-1)
        smat_all = smat_all.at[:nr0, nr0:].set(smat_mix_l)
        smat_all = smat_all.at[nr0:, :nr0].set(smat_mix_l.T.conj())
        metrics_new = jnp.einsum('nsji, msjk -> nmsik', rmats_new.conj(), rmats_new)
        smat_new = jnp.prod(jnp.linalg.det(metrics_new), axis=-1)
        smat_all = smat_all.at[nr0:, nr0:].set(smat_new) 

    # calculate the residual 
    inv_fix = jnp.linalg.inv(smat_fix)
    proj_old = jnp.einsum("np, pq, qn -> n", smat_mix_l.T.conj(), inv_fix, smat_mix_l)
    norm_new_proj = 1.0 - proj_old/jnp.diag(smat_new)

    return norm_new_proj

def snoci_criteria(rmats_fix, rmats_new, mo_coeff, h1e, h2e, E_fix=None, noci_vec=None, smat_fix=None, hmat_fix=None):
    '''
    Evaluate the linear dependency and energy contribution of the new vectors.
    '''

    nr0 = len(rmats_fix)

    if smat_fix is None or hmat_fix is None:
        rmats_all = jnp.vstack([rmats_fix, rmats_new])
        hmat_all, smat_all = slater.noci_matrices(rmats_all, mo_coeff, h1e, h2e)
        smat_fix = smat_all[:nr0, :nr0]
        hmat_fix = hmat_all[:nr0, :nr0]
    else:
        hmat_all, smat_all = slater.expand_hs(hmat_fix, smat_fix, rmats_new, rmats_fix, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    smat_mix_l = smat_all[:nr0, nr0:]
    smat_new = smat_all[nr0:, nr0:]
    inv_fix = jnp.linalg.inv(smat_fix)
    proj_old = jnp.einsum("np, pq, qn -> n", smat_mix_l.T.conj(), inv_fix, smat_mix_l)
    norm_new = jnp.diag(smat_new)
    proj_new = 1.0 - proj_old/norm_new
    sdiag_new = norm_new - proj_old
    # calculate the energy contribution
    if E_fix is None:
        E_fix, noci_vec = slater.solve_lc_coeffs(hmat_fix, smat_fix, return_vec=True)

    alpha = inv_fix @ smat_mix_l  # (nr0, nr) array 
    hmat_mix_l = hmat_all[:nr0, nr0:]
    hmat_new = hmat_all[nr0:, nr0:]
    norm_fix = jnp.einsum("i, ij, j ->", noci_vec.conj(), smat_fix, noci_vec)
    H_fix = E_fix * norm_fix
    T_part = noci_vec.T.conj() @ (hmat_mix_l - hmat_fix @ alpha)
    H_new = jnp.diag(hmat_new) - 2 * jnp.real(jnp.einsum("pn, pn -> n", alpha.conj(), hmat_mix_l))
    H_new = H_new + jnp.einsum("pn, pq, qn -> n", alpha.conj(), hmat_fix, alpha)
    E_new = H_new / sdiag_new
    R_term = np.sqrt((H_new*norm_fix - H_fix*sdiag_new)**2 + 4*sdiag_new*norm_fix*jnp.abs(T_part)**2) 
    de_ratio = (E_new - E_fix - R_term/(sdiag_new * norm_fix)) / (2 * E_fix)

    return proj_new, de_ratio