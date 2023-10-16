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

def metric_residual(rmats0, r_new, ovlp_mat0=None):
    '''
    Check if a new vector is not linearly dependent to the existing NOCI pool.
    '''
    nr0 = len(rmats0)
    nr = nr0 + len(r_new)
    if ovlp_mat0 is None:
        r_all = jnp.vstack([rmats0, r_new])
        metrics_all = jnp.einsum('nsji, msjk -> nmsik', r_all.conj(), r_all)
        ovlp_all = jnp.prod(jnp.linalg.det(metrics_all), axis=-1)
        ovlp_left = ovlp_all[:nr0, nr0:]
        ovlp_new = ovlp_all[nr0:, nr0:]
        ovlp_mat0 = ovlp_all[:nr0, :nr0]
    else:
        ovlp_all = jnp.zeros((nr, nr))
        ovlp_all = ovlp_all.at[:nr0, :nr0].set(ovlp_mat0) 
        metrics_mix = jnp.einsum('nsji, msjk -> nmsik', rmats0.conj(), r_new)
        ovlp_left = jnp.prod(jnp.linalg.det(metrics_mix), axis=-1)
        ovlp_all = ovlp_all.at[:nr0, nr0:].set(ovlp_left)
        ovlp_all = ovlp_all.at[nr0:, :nr0].set(ovlp_left.T.conj())
        metrics_new = jnp.einsum('nsji, msjk -> nmsik', r_new.conj(), r_new)
        ovlp_new = jnp.prod(jnp.linalg.det(metrics_new), axis=-1)
        ovlp_all = ovlp_all.at[nr0:, nr0:].set(ovlp_new) 

    # calculate the residual 
    inv0 = jnp.linalg.inv(ovlp_mat0)
    proj_nto = jnp.einsum("np, pq, qn -> n", ovlp_left.T.conj(), inv0, ovlp_left)
    resid_n = 1 - proj_nto/jnp.diag(ovlp_new)

    return resid_n

def snoci_criteria(rmats0, r_new, mo_coeff, h1e, h2e, E_fix=None, noci_vec=None, ovlp_mat0=None, ham_mat0=None):
    '''
    Evaluate the linear dependency and energy contribution of the new vectors.
    '''

    nr0 = len(rmats0)
    # nr = nr0 + len(r_new)

    if ovlp_mat0 is None or ham_mat0 is None:
        r_all = jnp.vstack([rmats0, r_new])
        ham_all, ovlp_all = slater.noci_matrices(r_all, mo_coeff, h1e, h2e)
        ovlp_mat0 = ovlp_all[:nr0, :nr0]

    else:
        ham_all, ovlp_all = slater.expand_hs(ham_mat0, ovlp_mat0, r_new, rmats0, h1e, h2e, mo_coeff)
    
    # calculate the residual 
    ovlp_left = ovlp_all[:nr0, nr0:]
    ovlp_new = ovlp_all[nr0:, nr0:]
    inv0 = jnp.linalg.inv(ovlp_mat0)
    proj_nto = jnp.einsum("np, pq, qn -> n", ovlp_left.T.conj(), inv0, ovlp_left)
    resid_n = 1 - proj_nto/jnp.diag(ovlp_new) 

    # calculate the energy contribution

    alpha = inv0 @ ovlp_left  # (nr0, nr) array 
    ham_left = ham_all[:nr0, nr0:]
    ham_mat0 = ham_all[:nr0, :nr0]
    ham_new = ham_all[nr0:, nr0:]
    if E_fix is None:
        E_fix, noci_vec = slater.solve_lc_coeffs(ham_mat0, ovlp_mat0, return_vec=True)

    norm_fix = jnp.einsum("i, ij, j ->", noci_vec.conj(), ovlp_mat0, noci_vec)
    H_fix = E_fix * norm_fix
    T_part = noci_vec.T.conj() @ (ham_left - ham_mat0 @ alpha)
    H_new = jnp.diag(ham_new) - 2 * jnp.real(jnp.einsum("pn, pn -> n", alpha.conj(), ham_left))
    H_new = H_new + jnp.einsum("pn, pq, qn -> n", alpha.conj(), ham_mat0, alpha)
    E_new = H_new / resid_n
    R_term = np.sqrt((H_new * norm_fix - H_fix * resid_n)**2 + 4 * resid_n * norm_fix * jnp.abs(T_part)**2) 
    resid_energy = (E_new - E_fix - R_term/(resid_n * norm_fix)) / (2 * E_fix)

    return resid_n, resid_energy
