# Copyright 2023-2024 NOCI_Jax developers. All Rights Reserved.
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
Given a spin Hamiltonian, perform spinless Hartree-Fock simulation 
on the JW-transformed Hamiltonian.
'''
import numpy as np

def jw_ham(M, V, w=0, no_tol=1e-8):
    '''
    Given a spin Hamiltonian, transfer it into two parts: with string and without string.
    The spin Hamiltonian is defined as:
    ```math
    H = \sum_{pq} M_{pq} S^+_p S^-_q + \sum_{pq} V_{pq} Z_p Z_q + \sum_p w_p Z_p,
    ```
    The fermionic Hamiltonian from JW transformation is 
    ```math
    Hf = \sum_{|p-q|>1} M_{pq} f^+_p f_q str_{pq} + \sum_{|p-q|<1} M_{pq} f^+_p f_q 
         \sum_{pq} V_{pq} n_p n_q + \sum_p w'_p n_p + e'_0,
    ```
    Args:
        M: (L, L) array, symmetric, coefs for S^+_p S_-.
        V: (L, L) array, symmetric, coeffs for Z_p Z_q.
        w: (L, ) array, coeffs of Z_p
    Returns:
        h1o: (L, L) array, the one-body Hamiltonian without string.
        g2o: (L, L) array, the two-body Hamiltonian, only acts on n_p n_q
        e0: the constant term.
        h1no: 0 or 1D array of size (L-1)(L-2)/2, flattened the top triangle 
            after removing tridiagonal terms.
    '''
    L = M.shape[-1]
    # initialize
    h1o = np.zeros((L, L)) 
    # get orthogonal Hamiltonian 
    v = np.sum(V, axis=1)
    np.fill_diagonal(h1o, np.diag(M) + w - v)
    for i in range(L - 1):
        h1o[i, i+1] = M[i, i+1] 
        h1o[i+1, i] = M[i+1, i]
    g2o = V
    e0 = np.sum(V)/4 - np.sum(w)/2
    # get non-orthogonal part 
    tri_ind = np.triu_indices(L, k=2, m=L)
    h1no = M[tri_ind] 
    val_h1 = np.linalg.norm(h1no)
    if val_h1 < no_tol:
        h1no = 0
    return h1o, g2o, e0, h1no

def eval_energy(det, h1o, g2o, h1no):
    '''
    Calculate the expectation value of a Fermionic Hamiltonian with strings.
    Parameters:
        det (np.ndarray): Molecular orbital (MO) coefficients of the occupied orbitals,
                          with the shape (L, N_occ), where L is the total number of orbitals,
                          and N_occ is the number of occupied orbitals.
        M (np.ndarray): Two-body integrals for the S^+ S^- terms, with shape (L, L).
        V (np.ndarray): Two-body integrals for the ZZ terms, with shape (L, L).
        w (np.ndarray, optional): One-body integrals for the Z terms, with shape (L,).
                                  Defaults to an array of zeros if not provided.
        e0 (float, optional): Scalar energy shift. Defaults to 0.

    Returns:
        float: The expectation value <Phi|H|Phi> of the Hamiltonian for the given
               Fermionic Slater determinant.
    '''

    norb = det.shape[0]

    
    return 0
    