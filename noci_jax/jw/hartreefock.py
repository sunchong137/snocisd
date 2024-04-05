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


def eval_energy(det, M, V, w=0, e0=0):
    '''
    Calculate the expectation value of a given spin Hamiltonian H for a Fermionic 
    Slater determinant |Phi>. We first use JW transformation to convert the spin 
    Hamiltonian into a fermionic Hamiltonian, and then evaluate  <Phi|H|Phi>.

    The spin Hamiltonian is defined as:
    ```math
    H = \sum_{pq} M_{pq} S^+_p S^-_q + \sum_{pq} V_{pq} Z_p Z_q + \sum_p w_p Z_p + e0,
    ```
    where `S^+_p` and `S^-_q` are the spin raising and lowering operators, respectively,
    and `Z_p` and `Z_q` are the Pauli-Z operators.

    Following the Jordan-Wigner transformation, the fermionic Hamiltonian (Hf) takes the form:
    ```math
    Hf = \sum_{p<q}(M_{pq}f^+_p f_q - M_{qp}f^+_q f_p) + \sum_{pq} V_{pq} n_p n_q 
         + \sum_p w'_p n_p + e'_0,
    ```
    where `f^+_p`, `f_q` are fermionic creation and annihilation operators, respectively, 
    and `n_p` is the number operator. The `str(p, q)` is the string from JW transformation.

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
    # get orthogonal part 
    
    # get non-orthogonal part 
    
    return 0
    