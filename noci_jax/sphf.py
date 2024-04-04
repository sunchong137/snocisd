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
Spin symmetry projected HF.
Reference: C. Jimenez-Hoyos et.al, JCP, 136, 164109 (2012)
'''
import numpy as np 
from scipy.special import factorial 

def gen_transmat_sphf(mo_coeffs, ngrid=2, from_roots=False):
    '''
    Generate transition matrices of the SPHF mo_coeffs from the UHF mo_coeffs.
    Args:
        mo_coeffs: (2, norb, norb) array, MO coefficients.
        ngrid: number of grids to generate
    Returns:
        (ngrid, 2, norb, norb) array
    '''
    norb = mo_coeffs.shape[-1]
    if from_roots:
        betas = gen_roots(ngrid)
    else: # uniform distribution
        betas = np.linspace(0, np.pi, ngrid+1, endpoint=True)[1:]
    Ca = mo_coeffs[0]
    Cb = mo_coeffs[1]
    Ca_inv = np.linalg.inv(Ca)
    Cb_inv = np.linalg.inv(Cb)
    U_all = []
    for b in betas:
        R = gen_rotations_ao(b, norb) # 2N x 2N 
        Ca_n = R[:norb, :norb] @ Ca + R[:norb, norb:] @ Cb 
        Cb_n = R[norb:, :norb] @ Ca + R[norb:, norb:] @ Cb
        Ua = Ca_inv @ Ca_n 
        Ub = Cb_inv @ Cb_n 
        U = np.array([Ua, Ub])
        U_all.append(U)
    
    U_all = np.asarray(U_all)
    return U_all

def gen_rotations_ao(beta, norb):
    '''
    Generate the GHF rotation matrices in the AO basis.
    Args:
        beta: float, angle.
        norb: int, number of spatial orbitals.
    Returns:
        Array of (norbx2, norbx2) or a list of arrays of size (norbx2, norbx2)
    NOTE: when beta = Pi, the rotation matrix flips the two spins. 
    '''
    try:
        lb = len(beta)
        Rs = np.zeros((lb, norb*2, norb*2))
        for i in range(lb):
            R = np.eye(norb*2) * np.cos(beta[i]/2)
            IS = np.eye(norb) * np.cos(beta[i]/2)
            R[:norb, norb:] = IS
            R[norb:, :norb] = - IS
            Rs[i] = R

    except:
        Rs = np.eye(norb*2) * np.cos(beta/2)
        IS = np.eye(norb) * np.sin(beta/2)
        Rs[:norb, norb:] = IS
        Rs[norb:, :norb] = - IS
    return Rs

def gen_roots_weights(ngrid, j, m):
    '''
    Generate the roots and weights for the integration. 
    The roots and weights generated in this way make the integration less costy.
    Reference: Golub, G.H. and Welsch, J.H. (1969) Calculation of Gauss Quadrature Rules. Mathematics of Computation, 23, 221-230. 

    Args:
        ngrid: int, number of grids.
        j: int, quantum number associated with S^2.
        m: int, quantum number associated with S_z.
    Returns:
        roots: 1D array of size ngrid.
        weights: 1D array of size ngrid.
    '''
    coeff = np.arange(1, ngrid)
    coeff = (1 - 1/(coeff+1)) / ((2 - 1/coeff) * (2 - 1/(coeff+1)))
    mat = np.zeros((ngrid, ngrid))
    for i in range(ngrid-1):
        mat[i, i+1] = mat[i+1, i] = np.sqrt(coeff[i]) 
    vals, vecs = np.linalg.eigh(mat)
    weights = np.pi * vecs[0] ** 2  
    roots = (vals + 1) * np.pi / 2
    wig =  wignerd(roots, j, m, m)
    weights *= np.sin(roots) * wig
    return roots, weights

def gen_roots(ngrid):
    '''
    Only generate roots.
    '''
    coeff = np.arange(1, ngrid)
    coeff = (1 - 1/(coeff+1)) / ((2 - 1/coeff) * (2 - 1/(coeff+1)))
    mat = np.zeros((ngrid, ngrid))
    for i in range(ngrid-1):
        mat[i, i+1] = mat[i+1, i] = np.sqrt(coeff[i]) 
    vals, vecs = np.linalg.eigh(mat)
    roots = (vals + 1) * np.pi / 2
    return roots
	
def wignerd(beta, j, m, n):
    '''
    Evaluate matrix elements of Wigner's small d-matrix, given by
         d^j_{m, n} (beta) = <j; m | exp(-i*beta*S_y) |j; n>,
    Using Wikipedia definition: search Wigner (small) d-matrix
    Args:
        beta: float, angle.
        j: int, eigenvalue of S^2,
        m: int, eigenvalue of S_z in the bra state.
        n: int, eigenvalue of S_z in the ket state.
    Returns:
        float, the (m, n)th element of the Wigner small d-matrix.
    '''
    assert j >= 0, "Unpysical value of j provided!"
    assert abs(m) <= j and abs(n) <= j, "Unphysical m and n value provided!"
    ang = beta/2.
    smin = max(0, int(n-m+1e-16))
    smax = min(int(j+n+1e-16), int(j-m+1e-16))
    dval = 0
    coeff = np.sqrt(factorial(j+m) * factorial(j-m) * factorial(j+n) * factorial(j-n))
    for s in range(smin, smax+1):
        upper = (-1)**(m-n+s) * (np.cos(ang))**(2*j+n-m-2*s) * (np.sin(ang))**(m-n+2*s)
        lower = factorial(j+n-s) * factorial(s) * factorial(m-n+s) * factorial(j-m-s) 
        dval += upper / lower 
    return dval * coeff


if __name__ == "__main__":
    roots, weights = gen_roots_weights(4, 3, 1)
    print(roots)