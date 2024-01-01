'''
Spin symmetry projected HF.
Reference: C. Jimenez-Hoyos et.al, JCP, 136, 164109 (2012)
'''
import numpy as np 

def gen_roots_weights(ngrid, j, m):
    '''
    Generate the roots and weights for the integration.
    Args:
        ngrid: int, number of grids.
        j: int, quantum number associated with S^2.
        m: int, quantum number associated with S_z.
    Returns:
        roots: 1D array of size ngrid.
        weights: 1D array of size ngrid.
    '''
    # roots = np.zeros(ngrid)
    weights = np.zeros(ngrid)
    for i in range(ngrid-1):
        coeff = (1-1/(i+2))/((2-1/(i+1))*(2-1/(i+2)))
        weights[i] = np.sqrt(coeff)
    mat = np.zeros((ngrid, ngrid))
    for i in range(ngrid-1):
        mat[i, i+1] = mat[i+1, i] = weights[i]
    vals, vecs = np.linalg.eigh(mat)
    for i in range(ngrid):
        weights[i] = 2 * vecs[0, i] ** 2    
        roots = (vals + 1) * np.pi/2
    weights = weights * np.pi/2
    for i in range(ngrid):
        wig = wignerd(roots[i], j, m, m, 1)
        weights[i] = weights[i] * np.sin(roots[i]) * wig
    return roots, weights
	

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
    pass

# def wignerd(beta, j, m, n, int_coeffs=True):
#     """
#     Evaluate matrix elements of Wigner's small d-matrix, given by
#         d^j_{m, n} (beta) = <j; m | exp(-i*beta*S_y) |j; n>,
#             S_y - Pauli operator of the y direction
#             beta - angle
#             |j; n> - eigenstate of S^2 and S_z with eigenvalues j*(j+1) and n, respectively.
#             j > 0, -j <= m, n <= j,
#         Note: d^j_{m, n} (beta) = d^j_{-m, -n} (beta) 
#     Args:
#         beta: float, angle.
#         j: int, eigenvalue of S^2,
#         m: int, eigenvalue of S_z in the bra state.
#         n: int, eigenvalue of S_z in the ket state.
#     Kwargs:
#         int_coeffs: Bool, if True, coefficients are integers. 
#                     If False: coefficeints are half-integers and (j, m, n) are all odd numbers. 
#     Returns:
#         float, the (m, n)th element of the Wigner small d-matrix.
#     """
#     assert j >= 0, "Unpysical value of j provided!"
#     assert abs(m) <= j and abs(n) <= j, "Unphysical m and n value provided!"
#     if not int_coeffs:
#         assert j%2 == 0 and m%2 == 0 and n%2 == 0, "int_coeff is False - j, m, n should all be odd!"
#         assert j < 5, "j >= 2 are not yet implemented."
#     else:
#         assert j < 3, "j >= 2 are not yet implemented."

#     # Define working variables.
#     jwrk = j
#     sgnf = 1
#     if (m >= 0):
#         # If m >= 0, then we already have one of the conditions met.
#         # We still require that m >= n.
#         if (m >= abs(n)): # m>abs(n)
#             mwrk = m
#             nwrk = n
#         elif (m < n): # n>m>=0, d^j_{m,n} = (-1)^(m-n) * d^j_{n,m}
#             mwrk = n
#             nwrk = m
#             ltst = int((int_coeffs == 1 and abs(n-m)%2 == 1) or
#                        (int_coeffs == 0 and (abs(n-m)//2)%2 == 1))
#             if (ltst == 1):
#                 sgnf = -1
#         elif (m > n): # m>=0>n, d^j_{m,n} = d^j_{-n,-m}
#             mwrk = -n
#             nwrk = -m
#     else: # m < 0
#         if (-m >= abs(n)): # d^j_{m,n} = (-1)^(m-n) * d^j_{-m,-n}
#             mwrk = -m
#             nwrk = -n
#             ltst = int((int_coeffs == 1 and abs(n-m)%2 == 1) or
#                        (int_coeffs == 0 and (abs(n-m)//2)%2 == 1))
#             if (ltst == 1):
#                 sgnf = -1
#         elif (-m < n): # m<0<n, use d^j_{m,n} = (-1)^(m-n) * d^j_{n,m} 
#             # NOTE in Tom's implementation it's (m < n)
#             mwrk = n
#             nwrk = m
#             ltst = int((int_coeffs == 1 and abs(n-m)%2 == 1) or
#                        (int_coeffs == 0 and (abs(n-m)//2)%2 == 1))
#             if (ltst == 1):
#                 sgnf = -1
#         elif (-m > n): # n<m<0, d^j_{m,n} = d^j_{-n,-m}
#             mwrk = -n
#             nwrk = -m
#     # evaluate matrix elements
#     z = 1
#     """
#     case j = 1/2
#       d (1/2, 1/2,  1/2)  =  cos(beta/2)
#       d (1/2, 1/2, -1/2)  = -sin(beta/2)
#     """
#     if (jwrk == 1 and int_coeffs == 0):
#         if (nwrk == 1):
#             z = np.cos(beta/2)
#         elif (nwrk == -1):
#             z = -np.sin(beta/2)

#     """
#     case j = 1
#       d (1, 1,  1)  =  1/2 * (1 + cos(beta))
#       d (1, 1,  0)  = -sin(beta) / sqrt(2)
#       d (1, 1, -1)  =  1/2 * (1 - cos(beta))
#       d (1, 0,  0)  =  cos(beta)
#     """
#     if (jwrk == 1 and int_coeffs == 1):
#         if (mwrk == 1):
#             if (nwrk == 1):
#                 z = 1/2 * (1 + np.cos(beta))
#             elif (nwrk == 0):
#                 z = -np.sin(beta) / np.sqrt(2)
#             elif (nwrk == -1):
#                 z = 1/2 * (1 - np.cos(beta))

#         elif (mwrk == 0):
#             z = np.cos(beta)

#     """
#     case j = 3/2

#       d (3/2, 3/2,  3/2)  =  1/2 * (1 + cos(beta)) * cos(beta/2)
#       d (3/2, 3/2,  1/2)  = -sqrt(3)/2 * (1 + cos(beta)) * sin(beta/2)
#       d (3/2, 3/2, -1/2)  =  sqrt(3)/2 * (1 - cos(beta)) * cos(beta/2)
#       d (3/2, 3/2, -3/2)  = -1/2 * (1 - cos(beta)) * sin(beta/2)
#       d (3/2, 1/2,  1/2)  =  1/2 * (3*cos(beta) - 1) * cos(beta/2)
#       d (3/2, 1/2, -1/2)  = -1/2 * (3*cos(beta) + 1) * sin(beta/2)
#     """

#     if (jwrk == 3 and int_coeffs == 0):
#         if (mwrk == 3):
#             if (nwrk == 3):
#                 z = 1/2 * (1 + np.cos(beta)) * np.cos(beta/2)
#             elif (nwrk == 1):
#                 z = -np.sqrt(3)/2 * (1 + np.cos(beta)) * np.sin(beta/2)
#             elif (nwrk == -1):
#                 z = np.sqrt(3)/2 * (1 - np.cos(beta)) * np.cos(beta/2)
#             elif (nwrk == -3):
#                 z = -1/2 * (1 - np.cos(beta)) * np.sin(beta/2)

#         elif (mwrk == 1):
#             if (nwrk == 1):
#                 z = 1/2 * (3*np.cos(beta) - 1) * np.cos(beta/2)
#             elif (nwrk == -1):
#                 z = -1/2 * (3*np.cos(beta) + 1) * np.sin(beta/2)

#     """
#     case j = 2

#       d (2, 2,  2)  =  ((1 + cos(beta)) / 2)^2
#       d (2, 2,  1)  = -1/2 * (1 + cos(beta)) * sin(beta)
#       d (2, 2,  0)  =  sqrt(6)/4 * sin(beta)^2
#       d (2, 2, -1)  = -1/2 * (1 - cos(beta)) * sin(beta)
#       d (2, 2, -2)  =  ((1 - cos(beta)) / 2)^2
#       d (2, 1,  1)  =  1/2 * (1 + cos(beta)) * (2*cos(beta) - 1)
#       d (2, 1,  0)  = -sqrt(3/2) * sin(beta) * cos(beta)
#       d (2, 1, -1)  =  1/2 * (1 - cos(beta)) * (2*cos(beta) + 1)
#       d (2, 0,  0)  =  1/2 * (3*(cos(beta))^2 - 1)
#     """

#     if (jwrk == 2 and int_coeffs == 1):
#         if (mwrk == 2):
#             if (nwrk == 2):
#                 z = np.linalg.matrix_power((1/2 * (1 + np.cos(beta))), 2)
#             elif (nwrk == 1):
#                 z = -1/2 * (1 + np.cos(beta)) * np.sin(beta)
#             elif (nwrk == 0):
#                 z = np.sqrt(6)/4 * \
#                     np.linalg.matrix_power((np.sin(beta)), 2)
#             elif (nwrk == -1):
#                 z = -1/2 * (1 - np.cos(beta)) * np.sin(beta)
#             elif (nwrk == -2):
#                 z = np.linalg.matrix_power((1/2 * (1 - np.cos(beta))), 2)

#         elif (mwrk == 1):
#             if (nwrk == 1):
#                 z = 1/2 * (1 + np.cos(beta)) * (2*np.cos(beta) - 1)
#             elif (nwrk == 0):
#                 z = -np.sqrt(3/2) * np.sin(beta) * np.cos(beta)
#             elif (nwrk == -1):
#                 z = 1/2 * (1 - np.cos(beta)) * (2*np.cos(beta) + 1)

#         elif (mwrk == 0):
#             z = 1/2 * (3*np.linalg.matrix_power((np.cos(beta)), 2) - 1)

#     # Account for the factor of (-1)^(m-n) if needed...

#     z = sgnf*z
#     return z


if __name__ == "__main__":
    roots, weights = gen_roots_weights(4, 3, 1)
    print(roots)