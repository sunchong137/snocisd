'''
Spin symmetry projected HF.
Reference: C. Jimenez-Hoyos et.al, JCP, 136, 164109 (2012)
'''
import numpy as np 
from scipy.special import factorial 

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
        wig = wignerd(roots[i], j, m, m)
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