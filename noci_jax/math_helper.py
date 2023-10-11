import numpy as np

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