import numpy as np

def generalized_eigh(A, B):
    L = np.linalg.cholesky(B)
    L_inv = np.linalg.inv(L)
    A_redo = L_inv.dot(A).dot(L_inv.T)
    e, v = np.linalg.eigh(A_redo)
    e0 = e[0]
    v0 = v[:, 0]
    c0 = L_inv.T.dot(v0) # rotate back 

    return e0, c0