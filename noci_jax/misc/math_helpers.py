import numpy as np


def generalized_eigh(H, S):
    '''
    Solve:
        Hv = eSv
    '''
    try:
        L = np.linalg.cholesky(S)
        L_inv = np.linalg.inv(L)
        H_redo = L_inv @ H @ L_inv.T
        e, v = np.linalg.eigh(H_redo)
        e0 = e[0]
        v0 = v[:, 0]
        c0 = L_inv.T @ v0 # rotate back 
    except: # singularity on B 
        print("Warning: Singluarity in overlap!")
        e0, c0 = generalized_eigh_singular(H, S)
    return e0, c0

def generalized_eigh_singular(H, S):
    '''
    S is singular.
    '''
    ew, ev = np.linalg.eigh(S)
    idx = ew > 1e-15
    rot = (ev[:, idx] / np.sqrt(ew[idx])) @ ev[:, idx].conj().T 
    H_redo = rot @ H @ rot 
    e, v = np.linalg.eigh(H_redo) 
    e0 = e[0]
    v0 = v[:, 0]
    c0 = rot @ v0 

    return e0, c0