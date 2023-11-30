'''
Bethe Ansatz for 1D Hubbard model.
Adapted from the Mathematical code by Carlos Jimenez-Hoyos.
Following Lieb&Wu's paper: https://arxiv.org/abs/cond-mat/0207529
'''
import numpy as np
from scipy.optimize import root as find_root
Pi = np.pi

# Construct arrays
def gen_grids(L, Nup, Ndn):
    '''
    Construct grid points needed in the integration.
    '''
    Ne = Nup + Ndn
    assert Ne <= L * 2
    # parity
    p_up = Nup % 2
    p_dn = Ndn % 2
    p_ne = Ne % 2
    # Iarr
    if (p_up + p_ne) % 2 == 0: # same parity
        Iarr = np.arange(-Ne/2, Ne/2, 1)
    else:
        Iarr = np.arange(-(Ne-1)/2, (Ne+1)/2, 1)

    # Jarr 
    if (p_up + p_dn) % 2 == 0:
        Jarr = np.arange(-(Nup-1)/2, (Nup+1)/2, 1)
    else:
        Jarr = np.arange(-Nup/2-1, Nup/2+1, 1)
    
    # karr
    coeff = 2 * Pi / L
    kup = np.arange(1, Nup+1, 1)
    kdn = np.arange(1, Ndn+1, 1)
    kup = (kup // 2) * (-1)**kup * coeff
    kdn = (kdn // 2) * (-1)**kdn * coeff 
    karr = np.concatenate([kup, kdn])
    karr.sort()

    # init guess of Larr TODO maybe not in this function
    Larr = (2*Pi/Nup) * np.arange(-(Nup-1)/2, (Nup+1)/2, 1)
    return Iarr, Jarr, karr, Larr 


def lieb_wu(L, Nup, Ndn, U, MaxIter=200, MaxIterL=100, MaxIterK=100, 
            e_tol=1e-10, L_tol=1e-8, k_tol=1e-8):
    '''
    Kernel.
    '''
    Iarr, Jarr, karr, Larr = gen_grids(L, Nup, Ndn)
    ThF = lambda x: -2 * np.arctan(2*x / U)
    Ne = Nup + Ndn
    E0 = 0
    for it in range(MaxIter):
        Lsav = np.zeros(Nup)
        ksav = np.zeros(Ne)
        # Loop for Larr
        for itL in range(MaxIterL):
            for k in range(Nup):
                x0 = Larr[k]
                def eq_to_solve(x):
                    left = -np.sum(ThF(2*x-2*np.sin(karr)))
                    right = -np.sum(ThF(x-Larr)) + ThF(x-Larr[k]) + 2 * Pi * Jarr[k]
                    return left - right
                Lsav[k] = find_root(eq_to_solve, x0=x0).x[0]
          
            err = np.linalg.norm(Lsav - Larr)
            if err < L_tol:
                break 
            Larr = Lsav.copy()

        # iteration over karr
        for itK in range(MaxIterK):
            for k in range(Ne):
                x0 = karr[k] 
                def eq_to_solve(x):
                    left = L * x
                    right = 2 * Pi * Iarr[k] + np.sum(ThF(2*np.sin(x) - 2*Larr[:Nup]))
                    return left - right
                ksav[k] = find_root(eq_to_solve, x0=x0).x[0]
            err = np.linalg.norm(ksav - karr)
            if err < k_tol:
                break 
            karr = ksav

        # evaluate energy
        E = -2 * np.sum(np.cos(karr))
        if abs(E - E0) < e_tol:
            print("Bethe Ansatz converged in {} iterations.".format(it+1))
            break
        E0 = E
    return E
