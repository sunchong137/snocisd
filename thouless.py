# Generate initial guesses for NOCI and RBM.

import numpy as np

def gen_thouless_random(nocc, nvir, max_nt):

    tmats = []
    for i in range(max_nt):
        t = np.random.rand(2, nvir, nocc)
        #t = np.random.normal(size=tshape)
        tmats.append(t)

    return np.asarray(tmats)

def gen_thouless_singles(nocc, nvir, max_nt=None, zmax=10, zmin=0.1):
    '''
    Generate rotations for near singly excited state for spinless systems.
    Input:
        nocc: number of occupied orbitals.
        nvir: number of virtual orbitals.
    Kwargs:
        max_nrot: maximum number of matrices to generate.
    Returns:
        A list of unnormalized Thouless parameters.
    '''

    if max_nt is None:
        max_nt = nvir * nocc

    # pick the excitations closest to the Fermi level    
    sqrt_nt = int(np.sqrt(max_nt)) + 1
    if nocc < nvir:
        if nocc < sqrt_nt: 
            d_occ = nocc 
            d_vir = nvir  
        else:
            d_occ = sqrt_nt 
            d_vir = sqrt_nt
    else:
        if nvir < sqrt_nt:
            d_occ = nocc 
            d_vir = nvir 
        else:
            d_occ = sqrt_nt 
            d_vir = sqrt_nt

    tmats = []
    t0 = np.zeros((nvir, nocc))
    k = 0
    for i in range(d_occ): # occupied
        for j in range(d_vir): # virtual
            if k == max_nt:
                break
            tm = np.ones((nvir, nocc)) * zmin 
            tm[j, nocc-i-1] = zmax
            tmats.append(np.array([tm, t0]))
            tmats.append(np.array([t0, tm]))
            k += 1
    tmats = np.asarray(tmats)
    return tmats


def gen_thouless_doubles(nocc, nvir, max_nt=None, zmax=10, zmin=0.1):
    '''
    Generate rotations for near doubly excited state for spinless systems.
    Since (i -> a, j -> b) and (j -> a, i -> b) corresponds to the same determinant,
    we do not allow cross excitation, i.e., for (i -> a, j -> b), i < j and a < b.

    '''
    if max_nt is None:
        max_nt = int(nvir*(nvir-1)/2) * int(nocc*(nocc-1)/2)
    max_nt = min(max_nt, int(nvir*(nvir-1)/2) * int(nocc*(nocc-1)/2))

    tmats = []
    k = 0
    t0 = np.zeros((nvir, nocc))
    for i in range(nocc-1): # top e occ
        for j in range(i+1, nocc): # bot e occ
            for a in range(1, nvir): # top e vir
                for b in range(a): # bot e occ
                    if k == max_nt:
                        break
                    tm = np.ones((nvir, nocc)) * zmin 
                    tm[a, nocc-i-1] = zmax
                    tm[b, nocc-j-1] = zmax # HOMO electron is further excited
                    tmats.append([tm, t0])
                    tmats.append([t0, tm])
                    k += 1
    tmats = np.asarray(tmats)
    return tmats