# Generate initial guesses for NOCI and RBM.

import numpy as np

def gen_thouless_singles(nocc, nvir, max_nt=1, zmax=5, zmin=0.1):

    '''
    Generate near-single-excitation initial guesses.
    '''

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
    tmats = np.ones((max_nt, 2, nvir, nocc)) * zmin

    k = 0
    for i in range(d_occ): # occupied
        for j in range(d_vir): # virtual
            # print(i, j)
            if k == max_nt:
                break
            tmats[k, 0, i, j] = zmax 
            k += 1
    return tmats

def gen_thouless_doubles_cross(nocc, nvir, max_nt=1, zmax=5, zmin=0.1):
    
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
    if max_nt % 2 == 0:
        tmats = np.ones((max_nt, 2, nvir, nocc)) * zmin
    else:
        tmats = np.ones((max_nt+1, 2, nvir, nocc)) * zmin

    nt = 0
    for i in range(d_occ): # occupied
        for j in range(d_vir): # virtual
            for k in range(d_occ):
                for l in range(d_vir):
                    # print(i, j)
                    if nt == max_nt:
                        break
                    tmats[nt, 0, i, j] = zmax 
                    tmats[nt+1, 1, k, l] = zmax
                    nt += 2
                    
    return tmats[:max_nt]


def gen_thouless_active(nocc, nvir, max_nt=1, ncas=6, ncas_elec=None, zmax=5, zmin=0.1):
    '''
    Generate all spinless excitations in the active space.
    Args:
        ncas: the number of active orbitals.
    '''
    if ncas_elec is None: # only for spin up
        ncas_elec = int(ncas/2)

    ncas_occ = int(ncas / 2)
    ncas_vir = ncas - ncas_occ 

    # single excitation ncas_occ * ncas_vir 

    # double excitation (ncas_occ)*(ncas_occ-1)*(ncas_vir)*(ncas_vir-1)/4

    # Triple excitation a lot


# def gen_thouless_singles_allspin(nocc, nvir, max_nt=1, zmax=10, zmin=0.1):

#     '''
#     Generate near-single-excitation initial guesses.
#     '''

#     # pick the excitations closest to the Fermi level    
#     sqrt_nt = int(np.sqrt(max_nt)) + 1
#     if nocc < nvir:
#         if nocc < sqrt_nt: 
#             d_occ = nocc 
#             d_vir = nvir  
#         else:
#             d_occ = sqrt_nt 
#             d_vir = sqrt_nt
#     else:
#         if nvir < sqrt_nt:
#             d_occ = nocc 
#             d_vir = nvir 
#         else:
#             d_occ = sqrt_nt 
#             d_vir = sqrt_nt
#     tmats = np.ones((max_nt, 2, nvir, nocc)) * zmin

#     k = 0
#     max_k = int(max_nt/2)+1
#     for i in range(d_occ): # occupied
#         for j in range(d_vir): # virtual
#             if k == max_k:
#                 break
#             tmats[2*k, 0, i, j] = zmax 
#             tmats[2*k+1, 1, i, j] = zmax
#             k += 1
#     return tmats


def gen_thouless_random(nocc, nvir, max_nt):

    tmats = []
    for i in range(max_nt):
        t = np.random.rand(2, nvir, nocc) - 0.5
        #t = np.random.normal(size=tshape)
        tmats.append(t)

    return np.asarray(tmats)

# def gen_thouless_singles_all(nocc, nvir, max_nt=None, zmax=10, zmin=0.1):
#     '''
#     Generate rotations for near singly excited state for spinless systems.
#     Input:
#         nocc: number of occupied orbitals.
#         nvir: number of virtual orbitals.
#     Kwargs:
#         max_nrot: maximum number of matrices to generate.
#     Returns:
#         A list of unnormalized Thouless parameters.
#     '''

#     if max_nt is None:
#         max_nt = nvir * nocc

#     # pick the excitations closest to the Fermi level    
#     sqrt_nt = int(np.sqrt(max_nt)) + 1
#     if nocc < nvir:
#         if nocc < sqrt_nt: 
#             d_occ = nocc 
#             d_vir = nvir  
#         else:
#             d_occ = sqrt_nt 
#             d_vir = sqrt_nt
#     else:
#         if nvir < sqrt_nt:
#             d_occ = nocc 
#             d_vir = nvir 
#         else:
#             d_occ = sqrt_nt 
#             d_vir = sqrt_nt

#     tmats = []
#     t0 = np.zeros((nvir, nocc))
#     k = 0
#     for i in range(d_occ): # occupied
#         for j in range(d_vir): # virtual
#             if k == max_nt:
#                 break
#             tm = np.ones((nvir, nocc)) * zmin 
#             tm[j, nocc-i-1] = zmax
#             tmats.append(np.array([tm, t0]))
#             tmats.append(np.array([t0, tm]))
#             k += 1
#     tmats = np.asarray(tmats)
#     return tmats

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
