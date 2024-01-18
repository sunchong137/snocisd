# Copyright 2023 NOCI_Jax developers. All Rights Reserved.
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

import numpy as np

def gen_init_singles(nocc, nvir, max_nt=1, zmax=5, zmin=0.1, spin=0):

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
            tmats[k, spin, i, j] = zmax 
            k += 1
    return tmats

def gen_init_singles_onedet(nocc, nvir, idx, zmax=5, zmin=0.1, spin=0):
    '''
    Generate one quasi-single-excitation initial guess given a number.
    '''
    tmats = np.ones((1, 2, nvir, nocc)) * zmin
    i = idx // nvir 
    j = idx % nvir
    tmats[0, spin, i, j] = zmax 
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
                    if nt >= max_nt:
                        break
                    tmats[nt, 0, i, j] = zmax 
                    tmats[nt+1, 1, k, l] = zmax
                    nt += 2
                    
    return tmats[:max_nt]


def gen_thouless_random(nocc, nvir, max_nt):

    tmats = np.random.rand(max_nt, 2, nvir, nocc) - 0.5
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
                    # tmats.append([t0, tm])
                    k += 1
    tmats = np.asarray(tmats)
    return tmats
