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

def gen_init_singles(nocc, nvir, max_nt=1, zmax=5, zmin=0.1):

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
    tmats = np.ones((max_nt, nvir, nocc)) * zmin

    k = 0
    for i in range(d_occ): # occupied
        for j in range(d_vir): # virtual
            # print(i, j)
            if k == max_nt:
                break
            tmats[k, i, j] = zmax 
            k += 1
    return tmats


def gen_thouless_random(nocc, nvir, max_nt):

    tmats = np.random.rand(max_nt, nvir, nocc) - 0.5
    return tmats

